# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import sys
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
import math
import torch
from rich.console import Console
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
CONSOLE = Console(width=120)
from typing import Optional
from diffusers import (
    DDIMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from transformers import logging

from utils.dge_utils import make_dge_block, register_extended_attention, isinstance_str,\
    register_normal_attention, register_normal_attn_flag, register_pivotal, register_batch_idx, \
        register_cams, compute_epipolar_constrains, register_epipolar_constrains
from diffusers.utils.import_utils import is_xformers_available
logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class MultiViewsEditor(nn.Module):
    """MultiViewsEditor implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False, enable_memory_efficient_atten=True) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.weights_dtype = (
            torch.float16 if not ip2p_use_full_precision else torch.float32
        )

        self.ip2p_use_full_precision = ip2p_use_full_precision
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": "./cache",
        }
        self.diffusion_steps = 20
        self.camera_batch_size = 6
        
        
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, **pipe_kwargs)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler",\
            torch_dtype=pipe_kwargs['torch_dtype'], cache_dir=pipe_kwargs["cache_dir"])
        
        pipe.scheduler.set_timesteps(self.diffusion_steps)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe
        

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        self.pipe.unet.eval()
        self.pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            self.pipe.unet.float()
            self.pipe.vae.float()
        else:
            if self.device.index:
                self.pipe.enable_model_cpu_offload(self.device.index)
            else:
                self.pipe.enable_model_cpu_offload(0)
        
        self.pipe.enable_vae_tiling()
        if (enable_memory_efficient_atten) and is_xformers_available():
            print("Enable memory efficient attention")
            self.pipe.enable_xformers_memory_efficient_attention()

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
            
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.grad_clip_val: Optional[float] = None

        CONSOLE.print("MultiViewsEditor loaded!")

        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
        register_extended_attention(self)
        

    @torch.cuda.amp.autocast(enabled=False) # do not use autocast for this function
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)
        
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def use_normal_unet(self):
        register_normal_attention(self)
        register_normal_attn_flag(self.unet, True)
    
    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        guidance_scale,
        image_guidance_scale,
        t: Int[Tensor, "B"],
        cams= None,
        
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]
        camera_batch_size = self.camera_batch_size
        print("Start editing images...")

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            for t in self.scheduler.timesteps:
                if t < 100:
                    self.use_normal_unet()
                else:
                    register_normal_attn_flag(self.unet, False)
                with torch.no_grad():
                    # pred noise
                    noise_pred_text = []
                    noise_pred_image = []
                    noise_pred_uncond = []
                    pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size)
                    register_pivotal(self.unet, True)
                    
                    key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                    latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
                    pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                    pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                    self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                    register_pivotal(self.unet, False)

                    for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                        register_batch_idx(self.unet, i)
                        register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
                        epipolar_constrains = {}
                        for down_sample_factor in [1, 2, 4, 8]:
                            H = current_H // down_sample_factor
                            W = current_W // down_sample_factor
                            epipolar_constrains[H * W] = []
                            for cam in cams[b:b + camera_batch_size]:
                                cam_epipolar_constrains = []
                                for key_cam in key_cams:
                                    cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                                epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                            epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                        register_epipolar_constrains(self.unet, epipolar_constrains)

                        batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                        batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                        batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                        batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        noise_pred_text.append(batch_noise_pred_text)
                        noise_pred_image.append(batch_noise_pred_image)
                        noise_pred_uncond.append(batch_noise_pred_uncond)

                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                    # perform classifier-free guidance
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # get previous sample, continue loop
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents


    def edit_image(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        img_mask: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        cams= None,
        guidance_scale=7.5,
        image_guidance_scale=1.5
    ):
        assert cams is not None, "cams is required for dge guidance"
        batch_size, H, W, _ = rgb.shape
        factor = IMG_DIM / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        img_mask_BCHW = img_mask.permute(0, 3, 1, 2)

        RH, RW = height, width

        
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)
        
        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)


        edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, \
            guidance_scale, image_guidance_scale, t, cams)
        edit_images = self.decode_latents(edit_latents)
        edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")
        
        edit_images = edit_images * img_mask_BCHW  + rgb_BCHW_HW8 * (1 - img_mask_BCHW)

        return edit_images

    def visualize_image(self, image: Float[Tensor, "BS 3 H W"]) -> None:
        """Visualize the edited image
        Args:
            image: Image tensor to visualize
        """
        image = image.cpu().detach().numpy().transpose(0, 2, 3, 1)
        plt.imshow(image[0])
        plt.axis('off')
        plt.show()
    
def config_to_primitive(config, resolve: bool = True) -> Any:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(config, resolve=resolve)

def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


if __name__ == "__main__":
    # Test the module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewsEditor(device=device)
