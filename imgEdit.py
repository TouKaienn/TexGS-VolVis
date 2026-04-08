#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from scene.opacity_trans import LearningOpacityTransform
from scene.palette_color import LearningPaletteColor
from scene.gamma_trans import LearningGammaTransform
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from utils.mesh_utils import GaussianExtractor
from utils.general_utils import PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision import transforms
from utils.multiviewEdit import MultiViewsEditor
import random
from torchvision.utils import save_image, make_grid
import numpy as np
from utils.nnfm_loss import NNFMLoss
from PIL import Image
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from icecream import ic

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, transform_dict):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(iteration, viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=True, transform_dict=transform_dict)

            visualization_list = [
                render_pkg["render"].permute(2, 0, 1),
                viewpoint_cam.original_image.cuda(),
            ]

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=2)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

@torch.no_grad()
def training_report(tb_writer, iteration, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, transform_dict=None):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    render_pkg = renderFunc(iteration, viewpoint, scene.gaussians, *renderArgs, transform_dict=transform_dict)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                 
                    if image.shape[2] == 3:
                        image = image.permute(2, 0, 1)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def visualize_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.cpu().detach().numpy())
    plt.show()

def save_img(img, path):
    img = torch.clamp(img, 0.0, 1.0)
    import matplotlib.pyplot as plt
    plt.imsave(path, img.permute(2,1,0).cpu().detach().numpy())

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, init_2DGS_path, partialEditing=False):
    useTexGS = True
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    
    

    """Initialize NNFM image editing loss"""
    gt_img_size = scene.getTrainCameras()[0].original_image#.size()
    nnfm_loss_fn = NNFMLoss(device="cuda")
    style_img = Image.open(args.style_img_path).resize(gt_img_size.shape[1:])
    style_img = PILtoTorch(style_img, style_img.size).to("cuda")
    style_img = style_img.permute(0,2,1).unsqueeze(0)
    
    unresized_style_img = Image.open(args.style_img_path)
    unresized_style_img = PILtoTorch(unresized_style_img, unresized_style_img.size).to("cuda")
    unresized_style_img = unresized_style_img.permute(0,2,1).unsqueeze(0)

    nnfm_loss_fn.set_style_img(style_img, unresized_style_img)
    
    # initialize texture with the mean color of the style image
    mean_color = style_img.mean(dim=(2,3))
        
    """
    Setup transform dict
    """
    transform_dict = dict()
    if (init_2DGS_path) and (useTexGS):
        if not partialEditing:
            gaussians.initTexGSfrom2DGS(init_2DGS_path, opt, init_color=mean_color)
        else:
            gaussians.load_ply(init_2DGS_path, isTexGS=True)
            gaussians.texture.texture.data = mean_color*torch.ones_like(gaussians.texture.texture.data)
            gaussians.training_setup(opt)

        palette_color_transforms = []
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.load_palette_color(args.source_path+'/train')
        palette_color_transform.training_setup(opt)
        palette_color_transforms.append(palette_color_transform)
        transform_dict["palette_colors"] = palette_color_transforms
        
        opacity_transforms = []
        opacity_transform = LearningOpacityTransform(opacity_factor=1.0)
        opacity_transforms.append(opacity_transform)
        transform_dict["opacity_factors"] = opacity_transforms
        
        gaussians.freeze_attributes()
    
        

    
    

    """ Render fn """
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
            
        # # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render_fn(iteration, viewpoint_cam, gaussians, pipe, background, \
            is_training=True, opt=opt, transform_dict=transform_dict, text_editing=False, img_editing=True, nnfm_loss_fn=nnfm_loss_fn)

        total_loss = render_pkg["loss"]
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            save_training_vis(viewpoint_cam, gaussians, background, render_fn_dict['stylize_inf'],
                                  pipe, opt, first_iter, iteration, transform_dict)


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration % opt.build_chart_every==0) and (not partialEditing):
                gaussians.build_charts()
                gaussians.reshape_in_all_optim()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or (iteration == opt.iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                for com_name, component in transform_dict.items():
                    if com_name == "palette_colors":
                        torch.save((component[0].capture(), iteration),
                                os.path.join(scene.model_path, 'point_cloud', f'iteration_{iteration}', f"{com_name}_chkpnt" + ".pth"))
                    
                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))
                
    if dataset.eval:
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussExtractor = GaussianExtractor(gaussians, render_fn_dict['stylize_inf'], pipe, bg_color=bg_color, transform_dict=transform_dict)
        os.makedirs(scene.model_path+f"/{args.edit_name}", exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(scene.model_path+f"/{args.edit_name}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--edit_name', type=str, default=None, required=True)
    parser.add_argument('--style_img_path', type=str, default=None)
    parser.add_argument("--partialEditing", action="store_true", default=False, help="Doing partial editing")

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument('-t', '--type', choices=['2DGS', 'TexGS', 'stylize'], default='stylize')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-init", "--init_TexGS_path", type=str, default = None)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.init_TexGS_path, partialEditing=False)

    # All done
    print("\nTraining complete.")