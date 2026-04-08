import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from gstex_cuda.texture_sample import texture_sample
from gstex_cuda._torch_impl import quat_to_rotmat, normalized_quat_to_rotmat, sample_texture

def texture_dims_to_int_coords(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs, hws, dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = local_idxs // query_dims[:,1]
    vv = local_idxs % query_dims[:,1]
    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv

def texture_dims_to_query(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs, hws, dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = (local_idxs // query_dims[:,1]).float() / query_dims[:,0].float()
    vv = (local_idxs % query_dims[:,1]).float() / query_dims[:,1].float()
    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv

class JaggedTexture(nn.Module):
    #! If you get wired error such as: CUDA illegal memory access
    #! Make sure the total_size is correct (should be equal to the sum of texture params)
    def __init__(self, texture_dims, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.texture = nn.Parameter(torch.zeros(1,self.out_dim), requires_grad=True)
        self.num_GSs_TFs = [-1]
        self.register_buffer("texture_dims", texture_dims)
        self.init_from_dims(texture_dims, initial=True)
    
    def get_texture(self):
        return self.texture[:self.total_size,:]
    
    def reset(self):
        with torch.no_grad():
            self.texture.data = torch.zeros_like(self.texture.data)

    def adjust_texture_size(self, new_size):
        diff = new_size - self.texture.shape[0]
        if diff <= 0:
            return
        self.texture = nn.Parameter(
            torch.cat(
                [
                    self.texture.detach(),
                    torch.zeros((diff, self.texture.shape[1]), dtype=self.texture.dtype, device=self.texture.device)
                ], dim=0
            )
        )

    def cull(self, cull_mask):
        idxs = torch.arange(self.texture_dims.shape[0], dtype=torch.int64, device=self.texture_dims.device)
        old_hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        ids = torch.repeat_interleave(idxs, old_hws, dim=0)
        texture_cull_mask = cull_mask[ids]
        new_size = self.total_size - texture_cull_mask.sum().item()
        self.adjust_texture_size(new_size)
        with torch.no_grad():
            self.texture.data[:new_size,:] = self.texture.data[:self.total_size,:][~texture_cull_mask,:]
        
        self.texture_dims = self.texture_dims[~cull_mask]
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        self.total_size = torch.sum(self.texture_dims[:,0] * self.texture_dims[:,1]).item()
        assert self.total_size == new_size

    def dup_and_split(self, dup_mask, split_mask, samps):
        idxs = torch.arange(self.texture_dims.shape[0], dtype=torch.int64, device=self.texture_dims.device)
        old_hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        ids = torch.repeat_interleave(idxs, old_hws, dim=0)
        texture_dup_mask = dup_mask[ids]
        texture_split_mask = split_mask[ids]

        new_size = self.total_size + texture_dup_mask.sum().item() + samps * texture_split_mask.sum().item()
        self.adjust_texture_size(new_size)

        with torch.no_grad():
            texture = self.get_texture().data
            self.texture.data[:new_size,:] = torch.cat(
                [
                    texture,
                    texture[texture_dup_mask,:],
                    texture[texture_split_mask,:].repeat(samps, 1),
                ],
                dim=0
            )
    
        self.texture_dims = torch.cat(
            [
                self.texture_dims,
                self.texture_dims[dup_mask],
                self.texture_dims[split_mask].repeat(samps, 1),
            ],
            dim=0
        )
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        self.total_size = torch.sum(self.texture_dims[:,0] * self.texture_dims[:,1]).item()
        assert self.total_size == new_size

    def init_from_dims(self, texture_dims, initial=False):
        if not initial:
            assert texture_dims.shape[0] == self.texture_dims.shape[0]
        
        total_size = torch.sum(texture_dims[:,0] * texture_dims[:,1]).item()
        with torch.no_grad():
            if initial:
                self.max_size = total_size
                self.texture = nn.Parameter(torch.zeros(total_size, self.out_dim, device=self.texture.device))
            else:
                idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
                hws = texture_dims[:,0] * texture_dims[:,1]
                ids = torch.repeat_interleave(idxs, hws, dim=0)
                self.texture_dims = self.texture_dims.to(texture_dims.device)
                query_dims = self.texture_dims[ids,:]
                _, uv = texture_dims_to_query(texture_dims)
                texture_info = (1, 1, self.texture.shape[-1])
                use_torch_impl = False
                if "cuda" not in str(self.texture.device):
                    use_torch_impl = True
                new_texture = texture_sample(texture_info, query_dims, self.get_texture(), uv, use_torch_impl=use_torch_impl)
                self.adjust_texture_size(total_size)
                self.texture.data[:total_size,:] = new_texture.detach()
        
        self.texture_dims = texture_dims
        self.total_size = total_size
        
    @classmethod
    def select_texture_from_indices(self, texture, indices):
        out_dim = texture.out_dim
        all_texture = texture.get_texture()
        texture_dims = texture.texture_dims
        selected_texture_dims = texture_dims[indices, :]
        selected_texture_dims_1d = torch.zeros((selected_texture_dims.shape[0], 2), dtype=torch.int32, device=texture_dims.device)
        selected_texture_dims_1d[:,0], selected_texture_dims_1d[:,1] = selected_texture_dims[:,0] * selected_texture_dims[:,1], selected_texture_dims[:,2]
        selected_texture_indices = torch.cat([torch.arange(start, start + length, device='cuda') for length, start in selected_texture_dims_1d])
        new_texture_data = all_texture[selected_texture_indices, :]
        new_texture_dims = selected_texture_dims
        new_texture_dims[:,2] = torch.cumsum(new_texture_dims[:,0] * new_texture_dims[:,1], dim=0) - new_texture_dims[:,0] * new_texture_dims[:,1]
        new_texture = JaggedTexture(new_texture_dims, out_dim=out_dim)
        new_texture.texture.data = new_texture_data
        return new_texture
    
    @classmethod
    def create_from_textures(self, textures, num_GSs_TFs):
        assert len(textures) > 0
        texture_dims = torch.cat([t.texture_dims for t in textures], dim=0)
        hws = texture_dims[:,0] * texture_dims[:,1]
        texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        compose_texture = JaggedTexture(texture_dims, out_dim=textures[0].out_dim)
        compose_texture.texture.data = torch.cat([t.get_texture().data for t in textures], dim=0)
        compose_texture.num_GSs_TFs = num_GSs_TFs
        return compose_texture
    
    def update_total_size(self):
        self.total_size = torch.sum(self.texture_dims[:,0] * self.texture_dims[:,1]).item()

    def forward(self, x, query_dims, detached=False):
        y = torch.clamp(x, min=0.0, max=1.0)

        if detached:
            texture = self.get_texture().detach()
        else:
            texture = self.get_texture()

        y = sample_texture(query_dims, texture, y)
        return y

if __name__ == "__main__":
    num_gaussians = 12_000
    texture_dims = torch.ones(num_gaussians, 3, dtype=torch.int32)
    hws = texture_dims[:,0] * texture_dims[:,1]
    texture_dims[:, 2] = torch.cumsum(hws, dim=0) - hws
    texture = JaggedTexture(texture_dims, out_dim=5)
    print(texture.total_size)
    print(texture.get_texture().shape)