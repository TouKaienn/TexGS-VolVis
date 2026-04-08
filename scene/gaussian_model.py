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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.normal_utils import compute_normal_world_space

from scene.jagged_texture import JaggedTexture

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        self.diffuse_factor_activation = torch.sigmoid
        self.shininess_activation = torch.sigmoid
        self.ambient_activation = torch.sigmoid
        self.specular_factor_activation = torch.sigmoid


    def __init__(self, sh_degree : int):
        self.useTexGS = False
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        

        self.texture = None
        self.useTexGS = False
        
        self._diffuse_factor = torch.empty(0)
        self._shininess = torch.empty(0)
        self._ambient_factor = torch.empty(0)
        self._specular_factor = torch.empty(0)

        self._num_GSs_TF = [-1]
        
    def set_num_GSs_TF(self, num_GSs_TF):
        self._num_GSs_TF = num_GSs_TF

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def initTexGSfrom2DGS(self, init_2DGS_path, training_args, init_color=None):
        self.load_ply(init_2DGS_path, isTexGS=False)
        
        self.pixel_num = training_args.pixel_num
        self.settings = training_args.settings
        self.sigma_factor = training_args.sigma_factor 
        self.pixel_scale = 10.0 * torch.ones(1, dtype=torch.float32)
        self._scaling = torch.nn.Parameter(torch.cat([self._scaling, torch.ones(self._scaling.shape[0], 1, device=self._scaling.device, dtype=self._scaling.dtype)], dim=-1))
        self.useTexGS = True
        self.num_points = self._xyz.shape[0]
        
        self.texture_dims = torch.ones((self.num_points, 3), dtype=torch.int32, device="cuda")
        hws = self.texture_dims[:,0] * self.texture_dims[:,1]
        self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws
        self.texture = JaggedTexture(self.texture_dims, out_dim=3).to(self.texture_dims.device)
        if init_color is not None:
            self.texture.texture.data = torch.ones_like(self.texture.texture.data)*init_color.detach().clone().to(self.texture.texture.device)
        temp_mappings = torch.ones((self.num_points, 2))
        self._mappings=torch.nn.Parameter(temp_mappings).to(self.texture_dims.device)
        self.build_charts()


        self.training_setup(training_args)  
        
        self.texture_optimizer_idx = self.get_texture_param_group_idx()
    
    
       
    def get_normal(self,viewmat):
        quats = self.get_rotation
        scaling = self.get_scaling
        normal = compute_normal_world_space(quats, scaling)
        return normal


    @property
    def get_diffuse_factor(self):
        return self._diffuse_factor

    @property
    def get_shininess(self):
        return self.shininess_activation(self._shininess)*50

    @property
    def get_ambient_factor(self):
        return self._ambient_factor

    @property
    def get_specular_factor(self):
        return self.specular_factor_activation(self._specular_factor)*5
    
    @property
    def get_texture(self):
        return self.texture.texture
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_gaussian_nums(self):
        return self._xyz.shape[0]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_num_GSs_TFs(self):
        return self._num_GSs_TF
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self._diffuse_factor = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))
        self._shininess = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))
        self._ambient_factor = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))
        self._specular_factor = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._diffuse_factor], 'lr': training_args.diffuse_lr, "name": "diffuse_factor"},
            {'params': [self._shininess], 'lr': training_args.shininess_lr, "name": "shininess"},
            {'params': [self._ambient_factor], 'lr': training_args.ambient_lr, "name": "ambient_factor"},
            {'params': [self._specular_factor], 'lr': training_args.specular_lr, "name": "specular_factor"}
        ]
        
        if self.useTexGS:
            l.extend([
                {'params': [self.texture.texture], 'lr': training_args.texture_lr, "name": "texture"},
            ])
        
        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        
    
    def get_texture_param_group_idx(self):
        for i, group in enumerate(self.optimizer.param_groups):
            if group["name"] == "texture":
                return i

    def get_texture_param_groups(self):
        params = {
            "texture": [self.texture.texture],
        }
        return params
    
    
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        if not self.useTexGS:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._normal.shape[1]): 
            l.append('normal_{}'.format(i)) # normal_0, normal_1, normal_2
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('diffuse_factor')
        l.append('shininess')
        l.append('ambient_factor')
        l.append('specular_factor')
        if self.useTexGS:
            for i in range(self._mappings.shape[1]):
                l.append('mappings_{}'.format(i))
        return l

    def build_charts(self, update_pixel_scale=True):
        total_pixel_num = self.pixel_num
        self.num_points = self._xyz.shape[0]
        with torch.no_grad():
            sigma_factor = self.sigma_factor
            length0 = torch.exp(self._scaling[:,0])
            length1 = torch.exp(self._scaling[:,1])
            def get_score(x):
                score = torch.sum(torch.ceil(sigma_factor * length0 / x) * torch.ceil(sigma_factor * length1 / x)).item()
                return score

            self.pixel_scales = torch.ones_like(length0)
            adjustments = torch.ones_like(length0)            
            adjustments = torch.sqrt(adjustments**2 / torch.mean(adjustments**2))

            if update_pixel_scale:
                lo = 10.0
                hi = np.sqrt(torch.sum(sigma_factor * sigma_factor * length0 * length1 * (adjustments ** 2)).item() / total_pixel_num)
                mid = 0.5 * (lo + hi)
                score = get_score(mid)
                iter_num = 0
                tol = 1e-3
                while score < (1-tol) * total_pixel_num or score > (1+tol) * total_pixel_num:
                    if score < (1-tol) * total_pixel_num:
                        lo = mid
                    else:
                        hi = mid
                    mid = 0.5 * (lo + hi)
                    score = get_score(mid / adjustments)
                    iter_num += 1
                    if iter_num > 30:
                        break
                self.pixel_scale[0] = mid
                self.pixel_scales = mid / adjustments


            self.texture_dims = torch.zeros(self.num_points, 3, dtype=self.texture_dims.dtype, device=self.texture_dims.device)
            self.texture_dims[:,0] = torch.ceil(sigma_factor * length0 / self.pixel_scales)
            self.texture_dims[:,1] = torch.ceil(sigma_factor * length1 / self.pixel_scales)
            hws = self.texture_dims[:,0] * self.texture_dims[:,1]
            self.texture_dims[:,2] = torch.cumsum(hws, dim=0) - hws

            self._mappings[:,0] = 1 / (2.0 * sigma_factor * length0)
            self._mappings[:,1] = 1 / (2.0 * sigma_factor * length1)

            self.texture.init_from_dims(self.texture_dims)
            self.edit_texture = None
            self.update_edit_texture()
    
    def reshape_in_all_optim(self):
        param_groups = self.get_texture_param_groups()
        for group, param in param_groups.items():
            self.reshape_in_optim(param)
        torch.cuda.empty_cache()


    def reshape_in_optim(self, new_params):
        if len(new_params) != 1:
            return
        assert len(new_params) == 1
        assert isinstance(self.optimizer, torch.optim.Adam), "Only works with Adam"

        param = self.optimizer.param_groups[self.texture_optimizer_idx]["params"][0]
        param_state = self.optimizer.state[param]

        if param in self.optimizer.state:
            param_state = self.optimizer.state[param]
            del self.optimizer.state[param]
            
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.zeros_like(new_params[0].data)
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.zeros_like(new_params[0].data)
            
            del self.optimizer.param_groups[self.texture_optimizer_idx]["params"][0]
            del self.optimizer.param_groups[self.texture_optimizer_idx]["params"]
            self.optimizer.param_groups[self.texture_optimizer_idx]["params"] = new_params

            self.optimizer.state[new_params[0]] = param_state

        else:
            del self.optimizer.param_groups[self.texture_optimizer_idx]["params"][0]
            del self.optimizer.param_groups[self.texture_optimizer_idx]["params"]
            self.optimizer.param_groups[self.texture_optimizer_idx]["params"] = new_params
    
    def update_edit_texture(self):
        edit_texture = torch.clone(SH2RGB(self.texture.get_texture()))
        self.edit_texture = edit_texture

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        if not self.useTexGS:
            xyz = self._xyz.detach().cpu().numpy()
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            normals = self._normal.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            
            diffuse_factor = self._diffuse_factor.detach().cpu().numpy()
            shininess = self._shininess.detach().cpu().numpy()
            ambient_factor = self._ambient_factor.detach().cpu().numpy()
            specular_factor = self._specular_factor.detach().cpu().numpy()

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, f_dc, f_rest, opacities,normals, scale, rotation, diffuse_factor, shininess, ambient_factor, specular_factor), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        else:
            xyz = self._xyz.detach().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            normals = self._normal.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            
            diffuse_factor = self._diffuse_factor.detach().cpu().numpy()
            shininess = self._shininess.detach().cpu().numpy()
            ambient_factor = self._ambient_factor.detach().cpu().numpy()
            specular_factor = self._specular_factor.detach().cpu().numpy()
            mappings = self._mappings.detach().cpu().numpy()
            
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, opacities, normals, scale, rotation, diffuse_factor, shininess, ambient_factor, specular_factor, mappings), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
    
    def save_texture_npz(self, path):
        mkdir_p(os.path.dirname(path))
        param_dict = {
            "texture_dc": self.texture.texture,
            "texture_dims": self.texture_dims,
            "mappings": self._mappings
        }
        np_dict = {
            k: param_dict[k].detach().cpu().numpy() for k in param_dict
        }
        np.savez(path, **np_dict)
    

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, isTexGS=False):
        if isTexGS: self.useTexGS = True

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        normal = np.stack((np.asarray(plydata.elements[0]["normal_0"]),
                           np.asarray(plydata.elements[0]["normal_1"]),
                           np.asarray(plydata.elements[0]["normal_2"])), axis=1)
        self.num_points = xyz.shape[0]
        if not self.useTexGS:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        if not self.useTexGS:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(torch.zeros((xyz.shape[0], 3, 1), dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_rest = nn.Parameter(torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        diffuse_factor = np.asarray(plydata.elements[0]["diffuse_factor"])[..., np.newaxis]
        shininess = np.asarray(plydata.elements[0]["shininess"])[..., np.newaxis]
        ambient_factor = np.asarray(plydata.elements[0]["ambient_factor"])[..., np.newaxis]
        specular_factor = np.asarray(plydata.elements[0]["specular_factor"])[..., np.newaxis]
        
        self._diffuse_factor = nn.Parameter(torch.tensor(diffuse_factor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shininess = nn.Parameter(torch.tensor(shininess, dtype=torch.float, device="cuda").requires_grad_(True))
        self._ambient_factor = nn.Parameter(torch.tensor(ambient_factor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular_factor = nn.Parameter(torch.tensor(specular_factor, dtype=torch.float, device="cuda").requires_grad_(True))

        
        if self.useTexGS:
            self.texture_dims = torch.ones((self.num_points, 3), dtype=torch.int32, device="cuda")
            self.texture = JaggedTexture(self.texture_dims, out_dim=3).to(self.texture_dims.device) 
            self.texture.texture.data = self._features_dc.data[:,0,:].clone().to(self.texture.texture.device)
            self._mappings=torch.nn.Parameter(torch.ones((self.num_points, 2))).to(self.texture_dims.device)

            mappings_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("mappings_")]
            mappings_names = sorted(mappings_names, key = lambda x: int(x.split('_')[-1]))
            mappings = np.zeros((xyz.shape[0], len(mappings_names)))
            for idx, attr_name in enumerate(mappings_names):
                mappings[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            self.load_texture_npz(os.path.dirname(path) + "/texture.npz")
    
    def load_jagged_texture(self, jagged_texture, mappings_list):
        self.texture = jagged_texture
        self.texture_dims = self.texture.texture_dims

        mappings = torch.cat(mappings_list, dim=0)
        self._mappings = nn.Parameter(mappings.to(self.texture_dims.device))
        
    
    def load_texture_npz(self, path):
        npz_data = np.load(path)
        self.texture.texture.data = torch.tensor(npz_data["texture_dc"], device="cuda")
        self.texture_dims = torch.tensor(npz_data["texture_dims"], device="cuda") # [ N, 3]
        self.texture.texture_dims = self.texture_dims
        self._mappings.data = torch.tensor(npz_data["mappings"], device="cuda") # [N, 2]
        self.texture.update_total_size()
        
    @property
    def non_texture_attribute(self):
        attribute_names = ['xyz', 'scaling', 'rotation', 'opacity', 'normal', 'diffuse_factor', 'shininess', "ambient_factor", "specular_factor"]
        return attribute_names
    
    @classmethod
    def select_from_gaussians(cls, compose_gaussian, indices):
        assert len(indices) > 0
        sh_degree = compose_gaussian.max_sh_degree
        gaussians = GaussianModel(sh_degree)
        non_texture_attributes = gaussians.non_texture_attribute
        for attribute_name in non_texture_attributes:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(compose_gaussian, "_" + attribute_name).data[indices]],
                                           dim=0).requires_grad_(True)))
        return gaussians
    
    @classmethod
    def create_from_gaussians(cls, gaussians_list):
        assert len(gaussians_list) > 0
        sh_degree = max(g.max_sh_degree for g in gaussians_list)
        gaussians = GaussianModel(sh_degree=sh_degree)
        non_texture_attributes = gaussians.non_texture_attribute
        num_GSs_TF = [g.get_gaussian_nums for g in gaussians_list]
        for attribute_name in non_texture_attributes:
            
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))
        gaussians.set_num_GSs_TF(num_GSs_TF)
        return gaussians
        

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        
        self._diffuse_factor = optimizable_tensors["diffuse_factor"]
        self._shininess = optimizable_tensors["shininess"]
        self._ambient_factor = optimizable_tensors["ambient_factor"]
        self._specular_factor = optimizable_tensors["specular_factor"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,new_normals, new_scaling, new_rotation, \
        new_diffuse_factor, new_shininess, new_ambient_factor, new_specular_factor):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        'normal': new_normals,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "diffuse_factor": new_diffuse_factor,
        "shininess": new_shininess,
        "ambient_factor": new_ambient_factor,
        "specular_factor": new_specular_factor}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._diffuse_factor = optimizable_tensors["diffuse_factor"]
        self._shininess = optimizable_tensors["shininess"]
        self._ambient_factor = optimizable_tensors["ambient_factor"]
        self._specular_factor = optimizable_tensors["specular_factor"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_normals = self._normal[selected_pts_mask].repeat(N,1)

        new_diffuse_factor = self._diffuse_factor[selected_pts_mask].repeat(N, 1)
        new_shininess = self._shininess[selected_pts_mask].repeat(N, 1)
        new_ambient_factor = self._ambient_factor[selected_pts_mask].repeat(N, 1)
        new_specular_factor = self._specular_factor[selected_pts_mask].repeat(N, 1)
        
        args = [new_xyz, new_features_dc, new_features_rest, new_opacity, new_normals, new_scaling, new_rotation, \
            new_diffuse_factor, new_shininess, new_ambient_factor, new_specular_factor]
        
        self.densification_postfix(*args)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_normals = self._normal[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        new_diffuse_factor = self._diffuse_factor[selected_pts_mask]
        new_shininess = self._shininess[selected_pts_mask]
        new_ambient_factor = self._ambient_factor[selected_pts_mask]
        new_specular_factor = self._specular_factor[selected_pts_mask] 
        
        args = [new_xyz, new_features_dc, new_features_rest, new_opacities, new_normals, new_scaling, new_rotation, \
            new_diffuse_factor, new_shininess, new_ambient_factor, new_specular_factor]  

        self.densification_postfix(*args)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def freeze_attributes(self):
        self._xyz.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._normal.requires_grad_(False)

        self._diffuse_factor.requires_grad_(False)
        self._shininess.requires_grad_(False)
        self._ambient_factor.requires_grad_(False)
        self._specular_factor.requires_grad_(False)

    def send_to_cpu(self):
        self._xyz = self._xyz.cpu()
        self._features_dc = self._features_dc.cpu()
        self._features_rest = self._features_rest.cpu()
        self._opacity = self._opacity.cpu()
        self._normal = self._normal.cpu()
        self._scaling = self._scaling.cpu()
        self._rotation = self._rotation.cpu()
        
        self._diffuse_factor = self._diffuse_factor.cpu()
        self._shininess = self._shininess.cpu()
        self._ambient_factor = self._ambient_factor.cpu()
        self._specular_factor = self._specular_factor.cpu()

        if self.useTexGS:
            self._mappings = self._mappings.cpu()
    
    def send_to_gpu(self):
        self._xyz = self._xyz.cuda()
        self._features_dc = self._features_dc.cuda()
        self._features_rest = self._features_rest.cuda()
        self._opacity = self._opacity.cuda()
        self._normal = self._normal.cuda()
        self._scaling = self._scaling.cuda()
        self._rotation = self._rotation.cuda()
        
        self._diffuse_factor = self._diffuse_factor.cuda()
        self._shininess = self._shininess.cuda()
        self._ambient_factor = self._ambient_factor.cuda()
        self._specular_factor = self._specular_factor.cuda()

        if self.useTexGS:
            self._mappings = self._mappings.cuda()