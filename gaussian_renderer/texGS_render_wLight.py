
import torch
import math
from gstex_cuda._torch_impl import quat_to_rotmat
from gstex_cuda.texture import texture_gaussians
from gstex_cuda.texture_edit import texture_edit
from gstex_cuda.get_aabb_2d import get_aabb_2d, get_num_tiles_hit_2d, project_points
from gstex_cuda.sh import num_sh_bases, spherical_harmonics
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.sh_utils import RGB2SH, SH2RGB
from arguments import OptimizationParams
from utils.loss_utils import l1_loss, ssim, sparsity_loss, bilateral_smooth_loss

from icecream import ic


def visualize_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.cpu().detach().numpy())
    plt.show()


def calculate_loss(iteration, viewpoint_camera, pc, results, opt):
    """
    Calculate loss for the rendered image.
    """
    # Compute the loss
    loss = 0.0
    tb_dict = {}
    render_img = results['render'].permute(2,0,1)
    render_texture = results['offset_color_norm_render']
    gt_image = viewpoint_camera.original_image.cuda()
    gt_alpha = viewpoint_camera.gt_alpha_mask.squeeze().cuda()
    #rgb loss
    Ll1 = l1_loss(render_img, gt_image)
    rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_img, gt_image))
    loss += rgb_loss
    tb_dict['rgb_loss'] = rgb_loss

    if opt.offset_color_sparsity > 0:
        render_texture_norm = render_texture.norm(dim=-1)
        loss_offset_color_sparsity = sparsity_loss(render_texture_norm, gt_alpha)
        loss += opt.offset_color_sparsity * loss_offset_color_sparsity
        tb_dict['offset_sparsity_loss'] = loss_offset_color_sparsity

    return loss, tb_dict

def texGS_render_wLight(iteration, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
    override_color = None, opt: OptimizationParams = False, is_training=False, transform_dict=None, debug=False):
    results = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, transform_dict=transform_dict, is_training=is_training)
    if is_training:
        loss, tb_dict = calculate_loss(iteration, viewpoint_camera, pc, results, opt)
        results['tb_dict'] = tb_dict
        results['loss'] = loss
        
    return results

def get_uv_mapping(means, quats, mappings):
    uv0 = 0.5 * torch.ones_like(means[:,:2])
    Rs = quat_to_rotmat(quats.detach())
    ax1 = Rs[:,:,0].detach() # important!! umap and vmap are in terms of axes but that's more of an unfortunate accident than anything
    ax2 = Rs[:,:,1].detach()
    ax3 = Rs[:,:,2].detach()

    umap = mappings[:,0,None].detach() * ax1
    vmap = mappings[:,1,None].detach() * ax2

    uv0 = uv0.unsqueeze(1)
    umap = umap.unsqueeze(1)
    vmap = vmap.unsqueeze(1)
    
    return uv0, umap, vmap

def depths_to_points(depths, viewmat, c2w, intrins, H, W):
    """
    Converts depth map to points
    """
    origin = torch.zeros((4, 1), device=c2w.device)
    origin[3] = 1
    origin = (c2w @ origin).squeeze(-1)

    image_jis = torch.stack(
        torch.meshgrid(torch.arange(0, H), torch.arange(0, W)),
        dim=-1
    ).to(device=c2w.device)
    image_jis = image_jis.reshape(-1, 2)

    fx, fy, cx, cy = intrins

    ndc_x = (image_jis[:,1] - cx + 0.5) / fx
    ndc_y = (image_jis[:,0] - cy + 0.5) / fy
    rays = torch.stack((ndc_x, ndc_y, torch.ones_like(ndc_x), torch.zeros_like(ndc_x)), dim=-1) @ c2w.T
    rays = rays / (torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True)) + 1e-9)
    view_rays = rays @ viewmat.T
    rays = rays.reshape(*depths.shape[:-1], -1)
    view_rays = view_rays.reshape(*depths.shape[:-1], -1)
    # don't use view depth
    ts = depths.squeeze(-1) / view_rays[...,2]
    samples = origin[...,:3] + ts[...,None] * rays[...,:3]
    samples = samples.reshape(H, W, -1)
    return samples

def depth_to_normal(depths, viewmat, c2w, intrins, H, W):
    """
    Estimates normal map from depths
    """
    points = depths_to_points(depths, viewmat, c2w, intrins, H, W)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, is_training=False, transform_dict=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    palette_color_transforms = transform_dict.get("palette_colors")
    opacity_transforms = transform_dict.get("opacity_factors")
    light_transforms = transform_dict.get("light_transform")
    BLOCK_WIDTH = 16 # this controls the tile size of rasterization, 16 is a good default
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    
    R_inv = viewpoint_camera.world_view_transform.inverse()[:3,:3]
    T = viewpoint_camera.camera_center.unsqueeze(1)
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R_inv.device, dtype=R_inv.dtype)
    
    viewmat[:3, :3] = R_inv[:3, :3]
    viewmat[:3, 3:4] = T_inv
    c2w = viewmat.inverse()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    cx = W / 2.0
    cy = H / 2.0
    
    intrinsics = (fx, fy, cx, cy)

    means3D = pc.get_xyz
    num_points = means3D.shape[0]
    means2D = screenspace_points
    opacity = pc.get_opacity
    texture_dc = pc.texture
    mappings = pc._mappings
    quats = pc.get_rotation

    _scaling = pc._scaling
    scales = torch.zeros_like(_scaling)
    scales[:,:-1] = torch.clamp(torch.exp(_scaling[:,:-1]), min=1e-9)
    scales[:,-1] = 1e-5 * torch.mean(scales[:,:-1], dim=-1).detach()
    
    diffuse_factor = pc.get_diffuse_factor
    shininess = pc.get_shininess
    ambient_factor = pc.get_ambient_factor
    specular_factor = pc.get_specular_factor 
    normal = pc.get_normal(viewpoint_camera.world_view_transform)
    num_GSs_TF = pc._num_GSs_TF
    
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    light_dir = None
    if light_transforms is not None:
        light_dir = light_transforms.get_light_dir()
        spcular_multi, diffuse_factor_multi, ambient_multi, shininess_multi,\
        specular_offset, diffuse_factor_offset, ambient_offset, shininess_offset = light_transforms.get_light_transform()
        diffuse_factor = diffuse_factor*diffuse_factor_multi # ambient white light intensity
        shininess = shininess*shininess_multi # specular white light intensity
        ambient_factor = ambient_factor*ambient_multi # ambient white light intensity
        specular_factor = specular_factor*spcular_multi # specular white light intensity
    if light_dir is not None:
        light_dir = F.normalize(light_dir, dim=-1).contiguous()
        incidents_dirs = light_dir.detach().contiguous()
    else:
        incidents_dirs = viewdirs.detach().contiguous()
    
    
    uv0, umap, vmap = get_uv_mapping(means3D, quats, mappings)
    xys, depths = project_points(means3D, viewmat.squeeze()[:3,:], intrinsics)
    centers, extents = get_aabb_2d(means3D, scales, 1, quats, viewmat.squeeze()[:3,:], intrinsics)
    num_tiles_hit = get_num_tiles_hit_2d(centers, extents, H, W, BLOCK_WIDTH)
    total_size = texture_dc.total_size
    
    
    texture_channels = 4 if is_training else 3 # R,G,B, RGB^2 (for color sparsity loss)
    texture = torch.zeros((total_size, texture_channels), device=means3D.device)
    texture_values = texture_dc.get_texture()
    texture_num_GSs_TFs = texture_dc.num_GSs_TFs
    
    texture_dims = pc.texture_dims
    
    
    if (len(texture_num_GSs_TFs) > 1) and (len(palette_color_transforms) > 1): # for multi GS rendering
        start_texture_idx = 0
        accumulate_GS_nums = 0
        for idx, GS_nums in enumerate(texture_num_GSs_TFs):
            accumulate_GS_nums += GS_nums
            end_texture_idx = texture_dims[accumulate_GS_nums-1,2]
            texture[start_texture_idx:end_texture_idx,0:3] = texture_values[start_texture_idx:end_texture_idx,:] + palette_color_transforms[idx].palette_color.clamp(0,1).detach()
            start_texture_idx = end_texture_idx
    else: # for individual GS optimization
        texture[:,0:3] = (texture_values + palette_color_transforms[0].palette_color).clamp(0,1)
        
        
    if is_training:
        texture[:,3] = torch.norm(texture_values, dim=-1)
    
    texture_info = (num_points, 1, texture_channels)


    
    colors, extra_results, opacity = rendering_equation_BlinnPhong_python(
        palette_color_transforms, opacity_transforms, opacity, diffuse_factor, shininess, \
        ambient_factor, specular_factor, normal, viewdirs, incidents_dirs, num_GSs_TF)
    
    gaussian_rgbs = colors
    ambient_diffuse_intesnity = extra_results["ambient_diffuse_intesnity"].squeeze(-1)
    specular_intensity = extra_results["specular_intensity"].squeeze(-1)
    
    
    def custom_texture_gaussians(custom_rgbs, custom_texture, custom_opacities, custom_settings):
        custom_outputs = texture_gaussians(
            texture_info,
            texture_dims,
            centers,
            extents,
            depths,
            num_tiles_hit,
            custom_rgbs,
            custom_opacities,
            ambient_diffuse_intesnity,
            specular_intensity*0.5,
            means3D,
            scales,
            1,
            quats,
            uv0,
            umap,
            vmap,
            custom_texture,
            viewmat.squeeze()[:3, :],
            c2w,
            fx,
            fy,
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            custom_settings,
            background=torch.zeros(custom_rgbs.shape[1], device=custom_rgbs.device),
            use_torch_impl=False,
        )
        return custom_outputs
    

    all_outputs = custom_texture_gaussians(gaussian_rgbs, texture, opacity, (1<<27)|(1<<15))
    out_img, out_depth, out_reg, out_alpha, out_texture, out_normal = all_outputs[:6]

    rgb = out_texture[:,:,0:3] + (1 - out_alpha[:,:,None]) * bg_color[None,None,:]
    rgb = torch.clamp(rgb, min=0.0, max=1.0)
    offset_color_norm_render = out_texture[:,:,3] if is_training else None
    
    clean_normal = (out_normal+1)*0.5 + (1 - out_alpha[:,:,None]) * bg_color[None,None,:]
    clean_normal = torch.clamp(clean_normal, min=0.0, max=1.0)
    
    rets =  {"render": rgb,
             "offset_color_norm_render": offset_color_norm_render,
            "viewspace_points": means2D,
            "visibility_filter" : None,
            "radii": None,
            'rend_alpha': out_alpha,
            'rend_normal': clean_normal,
            'surf_depth': out_depth,
    }

    return rets



def rendering_equation_BlinnPhong_python(palette_color_transforms, opacity_transforms, opacity, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs,
                                         incidents_dirs, num_GSs_TFs):
    guassian_nums = diffuse_factor.shape[0]

    diffuse_factor = diffuse_factor.unsqueeze(-2).contiguous() # ambient white light intensity
    shininess = shininess.unsqueeze(-2).contiguous() # specular white light intensity
    ambient_factor = ambient_factor.unsqueeze(-2).contiguous() # ambient white light intensity
    specular_factor = specular_factor.unsqueeze(-2).contiguous() # specular white light intensity
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()
    incident_dirs = incidents_dirs.unsqueeze(-2).contiguous()
    diffuse_color = torch.zeros_like(normals)


    palette_colors = []
    opacity_factors = []
    for i in range(len(palette_color_transforms)):
        palette_colors.append(palette_color_transforms[i].palette_color.clamp(0,1))
        if opacity_transforms is not None:
            opacity_factors.append(opacity_transforms[i].opacity_factor)
    if len(num_GSs_TFs) > 1: # for multi GS rendering
        start_GS = 0
        for i in range(len(num_GSs_TFs)):
            end_GS = num_GSs_TFs[i] + start_GS
            if opacity_transforms is not None:
                opacity[start_GS:end_GS,:] = opacity[start_GS:end_GS,:] * opacity_factors[i]
            start_GS = end_GS
    else: # for individual GS optimization
        pass

    # diffuse color
    cos_l = (normals*incident_dirs).sum(dim=-1, keepdim=True)
    
    diffuse_intensity = diffuse_factor*torch.abs(cos_l)
    diffuse_color = (ambient_factor+diffuse_intensity).repeat(1, 1, 3)

    # specular color
    h = F.normalize(incident_dirs + viewdirs, dim=-1) # bisector h
    cos_h = (normals*h).sum(dim=-1, keepdim=True)
    specular_intensity = specular_factor*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(shininess), 0.0)

    specular_color = specular_intensity.repeat(1, 1, 3)

    pbr = diffuse_color.squeeze() + specular_color.squeeze()
    pbr = torch.clamp(pbr, 0., 1.)
    
    ambient_diffuse_intesnity = (ambient_factor+diffuse_intensity).clamp(0,1)
    specular_intensity = specular_intensity.clamp(0,1)

    extra_results = {
    "ambient_diffuse_intesnity": ambient_diffuse_intesnity,
    "specular_intensity": specular_intensity, 
    }

    return pbr, extra_results, opacity