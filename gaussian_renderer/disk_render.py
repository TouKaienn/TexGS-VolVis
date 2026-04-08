
import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from arguments import OptimizationParams
from utils.loss_utils import l1_loss, ssim, sparsity_loss, bilateral_smooth_loss
from gstex_cuda._torch_impl import quat_to_rotmat
import torch.nn.functional as F


def get_disk_normal(quants):
    """
    Get the normal of the disk.
    """
    Rs = quat_to_rotmat(quants)
    normal = Rs[:,:,2]
    return normal

def calculate_loss(iteration, viewpoint_camera, pc, results, opt, light_aware_2DGS):
    """
    Calculate loss for the rendered image.
    """
    # Compute the loss
    loss = 0.0
    tb_dict = {}
    render_img = results['render']
    rend_dist = results["rend_dist"]
    rend_normal  = results['rend_normal']
    surf_normal = results['surf_normal']
    gt_image = viewpoint_camera.original_image.cuda()
    gt_alpha = viewpoint_camera.gt_alpha_mask.permute(2,0,1).cuda()
    
    # regularization
    lambda_normal = opt.lambda_normal if (iteration > 7000) else 0.0
    lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
    
    #rgb loss
    Ll1 = l1_loss(render_img, gt_image)
    rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_img, gt_image))
    loss += rgb_loss
    tb_dict['rgb_loss'] = rgb_loss
        
    # Normal consistency loss
    if lambda_normal > 0:
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        tb_dict['normal_loss'] = normal_loss
        loss += normal_loss
    
    # Depth distortion loss
    if lambda_dist > 0:
        dist_loss = lambda_dist * (rend_dist).mean()
        tb_dict['dist_loss'] = dist_loss
        loss += dist_loss
    
    if opt.lambda_opacity > 0:
        rendered_opacity = results['rend_alpha']
        Ll1_opacity = F.l1_loss(rendered_opacity, gt_alpha).item()
        ssim_val_opacity = ssim(rendered_opacity, gt_alpha)
        loss_opacity = (1.0 - opt.lambda_dssim) * Ll1_opacity + opt.lambda_dssim * (1.0 - ssim_val_opacity)
        tb_dict["loss_mask_entropy"] = loss_opacity.item()
        loss = loss + opt.lambda_opacity * loss_opacity
        
    if light_aware_2DGS:
        if opt.lambda_diffuse_factor_smooth > 0:
            render_diffuse_factor = results['diffuse_factor'].permute(2, 0, 1) 
            loss_diffuse_factor_smooth = bilateral_smooth_loss(render_diffuse_factor, gt_image, gt_alpha)
            tb_dict['diffuse_factor_smooth_loss'] = loss_diffuse_factor_smooth
            loss += opt.lambda_diffuse_factor_smooth * loss_diffuse_factor_smooth
        
        if opt.lambda_ambient_factor_smooth > 0:
            render_ambient_factor = results['ambient_factor'].permute(2, 0, 1) 
            loss_ambient_factor_smooth = bilateral_smooth_loss(render_ambient_factor, gt_image, gt_alpha)
            tb_dict['ambient_factor_smooth_loss'] = loss_ambient_factor_smooth
            loss += opt.lambda_ambient_factor_smooth * loss_ambient_factor_smooth
        
        if opt.lambda_specular_factor_smooth > 0:
            render_specular_factor = results['specular_factor'].permute(2, 0, 1) 
            loss_specular_factor_smooth = bilateral_smooth_loss(render_specular_factor, gt_image, gt_alpha)
            tb_dict['specular_factor_smooth_loss'] = loss_specular_factor_smooth
            loss += opt.lambda_specular_factor_smooth * loss_specular_factor_smooth
        
        if opt.lambda_shininess_factor_smooth > 0:
            render_shininess = results['shininess'].permute(2, 0, 1) 
            loss_shininess_smooth = bilateral_smooth_loss(render_shininess, gt_image, gt_alpha)
            tb_dict['shininess_smooth_loss'] = loss_shininess_smooth
            loss += opt.lambda_shininess_factor_smooth * loss_shininess_smooth
        
        if opt.lambda_normal_smooth > 0:
            loss_normal_smooth = bilateral_smooth_loss(rend_normal, gt_image, gt_alpha)
            tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
            loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
        
    return loss, tb_dict

def disk_render(iteration, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
    override_color = None, opt: OptimizationParams = False, is_training=False, transform_dict=None):
    
    light_aware_2DGS=True if iteration > 10_000 else False
    results = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,transform_dict=transform_dict, light_aware_2DGS=light_aware_2DGS)

    if is_training:
        loss, tb_dict = calculate_loss(iteration, viewpoint_camera, pc, results, opt, light_aware_2DGS)
        results['tb_dict'] = tb_dict
        results['loss'] = loss
        
    return results
    

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, transform_dict=None, light_aware_2DGS=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    palette_color_transforms = transform_dict.get("palette_colors")
    opacity_transforms = transform_dict.get("opacity_factors")
    light_transforms = transform_dict.get("light_transform")
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    diffuse_factor = pc.get_diffuse_factor
    shininess = pc.get_shininess
    ambient_factor = pc.get_ambient_factor
    specular_factor = pc.get_specular_factor 
    normal = pc.get_normal(viewpoint_camera.world_view_transform)
    
    num_GSs_TF = pc._num_GSs_TF
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
    light_dir = None
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    if light_transforms is not None:
        light_dir = light_transforms.get_light_dir()
        spcular_multi, diffuse_factor_multi, ambient_multi, shininess_multi,\
        specular_offset, diffuse_factor_offset, ambient_offset, shininess_offset = light_transforms.get_light_transform()
        diffuse_factor = diffuse_factor.contiguous()*diffuse_factor_multi # ambient white light intensity
        shininess = shininess.contiguous()*shininess_multi # specular white light intensity
        ambient_factor = ambient_factor.contiguous()*ambient_multi # ambient white light intensity
        specular_factor = specular_factor.contiguous()*spcular_multi # specular white light intensity
    if light_dir is not None:
        incidents_dirs = light_dir.detach().contiguous()
    else:
        incidents_dirs = viewdirs.detach().contiguous()
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = True
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            if not light_aware_2DGS:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(0, shs_view, dir_pp_normalized)
                diffuse_color = torch.clamp_min(sh2rgb + 0.5, 0.0) # view-independent color
                brdf_color, _, opacity = rendering_equation_BlinnPhong_python(palette_color_transforms, opacity_transforms, opacity, diffuse_color, \
                diffuse_factor, shininess, ambient_factor, specular_factor, normal, viewdirs, incidents_dirs, num_GSs_TF)
                colors_precomp = brdf_color
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if light_aware_2DGS:
        features = torch.cat([brdf_color, diffuse_color, normal, diffuse_factor, shininess, ambient_factor, specular_factor], dim=-1)
    else:
        features = torch.cat([diffuse_factor], dim=-1) # just to avoid empty tensor error

    
    rendered_image, radii, allmap, rendered_feature = rasterizer(
        means3D = means3D,
        means2D = means2D,
        features = features,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })
    
    if light_aware_2DGS:
        rendered_brdf_color, rendered_diffuse_color, rendered_normal, rendered_diffuse_factor, rendered_shininess, rendered_ambient_factor, rendered_specular_factor = \
            rendered_feature.split([3, 3, 3, 1, 1, 1, 1], dim=0)
        rendered_phong = rendered_brdf_color + (1 - render_alpha)*bg_color[:,None,None]
        rets.update({
            'render': rendered_phong,
            'diffuse_factor': rendered_diffuse_factor,
            'shininess': rendered_shininess,
            'ambient_factor': rendered_ambient_factor,
            'specular_factor': rendered_specular_factor,
        })

    return rets

def visualize_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img.cpu().detach().numpy())
    plt.show()


def rendering_equation_BlinnPhong_python(palette_color_transforms, opacity_transforms, opacity, diffuse_color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs,
                                         incidents_dirs, num_GSs_TFs):
    
    guassian_nums = opacity.shape[0]
    diffuse_factor = diffuse_factor.contiguous() # ambient white light intensity
    shininess = shininess.contiguous() # specular white light intensity
    ambient_factor = ambient_factor.contiguous() # ambient white light intensity
    specular_factor = specular_factor.contiguous() # specular white light intensity
    normals = normals.contiguous()
    viewdirs = viewdirs.contiguous()
    incident_dirs = incidents_dirs.contiguous()

    palette_colors = []
    opacity_factors = []
    if palette_color_transforms is not None:
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
            
    # diffuse color
    cos_l = (normals*incident_dirs).sum(dim=-1, keepdim=True)
    
    diffuse_intensity = diffuse_factor*torch.abs(cos_l)
    diffuse_color = (ambient_factor+diffuse_intensity).clamp(0,1)*diffuse_color

    # specular color
    h = F.normalize(incident_dirs + viewdirs, dim=-1) # bisector h
    cos_h = (normals*h).sum(dim=-1, keepdim=True)
    specular_intensity = specular_factor*torch.where(cos_l != 0, (torch.abs(cos_h)).pow(shininess), 0.0)

    specular_color = specular_intensity.clamp(0,1).repeat(1, 3)

    pbr = diffuse_color.squeeze() + specular_color.squeeze()
    pbr = torch.clamp(pbr, 0., 1.)

    extra_results = {
    "diffuse_render": diffuse_color,
    "specular_render": specular_color,
    }

    return pbr, extra_results, opacity