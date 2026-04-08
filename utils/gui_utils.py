from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import glob
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scene.jagged_texture import JaggedTexture
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from utils.system_utils import searchForMaxIteration
from utils.segment_utils import *
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import GaussianModel
from tqdm import tqdm
from scene.dataset_readers import sceneLoadTypeCallbacks, readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import itertools
from copy import deepcopy

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def screen_to_arcball(p:np.ndarray):
    dist = np.dot(p, p)
    if dist < 1.:
        return np.array([*p, np.sqrt(1.-dist)])
    else:
        return np.array([*normalize_vec(p), 0.])

def normalize_vec(v: np.ndarray):
    if v is None:
        print("None")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norm == np.zeros_like(norm)):
        return np.zeros_like(v)
    else:
        return v/norm

def apply_mask_and_draw_circles(
    image: np.ndarray,
    mask: np.ndarray,
    points: list,
    labels: list,
    alpha: float = 0.6,
    circle_radius: int = 5
) -> np.ndarray:
    """
    Apply a semi-transparent red mask overlay on the image and draw colored circles at given points.

    Args:
        image: Original image, shape (H, W, 3), float, range [0,1].
        mask: Boolean mask, shape (1, H, W), True indicates overlay region.
        points: List of circle center coordinates, each element is [x, y].
        labels: List of labels for each point (1 for positive, 0 for negative).
        alpha: Transparency of the red overlay, default 0.6.
        circle_radius: Radius of the circles, default 5.

    Returns:
        Processed image, shape (H, W, 3), float, range [0,1].
    """
    output_bgr = (image * 255).astype(np.uint8)

    mask_bool = mask[0]
    b_val, g_val, r_val = 0, 0, 255

    out_b = output_bgr[..., 0]
    out_g = output_bgr[..., 1]
    out_r = output_bgr[..., 2]

    out_b[mask_bool] = np.clip((1 - alpha) * out_b[mask_bool] + alpha * b_val, 0, 255)
    out_g[mask_bool] = np.clip((1 - alpha) * out_g[mask_bool] + alpha * g_val, 0, 255)
    out_r[mask_bool] = np.clip((1 - alpha) * out_r[mask_bool] + alpha * r_val, 0, 255)

    for idx, (x, y) in enumerate(points):
        if labels[idx] == 1:
            cv2.circle(output_bgr, center=(int(x), int(y)), radius=circle_radius, color=(0, 255, 0), thickness=-1)
        else:
            cv2.circle(output_bgr, center=(int(x), int(y)), radius=circle_radius, color=(255, 0, 0), thickness=-1)
    final_output = output_bgr.astype(np.float32) / 255.0

    return final_output

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    coords = (np.asarray(coords)).astype(np.int32)
    labels = np.array(labels)
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def load_ckpts_paths(source_dir, style_name='texgs'):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = sorted([os.path.basename(folder) for folder in TFs_folders])

    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        one_TF_json = {'path': None, 'palette':None, 'texture':None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        ckpt_dir = os.path.join(TF_folder,style_name,"point_cloud")
        if not os.path.exists(ckpt_dir):
            continue
        max_iters = searchForMaxIteration(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "point_cloud.ply")
        palette_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "palette_colors_chkpnt.pth")
        texture_path = os.path.join(ckpt_dir, f"iteration_{max_iters}", "texture.npz")
        one_TF_json['path'] = ckpt_path
        one_TF_json['palette'] = palette_path
        one_TF_json['texture'] = texture_path
        ckpts_transforms[TFs_names[idx]] = one_TF_json

    return ckpts_transforms

def load_compose_texture(texture_path):
    texture_npz = np.load(texture_path)
    texture_dim = torch.tensor(texture_npz['texture_dims'], dtype=torch.int32, device="cuda")
    texture = JaggedTexture(texture_dim, out_dim=3)
    texture.texture.data = torch.tensor(texture_npz['texture_dc'], dtype=torch.float32, device="cuda")
    mappings = torch.tensor(texture_npz['mappings'], device="cuda")

    return texture, mappings

def texGS_scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    texture_list = []
    mappings_list = []
    
    gs_list_kwargs = {'gaussian_list': gaussians_list, 'texture_list': texture_list, 'mappings_list': mappings_list}
    
    for scene in scene_dict:
        gaussians = GaussianModel(dataset.sh_degree)
        print("Compose scene from GS path:", scene_dict[scene]["path"])
        gaussians.load_ply(scene_dict[scene]["path"], isTexGS=True)
        gaussians_list.append(gaussians)
        one_TF_texture, one_TF_mappings = load_compose_texture(scene_dict[scene]["texture"])
        texture_list.append(one_TF_texture)
        mappings_list.append(one_TF_mappings)
    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list)
    texture = JaggedTexture.create_from_textures(texture_list, gaussians_composite.get_num_GSs_TFs)
    gaussians_composite.load_jagged_texture(texture, mappings_list)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    return gaussians_composite, gs_list_kwargs

def get_ensemble_cameras(args, view_path):
    ensemble_camera_info = readCamerasFromTransforms(view_path, "transforms_train.json", True)
    ensemble_cameras = cameraList_from_camInfos(ensemble_camera_info, 1.0, args)
    return ensemble_cameras

def split_compose_gaussians(compose_gaussians, focus_TF_idx, render_fn, \
                            render_kwargs, ref_view_cam, \
                            ensemble_view_cams, predictor, input_points, input_labels, threshold=0.7):
    num_GSs_TF = compose_gaussians.get_num_GSs_TFs
    start_end_GSs_idx = [0] + list(itertools.accumulate(num_GSs_TF))
    segment_GSs_start_idx = start_end_GSs_idx[focus_TF_idx]
    segment_GSs_end_idx = start_end_GSs_idx[focus_TF_idx+1]
    target_GSs_xyz = compose_gaussians.get_xyz[segment_GSs_start_idx:segment_GSs_end_idx]

    prompts_3d = generate_3d_prompts(target_GSs_xyz, ref_view_cam, input_points)
    mask_id = 1
    multiview_masks = []
    sam_masks = []

    for i, view in tqdm(enumerate(ensemble_view_cams)):
        render_pkg = render_fn(iteration=-1, viewpoint_camera=view, **render_kwargs)
        render_image = render_pkg["render"].detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
        prompts_2d = porject_to_2d(view, prompts_3d)
        predictor.set_image(render_image)
        sam_feature = predictor.features
        sam_mask = self_prompt(prompts_2d, input_labels, sam_feature, predictor, mask_id)
        if len(sam_mask.shape) != 2:
            sam_mask = torch.from_numpy(sam_mask).squeeze(-1).to("cuda")
        else:
            sam_mask = torch.from_numpy(sam_mask).to("cuda")
        sam_mask = sam_mask.long()
        sam_masks.append(sam_mask)
        point_mask, indices_mask = mask_inverse(target_GSs_xyz, view, sam_mask)
        multiview_masks.append(point_mask.unsqueeze(-1))
    _, final_mask = ensemble(multiview_masks, threshold=threshold)
    new_compose_guassians, new_transform_dict, temp_save_kwargs = sorted_GSs(compose_gaussians, focus_TF_idx, final_mask, render_kwargs)
    return new_compose_guassians, new_transform_dict, temp_save_kwargs

def save_segmentations(temp_save_kwargs, transform_dict, save_dir):
    num_TFs = len(transform_dict['palette_colors'])
    for tf_idx in range(num_TFs):
        tf_save_dir = os.path.join(save_dir, f"TF{tf_idx}", 'texgs', 'point_cloud', 'iteration_0')
        os.makedirs(tf_save_dir, exist_ok=True)
        torch.save((transform_dict['palette_colors'][tf_idx].capture(), 0), 
                                os.path.join(tf_save_dir, f"palette_colors__chkpnt" + ".pth")) # save palette color
        gaussian = temp_save_kwargs['guassians_list'][tf_idx]
        texture = temp_save_kwargs['texture_list'][tf_idx]
        mappings = [temp_save_kwargs['mappings_list'][tf_idx]]
        gaussian.useTexGS = True
        gaussian.load_jagged_texture(texture, mappings)
        gaussian.save_ply(os.path.join(tf_save_dir, "point_cloud.ply"))
        gaussian.save_texture_npz(os.path.join(tf_save_dir, "texture.npz"))
    

def sorted_GSs(gaussians, focus_TF_idx, final_mask, render_kwargs):
    num_GSs_TF = gaussians.get_num_GSs_TFs
    all_mappings = gaussians._mappings
    start_end_GSs_idx = [0] + list(itertools.accumulate(num_GSs_TF))
    texture = gaussians.texture
    new_transform_dict = render_kwargs['transform_dict']

    gaussians_list = []
    texture_list = []
    mappings_list = []
    segment_GSs_start_idx = 0
    for tf_idx in range(len(num_GSs_TF)):
        segment_GSs_start_idx = start_end_GSs_idx[tf_idx]
        segment_GSs_end_idx = start_end_GSs_idx[tf_idx+1]
        print(segment_GSs_start_idx, segment_GSs_end_idx, start_end_GSs_idx)
        if tf_idx == focus_TF_idx: # do sth different
            # duplicate the opacity and palette_color
            new_palette_color = deepcopy(new_transform_dict['palette_colors'][tf_idx])
            new_opacity = deepcopy(new_transform_dict['opacity_factors'][tf_idx])
            new_transform_dict['palette_colors'].insert(tf_idx+1, new_palette_color)
            new_transform_dict['opacity_factors'].insert(tf_idx+1, new_opacity)
            
            all_indices = torch.arange(segment_GSs_start_idx, segment_GSs_end_idx, device="cuda")
            seg_indices = final_mask+segment_GSs_start_idx
            seg_indices = seg_indices.to("cuda")
            
            unseg_indices = all_indices[~torch.isin(all_indices, seg_indices)]
            texture_list.append(JaggedTexture.select_texture_from_indices(texture, unseg_indices))
            mappings_list.append(all_mappings[unseg_indices,:])
            gaussians_list.append(GaussianModel.select_from_gaussians(gaussians, unseg_indices))
            
            texture_list.append(JaggedTexture.select_texture_from_indices(texture, seg_indices))
            mappings_list.append(all_mappings[seg_indices,:])
            gaussians_list.append(GaussianModel.select_from_gaussians(gaussians, seg_indices))
        else:
            indices = torch.arange(segment_GSs_start_idx, segment_GSs_end_idx, device="cuda")
            texture_list.append(JaggedTexture.select_texture_from_indices(texture, indices))
            mappings_list.append(all_mappings[indices,:])
            gaussians_list.append(GaussianModel.select_from_gaussians(gaussians, indices))
        
    res_gaussians = GaussianModel.create_from_gaussians(gaussians_list)
    texture = JaggedTexture.create_from_textures(texture_list, res_gaussians.get_num_GSs_TFs)
    res_gaussians.load_jagged_texture(texture, mappings_list)
    
    temp_save_kwargs = {'guassians_list': gaussians_list, 'texture_list': texture_list, 'mappings_list': mappings_list}
    
    return res_gaussians, new_transform_dict, temp_save_kwargs

class ArcBallCamera:
    def __init__(self, W, H, fovy=60, near=0.1, far=10, rot=None, translate=None, center=None):
        self.W = W
        self.H = H
        if translate is None:
            self.radius = 1
            self.original_radius = 1
        else:
            self.radius = np.linalg.norm(translate)
            self.original_radius = np.linalg.norm(translate)

        self.radius *= 2
        self.fovy = fovy  # in degree
        self.near = near
        self.far = far

        if center is None:
            self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        else:
            self.center = center

        if rot is None:
            self.rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # looking back to z axis
            self.original_rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        else:
            self.rot = R.from_matrix(rot)
            self.original_rot = R.from_matrix(rot)

        self.up = -self.rot.as_matrix()[:3, 1]

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)
    
    def reset_view(self):
        self.rot = self.original_rot
        self.radius = self.original_radius
        self.radius *= 2

    def orbit(self, lastX, lastY, X, Y):
        def vec_angle(v0: np.ndarray, v1: np.ndarray):
            return np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.))
        ball_start = screen_to_arcball(np.array([lastX+1e-6, lastY+1e-6]))
        ball_curr = screen_to_arcball(np.array([X, Y]))
        rot_radians = vec_angle(ball_start, ball_curr)
        rot_axis = normalize_vec(np.cross(ball_start, ball_curr))
        q = Quaternion(axis=rot_axis, radians=rot_radians)
        self.rot = self.rot * R.from_matrix(q.inverse.rotation_matrix)
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

def replace_color_to_contrast(color):
    return (1 - color) * 0.7
