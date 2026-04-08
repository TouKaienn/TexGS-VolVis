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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
import torchvision
from scene.opacity_trans import LearningOpacityTransform
from scene.palette_color import LearningPaletteColor
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from utils.system_utils import searchForMaxIteration
from utils.gui_utils import texGS_scene_composition
import glob
import open3d as o3d


def load_ckpts_paths(source_dir, style_names=[]):
    TFs_folders = sorted(glob.glob(f"{source_dir}/TF*"))
    TFs_names = sorted([os.path.basename(folder) for folder in TFs_folders])
    if len(style_names) == 0:
        style_names=['texgs' for i in range(len(TFs_folders))]
    ckpts_transforms = {}
    for idx, TF_folder in enumerate(TFs_folders):
        one_TF_json = {'path': None, 'palette':None, 'texture':None, 'transform': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        ckpt_dir = os.path.join(TF_folder,style_names[idx],"point_cloud")
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument('--output_dir', default=None, required=True, help="the output dir")
    parser.add_argument('--style_names', type=str, nargs='+', help='The path of image dir, can be multiple dirs separated by space or can be several img file path (WARNING: multiple paths have not been tested yet)')
    parser.add_argument("--stylized_texture", action="store_true", help='Using stylized texture, set the palette color to black')
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_false")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args()
    
    safe_state(args.quiet)
    dataset = model.extract(args)
    
    scene_dict = load_ckpts_paths(args.source_dir, style_names=args.style_names)
    gaussians_composite, _ = texGS_scene_composition(scene_dict, dataset)
    render_fn = render_fn_dict["TexGS"]

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)

    scene = Scene(dataset, gaussians_composite, load_iteration=iteration, shuffle=False, useTexGS=True, overLoadGS=False)
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    """
    Setup transform dict
    """
    transform_dict = dict()
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)
    palette_color_transforms = []
    opacity_transforms = []
    for TFs_name in TFs_names:
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.create_from_ckpt(f"{scene_dict[TFs_name]['palette']}")
        opacity_transform = LearningOpacityTransform(opacity_factor=1.0)
        opacity_transforms.append(opacity_transform)
        if args.stylized_texture:
            palette_color_transform.set_black()
        palette_color_transforms.append(palette_color_transform)
    
    transform_dict["palette_colors"] = palette_color_transforms
    transform_dict["opacity_factors"] = opacity_transforms

    train_dir = os.path.join(args.output_dir, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.output_dir, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians_composite, render_fn, pipe, bg_color=bg_color, transform_dict=transform_dict)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)