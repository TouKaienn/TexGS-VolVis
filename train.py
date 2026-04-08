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
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
import torch.profiler
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from icecream import ic
import torch
import psutil
import time

def print_memory_usage(gpu_id=0):
    # GPU memory
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
    reserved  = torch.cuda.memory_reserved(gpu_id) / 1024**2

    # CPU memory
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024**2
    vms = process.memory_info().vms / 1024**2

    print(f"[GPU {gpu_id}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    print(f"[CPU] RSS: {rss:.2f} MB | VMS: {vms:.2f} MB")
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, init_2DGS_path):
    useTexGS = True if args.type in ["stylize", "TexGS"] else False
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    
    """
    Setup transform dict
    """
    transform_dict = dict()
    if (init_2DGS_path) and (useTexGS):
        palette_color_transforms = []
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.load_palette_color(args.source_path+'/train')
        palette_color_transform.training_setup(opt)
        palette_color_transforms.append(palette_color_transform)
        transform_dict["palette_colors"] = palette_color_transforms
        
        gaussians.initTexGSfrom2DGS(init_2DGS_path, opt)
        gaussians.freeze_attributes()
    
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    tic = time.time()
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render_fn(iteration, viewpoint_cam, gaussians, pipe, background, \
            is_training=True, opt=opt, transform_dict=transform_dict)
        
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        total_loss = render_pkg["loss"]
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

            save_training_vis(viewpoint_cam, gaussians, background, render_fn,
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


            training_report(tb_writer, iteration, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fn, (pipe, background), transform_dict=transform_dict)
            if (iteration in saving_iterations):
                print_memory_usage()
                print("\n[ITER {}] Training Time Used {}".format(iteration, time.strftime("%H:%M:%S", time.gmtime(time.time() - tic))))
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification for 2DGS
            if (iteration < opt.densify_until_iter) and (args.type == "2DGS"):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            if (iteration % opt.build_chart_every==0) and (args.type == "TexGS"):
                gaussians.build_charts()
                gaussians.reshape_in_all_optim()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                for component in transform_dict.values():
                    if isinstance(component, list):
                        for c in component:
                            c.step()
                    else:
                        component.step()
                
            

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
        gaussExtractor = GaussianExtractor(gaussians, render_fn, pipe, bg_color=bg_color, transform_dict=transform_dict)
        os.makedirs(scene.model_path+"/test", exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(scene.model_path+"/test")


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
                ssim_test = 0.0
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
                    ssim_test += ssim(image, gt_image).mean().double()
                ssim_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()
        
def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, transform_dict):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(iteration, viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, transform_dict=transform_dict)
            render = render_pkg["render"] if render_pkg["render"].shape[2] != 3 else render_pkg["render"].permute(2, 0, 1)
            visualization_list = [
                render,
                viewpoint_cam.original_image.cuda(),
            ]

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=2)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument('-t', '--type', choices=['2DGS', 'TexGS', 'stylize'], default='2DGS')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("-init", "--init_2DGS_path", type=str, default = None)
    parser.add_argument("-c", "--checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.init_2DGS_path)

    # All done
    print("\nTraining complete.")