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

import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from random import randint
import torch
import torch.nn.functional as F
import torch.profiler
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import resetRNGseed
from utils.train_funcs import training_report, prepare_output_and_logger

PROFILER = False

def training(dataset, opt, pipe):
    prepare_output_and_logger(dataset)
    # Gaussian setup
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians) 
    first_iter = 0 if scene.loaded_iter is None else scene.loaded_iter
    gaussians.training_setup(opt)

    # ProCams setup
    scene.projector.training_setup(opt.PEModel_lr)
    procams_dict = {"projector": scene.projector}

    # Render setup
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training", dynamic_ncols=True)
    first_iter += 1

    # Profiler
    profiler = None
    if PROFILER:
        profiler_dir = os.path.join(dataset.model_path, "log", "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(skip_first=1000, wait=300, warmup=150, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
        )

    train_start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration) 

        # regularization
        opt.normal_consistency_loss = opt.lambda_normal if iteration > 7000 else 0.0 
        opt.lambda_depth_distortion = opt.lambda_dist if iteration > 3000 else 0.0

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.train_cameras_train.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        procams_dict.update(pattern=scene.patterns[viewpoint_cam.pattern_idx])
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, procams_dict, training=True, opt=opt)
        viewspace_point_tensor, visibility_filter, radii =render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        loss = render_pkg["loss"]
        log_items = render_pkg["log_items"]
        loss.backward()

        # Log
        with torch.no_grad():
            # Progress bar
            progress_bar_dict = {"num": gaussians.get_xyz.shape[0]}
            for item in log_items:
                if item in ["psnr", "ssim"]:
                    ema_dict_for_log[item] = 0.4 * log_items[item] + 0.6 * ema_dict_for_log[item]
                    progress_bar_dict[item] = f"{ema_dict_for_log[item]:.{3}f}"
            if iteration % 10 == 0: progress_bar.set_postfix(progress_bar_dict)
            progress_bar.update(1)
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
 
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            scene.projector.step()

            # Profiler
            if profiler is not None: profiler.step()

            # Log
            if dataset.save_interval and (iteration % dataset.save_interval == 0 or iteration == opt.iterations):
                scene.save(iteration)
                print(f"\n[ITER {iteration}] Saving GS-ProCams")
            if dataset.report_interval and (iteration % dataset.report_interval == 0 or iteration == opt.iterations):
                progress_bar.set_description(f"[ITER {iteration:5d}] Training Report")
                _ = training_report(scene, pipe, background, render, iteration, metric=["psnr", "ssim", "lpips"], fps=True)
                progress_bar.set_description(f"Training")
                
            
    train_end_time = time.time()
    train_time = (train_end_time - train_start_time) / 60.0 # in minutes
    progress_bar.close()
    if profiler is not None: profiler.stop()
    
    # Final evaluation if requested
    if dataset.evaluate:
        print(f"\n[ITER {iteration}] Final Evaluation")
        _ = training_report(scene, pipe, background, render, iteration, metric=["psnr", "ssim", "lpips"], fps=True)
            
    # Checkpoint
    scene.save(iteration)
    print(f"[ITER {iteration}] Saving Gaussians")
    print(f"Finished training after {train_time:.4f} minutes")
    extra_log = {"train_time": train_time}
    json_path = os.path.join(args.model_path, "extra_log.json")
    with open(json_path, "w") as f:
        json.dump(extra_log, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print(f"Optimizing {args.model_path}")

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, the device becomes 0

    # Initialize system state (RNG)
    resetRNGseed(0)

    training(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
