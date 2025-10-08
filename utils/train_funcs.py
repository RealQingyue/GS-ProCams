import json
import os
import os.path as osp
import time
import uuid
from argparse import Namespace
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str=os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = osp.join("./output/", unique_str[0:10])
        
    # Set up output folder
    os.makedirs(args.model_path, exist_ok = True)
    with open(osp.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

@torch.no_grad()
def training_report(scene, pipe, background, render, iteration, metric=["psnr", "ssim", "lpips"], fps=True):
    valid_configs = ({"name": "train_train", "cameras": scene.train_cameras_train},
                     {"name": "train_test", "cameras": scene.train_cameras_test},
                     {"name": "test_train", "cameras": scene.test_cameras_train},
                     {"name": "test_test", "cameras": scene.test_cameras_test})
    save_root = osp.join(scene.model_path, "log", f"iteration_{iteration}")
    scene_save_root = osp.join(save_root, "visualization", "scene")

    all_metrics = {}
    saved_views = {}
    for config in valid_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            saved_views[config["name"]] = []
            scene_save_dir = osp.join(scene_save_root, config["name"])
            os.makedirs(scene_save_dir, exist_ok=True)
            metrics = {key: 0.0 for key in metric}
            patterns = scene.patterns if config["name"] in ["train_train", "test_train"] else scene.patterns_test
            for viewpoint in tqdm(config["cameras"], desc=f"Reporting for {config['name']} set", leave=False, dynamic_ncols=True):
                procams_dict = {"projector": scene.projector, "pattern": patterns[viewpoint.pattern_idx]}
                render_image = render(viewpoint, scene.gaussians, pipe, background, procams_dict)["render"]
                scene_images = render(viewpoint, scene.gaussians, pipe, background)
                gt_image = viewpoint.original_image.to("cuda") * viewpoint.gt_alpha_mask
                render_image_masked = render_image * viewpoint.gt_alpha_mask
                # 1. compute metrics
                for key in metrics.keys():
                    if key == "psnr":
                        metrics[key] += psnr(render_image_masked, gt_image).mean().double()
                    elif key == "ssim":
                        metrics[key] += ssim(render_image_masked, gt_image).mean().double()
                    elif key == "lpips":
                        metrics[key] += lpips(render_image_masked, gt_image, net_type="vgg").mean().double()
                # # 2. save render images
                save_dir = osp.join(save_root, "visualization", "render", config["name"], f"{viewpoint.view_id}")
                os.makedirs(save_dir, exist_ok=True)
                save_image(render_image, osp.join(save_dir, f"{viewpoint.pattern_name}.png"))
                # 3. save brdf maps for each view
                if viewpoint.view_id not in saved_views[config["name"]]:
                    os.makedirs(save_dir, exist_ok=True)
                    save_image(scene_images["base_color"], osp.join(scene_save_dir, f"{viewpoint.view_id}_albedo.png"))
                    save_image(scene_images["roughness"], osp.join(scene_save_dir, f"{viewpoint.view_id}_roughness.png"))
                    save_image(scene_images["render_normal"], osp.join(scene_save_dir, f"{viewpoint.view_id}_render_normal.png"))
                    save_image(scene_images["surf_normal"], osp.join(scene_save_dir, f"{viewpoint.view_id}_surf_normal.png"))
                    depth = scene_images["depth"] # normalization
                    save_image((depth - depth.min()) / (depth.max() - depth.min()), osp.join(scene_save_dir, f"{viewpoint.view_id}_depth.png"))
                    saved_views[config["name"]].append(viewpoint.view_id)
            # average metrics
            for key in metrics.keys():
                metrics[key] = (metrics[key]/len(config["cameras"])).item()
            all_metrics.update({config["name"]: metrics})

            if fps:
                # Run FPS test with multiple iterations for better accuracy
                fps_iterations = 5  # Repeat the camera loop multiple times
                total_renders = 0
                
                start_time = time.time()
                for iteration in range(fps_iterations):
                    for viewpoint in tqdm(config["cameras"], desc=f"FPS Test Iter {iteration+1}/{fps_iterations} for {config['name']} set", leave=False, dynamic_ncols=True):
                        procams_dict = {"projector": scene.projector, "pattern": patterns[viewpoint.pattern_idx]}
                        render(viewpoint, scene.gaussians, pipe, background, procams_dict=procams_dict)
                        total_renders += 1
                end_time = time.time()
                
                runtime = end_time - start_time
                fps = total_renders / runtime
                all_metrics[config["name"]]["fps"] = fps

    for key, value in all_metrics.items():
        print(f"    * Metrics for {key} set: {value}")
    json_path = osp.join(save_root, f"metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f)
    return all_metrics



    
    
