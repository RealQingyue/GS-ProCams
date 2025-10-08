import json
import os
import time
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from gaussian_renderer import render
from scene import GaussianModel
from utils.camera_utils import loadMicroCameras_COLMAP, loadMicroCameras_JSON, LoadProjector_JSON
from utils.image_utils import loadImage

def save_img_f32(depthmap: np.ndarray, path: str):
    """Save a float32 depth map as a TIFF using OpenCV."""
    clean = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not cv2.imwrite(path, clean):
        raise IOError(f"Failed to write image to {path}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Simple relighting script parameters")
    parser.add_argument("--model_path", '-m', type=str, required=True, help="Path to the model")
    parser.add_argument("--root", '-r', type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--setup", '-s', type=str, required=True, help="Setup name")
    parser.add_argument("--output", '-o', type=str, required=True, help="Output directory")
    parser.add_argument("--iteration", type=int, default=20_000, help="Iteration of GS-ProCams to load")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    parser.add_argument("--views", type=int, nargs='+', default=[1, 6, 11, 16, 21, 26, 27, 28, 29, 30, 31, 32, 33], help="View ids to relight")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--test_fps", action="store_true", help="Test FPS")
    parser.add_argument("--render_scene", action="store_true", help="Render the scene")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    with torch.no_grad():
        # Gaussian model loading
        gaussians = GaussianModel(args.sh_degree)
        ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(ply_path): raise FileNotFoundError(f'Can not find ply file: {ply_path}')
        gaussians.load_ply(ply_path)
        print(f'Gaussians loaded from: {ply_path}')
        
        # Cameras loading
        cam_json_path = os.path.join(args.model_path, "cameras.json")
        try:
            cameras = loadMicroCameras_JSON(cam_json_path, args.views)
            cameras = [cameras] if not isinstance(cameras, list) else cameras
            print(f'Cameras ({len(cameras)}) loaded from json file')
        except:
            colmap_model_dir = os.path.join(args.root, "setups", args.setup, "colmap", "sparse", "0")
            cameras = loadMicroCameras_COLMAP(colmap_model_dir, args.views)
            cameras = [cameras] if not isinstance(cameras, list) else cameras
            print(f"Cameras ({len(cameras)}) loaded from colmap model")
        
        cameras_dict = {}
        for i, view_id in enumerate(args.views):
            cameras_dict[view_id] = cameras[i]

        # Projector loding
        prj_json_path = os.path.join(args.model_path, "projector.json")
        projector = LoadProjector_JSON(prj_json_path) 
        ckpt_path = os.path.join(args.model_path, "procams", f"iteration_{args.iteration}", "procams.ckpt")
        projector.load_ckpt(ckpt_path, weights_only=True)
        print("Projector loaded")
        procams_dict = {"projector": projector}

        # Validation patterns loading
        patterns_valid_dir = os.path.join(args.root, "patterns", "test")
        patterns_valid = [loadImage(os.path.join(patterns_valid_dir, pattern_valid_file)) for pattern_valid_file in tqdm(sorted(os.listdir(patterns_valid_dir)), leave=False)]
        patterns_valid = [torch.from_numpy(pattern_valid).float().cuda().permute(2, 0, 1).clamp(0, 1) for pattern_valid in patterns_valid]
        patterns_valid = torch.stack(patterns_valid, dim=0)
        
        # Relighting
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda") if not args.white_background else torch.ones(3, dtype=torch.float32, device="cuda")
        for view_id, camera in cameras_dict.items():
            save_dir = os.path.join(args.output, f"{view_id:02d}", "relit")
            Ip_out_dir = os.path.join(args.output, f"{view_id:02d}", "Ip_out")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(Ip_out_dir, exist_ok=True)
            for i, pattern in enumerate(tqdm(patterns_valid, desc=f"Relighting view {view_id:02d}", leave=False)):
                procams_dict.update({"pattern": pattern})
                render_dic = render(camera, gaussians, pipe=None, bg_color=bg_color, procams_dict=procams_dict)
                render_image = render_dic["render"]
                save_path = os.path.join(save_dir, f"{i+1:02d}.png")
                save_image(render_image, save_path)
                Ip_out = render_dic["Ip_out"]
                save_path = os.path.join(Ip_out_dir, f"{i+1:02d}.png")
                save_image(Ip_out, save_path)

            if args.render_scene:
                save_scene_dir = os.path.join(args.output, f"{view_id:02d}", "scene")
                os.makedirs(save_scene_dir, exist_ok=True)
                render_pkg = render(camera, gaussians, pipe=None, bg_color=bg_color)
                save_image(render_pkg["base_color"], os.path.join(save_scene_dir, "base_color.png"))
                save_image(render_pkg["roughness"], os.path.join(save_scene_dir, "roughness.png"))
                save_image(render_pkg["render_normal"], os.path.join(save_scene_dir, "render_normal.png"))
                save_image(render_pkg["surf_normal"], os.path.join(save_scene_dir, "surf_normal.png"))
                depth = render_pkg["depth"]
                save_image((depth - depth.min()) / (depth.max() - depth.min()), os.path.join(save_scene_dir, "depth.png"))
                # save depth as a float32 TIFF w/o normalization
                save_img_f32(depth[0].cpu().numpy(), os.path.join(save_scene_dir, 'depth.tiff'))
                save_image(render_pkg["render_shs"], os.path.join(save_scene_dir, "render_shs.png"))
        print("Relighting done")

        # Test FPS
        if args.test_fps:
            start_time = time.time()
            for view_id, camera in cameras_dict.items():
                for i, pattern in enumerate(patterns_valid):
                    procams_dict.update({"pattern": pattern})
                    render_image = render(camera, gaussians, pipe=None, bg_color=bg_color, procams_dict=procams_dict)['render']
            end_time = time.time()
            runtime = end_time - start_time
            fps = len(args.views) * len(patterns_valid) / runtime
            print(f"FPS: {fps}")
            json_file = os.path.join(args.output, "fps.json")
            with open(json_file, "w") as f:
                json.dump({"fps": fps}, f)





