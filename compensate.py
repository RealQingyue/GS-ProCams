import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from gaussian_renderer import CompensationOptions, Compensater
from scene import GaussianModel
from utils.camera_utils import loadMicroCameras_COLMAP, loadMicroCameras_JSON, LoadProjector_JSON
from utils.general_utils import create_logger
from utils.image_utils import loadImage
import time

if __name__ == '__main__':
    parser = ArgumentParser(description="Compensation script parameters")
    parser.add_argument("--model_path", '-m', type=str, required=True, help="Path to the model")
    parser.add_argument("--root", '-r', type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--setup", '-s', type=str, required=True, help="Setup name")
    parser.add_argument("--iteration", type=int, default=20_000, help="Iteration of GS-ProCams to load")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    parser.add_argument("--view_id", type=int, default=25, help="View id to compensate")
    parser.add_argument("--vis", action="store_true", help="Show the compensation process")
    parser.add_argument("--wo_cmp_init", action="store_true", help="Init compensation by prj depth")
    parser.add_argument("--prj_mask", action="store_true", help="Use the projector mask derived for the desired images")
    parser.add_argument("--desired", '-d', type=str, default=None, help="Path to the desired images")
    parser.add_argument("--output", '-o', type=str, default=None, help="Output directory")
    args = parser.parse_args()

    device = torch.device(f"cuda:{0}")
    torch.cuda.set_device(device)

    # Set default directories
    if args.desired is None:
        args.desired = osp.join(args.root, "setups", f"{args.setup}", "views", f"{args.view_id:02d}", "cam", "desire", "test")
    if args.output is None:
        args.output = osp.join(args.model_path, "prj", "cmp", f"{args.view_id:02d}", "test")
    os.makedirs(args.output, exist_ok=True)
    log_save_path = osp.join(args.output, f"compensate.log")
    # Create a logger
    logger = create_logger(log_save_path)
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Script invoked at {start_time}")
    logger.info(f"Log file saved to {log_save_path}")
    
    # Gaussian model loading
    gaussians = GaussianModel(args.sh_degree)
    ply_path = osp.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path): raise FileNotFoundError(f'Can not find ply file: {ply_path}')
    gaussians.load_ply(ply_path)
    logger.info(f'Gaussians loaded from: {ply_path}')

    # Camera loading
    cam_json_path = osp.join(args.model_path, "cameras.json")
    try:
        cameras = loadMicroCameras_JSON(cam_json_path, args.view_id)
        logger.info(f'Cameras infos loaded from: {cam_json_path}')
    except:
        logger.info(f"{args.view_id} May not be in the trained camera.json, finding the camera in the novel colmap model")
        colmap_model_dir = osp.join(args.root, "setups", f"{args.setup}", "colmap", "sparse", "0")
        camera = loadMicroCameras_COLMAP(colmap_model_dir, args.view_id)

    # Projector loding
    prj_json_path = osp.join(args.model_path, "projector.json")
    projector = LoadProjector_JSON(prj_json_path)
    logger.info(f'Projector info loaded from: {prj_json_path}')

    ckpt_path = osp.join(args.model_path, "procams", f"iteration_{args.iteration}", "procams.ckpt")
    projector.load_ckpt(ckpt_path, weights_only=True)
    logger.info(f"Projector ({projector.image_width}x{projector.image_height}) loaded from: {ckpt_path}")

    # Compen data loading
    ## Desired images
    cam_desired = [loadImage(osp.join(args.desired, img_desired_file)) for img_desired_file in tqdm(sorted(os.listdir(args.desired)), leave=False)]
    cam_desired = [torch.from_numpy(img_desired).float().cuda().permute(2, 0, 1).clamp(0, 1) for img_desired in cam_desired]
    cam_desired = torch.stack(cam_desired, dim=0)
    cam_mask = None if not args.prj_mask else cam_desired.mean(0).mean(0).unsqueeze(0) > 0
    logger.info(f'{len(cam_desired)} desired images (test) loaded from: {args.desired}')
            
    # Options used in the paper
    cmp_option = CompensationOptions()
    logger.info(f"Compensation options: {cmp_option}")
    compensater = Compensater(gaussians, camera, projector, cmp_option, cam_mask, simplified=True, logger=logger)

    l1_list = []
    ssim_list = []
    psnr_list = []
    total_time = 0
    for i, desired in enumerate(tqdm(cam_desired, desc='Compensating images', leave=False, dynamic_ncols=True)):
        time_start = time.time()
        pattern_cmp, img_cmp, logs_items = compensater.compensate(desired, args.vis)
        time_end = time.time()
        total_time += time_end - time_start
        ## save
        img_cmp_save_dir = osp.join(args.output, 'imgs')
        pattern_cmp_save_dir = osp.join(args.output, 'patterns')
        os.makedirs(img_cmp_save_dir, exist_ok=True)
        os.makedirs(pattern_cmp_save_dir, exist_ok=True)
        save_image(img_cmp, osp.join(img_cmp_save_dir, f"{i+1:03d}.png"))
        save_image(pattern_cmp, osp.join(pattern_cmp_save_dir, f"{i+1:03d}.png"))
        ## metrics log
        if 'l1' in logs_items: l1_list.append(logs_items['l1'])
        if 'ssim' in logs_items: ssim_list.append(logs_items['ssim'])
        if 'psnr' in logs_items: psnr_list.append(logs_items['psnr'])
    l1_avg = np.mean(l1_list) if len(l1_list) > 0 else 0
    ssim_avg = np.mean(ssim_list) if len(ssim_list) > 0 else 0
    psnr_avg = np.mean(psnr_list) if len(psnr_list) > 0 else 0
    fps = len(cam_desired) / total_time if total_time > 0 else 0
    logger.info(f"Finished!\n  * {len(l1_list)} patterns are saved in {args.output}\n  * Simulation Quality - L1: {l1_avg:.4f}, SSIM: {ssim_avg:.4f}, PSNR: {psnr_avg:.2f}, FPS: {fps:.4f}")






    