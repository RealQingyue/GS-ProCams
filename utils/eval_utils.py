import json
from enum import Enum
import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm
import yaml
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim

class Metrics(Enum):
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"

def computeMetrics(imgs1:torch.Tensor, imgs2:torch.Tensor, metrics:list, mask:torch.Tensor=None):
    '''
    compute the metrics of the images
    imgs1: the first image tensor
    imgs2: the second image tensor
    mask: the mask tensor

    '''
    if imgs1.shape != imgs2.shape:
        raise ValueError(f"Images shape mismatch: {imgs1.shape} != {imgs2.shape}")
    if imgs1.ndim == 3:
        imgs1 = imgs1.unsqueeze(0)
        imgs2 = imgs2.unsqueeze(0)
    if mask is not None:
        imgs1 = imgs1 * mask
        imgs2 = imgs2 * mask
    
    results = {}
    vals = {}
    for metric in metrics:
        if metrics is None or len(metrics) == 0:
            print("No metrics to compute")
            return {}
        else:
            vals[metric] = []

    for i, img1 in tqdm(enumerate(imgs1), leave=False, desc="Benchmarking"):
        img1 = img1 
        img2 = imgs2[i]
        if Metrics.PSNR in metrics:
            vals[Metrics.PSNR].append(psnr(img1, img2).mean().double())
        if Metrics.SSIM in metrics:
            vals[Metrics.SSIM].append(ssim(img1, img2).mean().double())
        if Metrics.LPIPS in metrics:
            lpips_val = lpips(img1, img2, net_type='vgg').mean().double()
            vals[Metrics.LPIPS].append(lpips_val)
    
    for metric in metrics:
        results[metric] = torch.tensor(vals[metric]).mean().item()
    
    return results

def save_metrics_to_json(metrics_dict, json_path):
    # save metrics to JSON
    try:
        with open(json_path, 'w') as json_file:
            json.dump(metrics_dict, json_file, indent=4)
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")

def load_image_as_tensor(image_path, device):
    # Load image as tensor
    try:
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(image_np.shape) == 2:
            # Grayscale image: add channel dimension
            image_tensor = torch.tensor(image_np).unsqueeze(0).float().to(device) / 255.0
        elif len(image_np.shape) == 3:
            # Color image: permute to (C, H, W)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float().to(device) / 255.0
        else:
            print(f"Unsupported image dimensions {image_np.shape} for image {image_path}")
            return None
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_depth_data(depth_path, mask=None, dtype=np.float32):
    depth_data = np.loadtxt(depth_path) if depth_path.endswith('.txt') else cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if mask is not None: depth_data = depth_data[mask]
    return depth_data.astype(dtype)

def loadCalib(file_name):
    def stringToMat(m):
        n_rows = len(m)
        n_cols = len(m[0].split(','))
        mat = np.zeros((n_rows, n_cols))
        for r in range(n_rows):
            cur_row = m[r].split(',')
            for c in range(n_cols):
                mat[r][c] = float(cur_row[c])
        return mat

    with open(file_name) as f:
        raw_data = yaml.load(f, yaml.Loader)

    calib_data = {}
    for m in raw_data:
        calib_data[m] = stringToMat(raw_data[m])

    # convert to Bx4x4
    tensor_4x4 = np.eye(4)
    tensor_4x4[0:3, 0:3] = calib_data['camK']
    calib_data['camK'] = np.expand_dims(tensor_4x4, axis=0).copy()
    tensor_4x4[0:3, 0:3] = calib_data['prjK']
    calib_data['prjK'] = np.expand_dims(tensor_4x4, axis=0).copy()

    # extrinsics 3x4 ->1x4x4
    tensor_4x4[0:3, :] = calib_data['camRT']
    calib_data['camRT'] = np.expand_dims(tensor_4x4, axis=0).copy()
    tensor_4x4[0:3, :] = calib_data['prjRT']
    calib_data['prjRT'] = np.expand_dims(tensor_4x4, axis=0).copy()

    return calib_data

def depth_to_points(depthmap, cam_KRT):
    '''
    depthmap: (H, W)
    cam_KRT: (3, 3)

    return: (H, W, 3) points
    '''
    H, W = depthmap.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    points = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3)
    rays_d = points @ np.linalg.inv(cam_KRT).T
    rays_o = np.zeros(3) # camera center is the origin as in DeProCams
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points.reshape(H, W, 3)

def align_point_clouds(pc_pred, pc_gt, mask=None):
    '''
    pc_pred: (H, W, 3)
    pc_gt: (H, W, 3)
    
    mask: (H, W)

    return: float
    '''
    if mask is None: mask = np.ones(pc_gt.shape[:2], dtype=bool)
    d_err, _ = cKDTree(pc_gt[mask]).query(pc_pred[mask], 1)
    return d_err.mean()



