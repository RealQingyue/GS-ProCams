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
import numpy as np
import cv2 as cv
from scene.cameras import Camera
from scene.projector import Projector
from utils.graphics_utils import fov2focal, focal2fov
from tqdm import tqdm

from pathlib import Path
import os
import os.path as osp
from scene.projector import Projector
from scene.cameras import MicroCam
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
import json

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.shape[:2]

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if resolution_scale == 1:
        resized_image_rgb = cam_info.image  
        resized_image_mask = cam_info.image_mask
    else: 
        raise NotImplementedError("Not implemented for resolution_scale != 1") 
    
    gt_image = torch.from_numpy(resized_image_rgb).float().permute(2, 0, 1)
    image_mask = torch.from_numpy(resized_image_mask).float().permute(2, 0, 1)
    
    return Camera(colmap_id=cam_info.uid, view_id=cam_info.view_id, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, K = cam_info.K,
                  image=gt_image, gt_alpha_mask=image_mask,
                  image_name=cam_info.image_name, uid=id,
                  pattern_name=cam_info.pattern_name, pattern_idx=cam_info.pattern_idx,
                  principal_point_ndc=cam_info.principal_point_ndc, data_device=args.data_device)

def cameraList_from_camInfos(procams_infos, resolution_scale, args, type):
    camera_list = []

    for id, c in enumerate(tqdm(procams_infos, desc=f"Loading {type} to device with resolution scale: {resolution_scale}", leave=False, dynamic_ncols=True)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'view_id' : camera.view_id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'T': camera.T.tolist(),
        'K': camera.K.tolist(),
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'principal_point_ndc': camera.principal_point_ndc.tolist()
    }
    return camera_entry

def projector_to_JSON(projector: Projector):
    projector_entry = {
        'width' : projector.width,
        'height' : projector.height,
        'R' : projector.R.tolist(),
        'T' : projector.T.tolist(),
        'K' : projector.K.tolist(),
        'FoVx' : projector.FovX,
        'FoVy' : projector.FovY,
        'principal_point_ndc' : projector.principal_point_ndc.tolist(),
        'w_psf' : projector.w_psf,
    }
    return projector_entry

def projector_from_prjInfo(prj_info):
    return Projector(
        R=prj_info.R, 
        T=prj_info.T, 
        K=prj_info.K, 
        FoVx=prj_info.FovX,
        FoVy=prj_info.FovY, 
        height=prj_info.height, 
        width=prj_info.width,
        principal_point_ndc=prj_info.principal_point_ndc,
        w_psf=prj_info.w_psf,
        )

def patternsTensor_from_numpy(patterns):
    patterns_tensor = [torch.from_numpy(pattern).float().clamp(0.0, 1.0).permute(2, 0, 1) for pattern in patterns]
    return torch.stack(patterns_tensor).cuda().contiguous()

def loadMicroCameras_COLMAP(colmap_dir, target_view_id:list[int]):
    '''
    Read the colmap scene info, return the specific view info

    Return: List of MicroCams
    '''
    if not isinstance(target_view_id, list): target_view_id = [target_view_id]
    try:
        cameras_extrinsic_file = osp.join(colmap_dir, "images.bin")
        cameras_intrinsic_file = osp.join(colmap_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(colmap_dir, "images.txt")
        cameras_intrinsic_file = osp.join(colmap_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_params = {}
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        view_id = Path(extr.name).stem.split("_")[0]
        if view_id in ['spackle', 'calib']: continue
        else: cam_params[int(view_id)] = (extr, intr)
    
    micro_cameras = []
    for target in target_view_id:
        try:
            extr, intr = cam_params[target]
        except:
            raise ValueError(f"Can not find the camera with view_id={target} in the colmap model {cameras_intrinsic_file}")
        
        height = intr.height
        width = intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            raise ValueError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_y, width)
        K = np.array([[focal_length_x, 0.0, cx], [0.0, focal_length_y, cy], [0.0, 0.0, 1.0]])
        principal_point_ndc = np.array([cx / width, cy / height])
        micro_cameras.append(MicroCam(R=R, T=T, K=K, FoVx=FovX, FoVy=FovY, width=width, height=height, principal_point_ndc=principal_point_ndc))
    return micro_cameras if len(micro_cameras) > 1 else micro_cameras[0]
    
def loadMicroCameras_JSON(json_path, target_view_id:list[int]):
    '''
    Read the json file, return the specific view info

    Return: List of MicroCams
    '''
    if not isinstance(target_view_id, list): target_view_id = [target_view_id]
    if not os.path.exists(json_path): raise FileNotFoundError(f"Can not find the json file {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cam_params = {}
    for cam_info in data:
        view_id = cam_info["view_id"]
        cam_params[view_id] = cam_info

    micro_cameras = []
    for target in target_view_id:
        try:
            cam_info = cam_params[target]
        except:
            raise ValueError(f"Can not find the camera with view_id={target} in the json file {json_path}")
        micro_cameras.append(MicroCam(R=np.array(cam_info["rotation"]),
                                      T=np.array(cam_info["T"]),
                                      K=np.array(cam_info["K"]), 
                                      FoVx=cam_info["fx"],
                                      FoVy=cam_info["fy"],
                                      width=cam_info["width"],
                                      height=cam_info["height"], 
                                      principal_point_ndc=np.array(cam_info["principal_point_ndc"])))
    return micro_cameras if len(micro_cameras) > 1 else micro_cameras[0]

def LoadProjector_JSON(json_path):
    if not os.path.exists(json_path): raise FileNotFoundError(f"Can not find the json file {json_path}")
    with open(json_path, 'r') as f:
        prj_info = json.load(f)[0]
        projector = Projector(
            R=np.array(prj_info["R"]),
            T=np.array(prj_info["T"]),
            K=np.array(prj_info["K"]),
            FoVx=prj_info["FoVx"],
            FoVy=prj_info["FoVy"],
            height=prj_info["height"],
            width=prj_info["width"],
            principal_point_ndc=np.array(prj_info["principal_point_ndc"]),
            w_psf=prj_info["w_psf"],
            )
    return projector