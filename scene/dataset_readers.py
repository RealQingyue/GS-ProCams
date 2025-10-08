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
import os.path as osp
import sys
from os import listdir
from pathlib import Path
from typing import NamedTuple
import cv2 as cv
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, read_points3D_binary, read_points3D_text, qvec2rotmat
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.image_utils import loadImage
from utils.sh_utils import SH2RGB

class ProjectorInfo(NamedTuple):
    width: int
    height: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    principal_point_ndc: np.array
    w_psf: bool=True

class CameraInfo(NamedTuple):
    uid: int
    view_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    image: np.array
    image_name: str
    image_mask: np.array
    width: int
    height: int
    principal_point_ndc: np.array
    pattern_idx: int
    pattern_name: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    train_cameras_test: list
    test_cameras_test: list
    nerf_normalization: dict
    ply_path: str
    prj_info: dict
    patterns: list
    patterns_test: list

def getNerfppNorm(cam_infos):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam_info in cam_infos:
        cam = cam_info
        W2C= getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapCameras(params, path, train_views, valid_views, calib_name, wo_mask, wo_psf, num_images):
    patterns_dir = osp.join(os.path.dirname(os.path.dirname(path)), "patterns")
    
    # Info container
    infos = {
        (False, False): [],  # Training viewpoint, training images
        (False, True): [],   # Training viewpoint, testing images
        (True, False): [],   # Testing viewpoint, training images
        (True, True): []     # Testing viewpoint, testing images
    }
    prj_info = None

    # Load patterns path dict
    patterns = []
    loaded_patterns = {}
    patterns_test = []
    loaded_patterns_test = {}
    patterns_train_dir = osp.join(patterns_dir, "train")
    patterns_test_dir = osp.join(patterns_dir, "test")
    patterns_train_dict = {Path(p).stem: osp.join(patterns_train_dir, p) for p in sorted(listdir(patterns_train_dir))}
    patterns_test_dict = {Path(p).stem: osp.join(patterns_test_dir, p) for p in sorted(listdir(patterns_test_dir))}

    # Functions
    def load_pattern(pattern_name, p_dict, loaded, patterns):
        """ If pattern is already loaded, return its index. Otherwise, load it in patterns and return its index. """
        if pattern_name in loaded:
            return loaded[pattern_name]
        try:
            p_path = p_dict[pattern_name]
        except ValueError:
            raise SystemError(f"Pattern '{pattern_name}' not found in {len(p_dict)} patterns")
        pattern = loadImage(p_path)
        patterns.append(pattern)
        loaded[pattern_name] = len(patterns) - 1
        return loaded[pattern_name]

    def create_cam_info(uid, param, image, image_name, pattern_idx, image_mask):
        """ Return a CameraInfo object. """
        return CameraInfo(
            uid=uid,
            view_id=param["view_id"],
            R=param["R"],
            T=param["T"],
            FovY=param["FovY"],
            FovX=param["FovX"],
            K=param["K"],
            image=image,
            image_name=image_name,
            image_mask=image_mask,
            width=param["width"],
            height=param["height"],
            principal_point_ndc=param["principal_point_ndc"],
            pattern_idx=pattern_idx,
            pattern_name=image_name
        )

    def load_procams_info(image_path, pattern_name, patterns_dict, loaded_patterns, patterns, image_mask, uid, param):
        """ 
        Process the information of a pro-cam viewpoint and return a CameraInfo object.

        :param image_path: Path to the image.
        :param pattern_name: Name of the pattern.
        :param patterns_dict: {pattern_name: pattern_path}.
        :param loaded_patterns: {pattern_name: idx of patterns}.
        :param patterns: List of patterns, will be appendent if not loaded.
        :param image_mask: Image mask.
        :param uid: Unique identifier for the camera.
        :param param: Parameters of the camera.
        :return: CameraInfo object.
        """
        pattern_idx = load_pattern(pattern_name, patterns_dict, loaded_patterns, patterns)
        image = loadImage(image_path)
        
        # Ensure 3-channel RGB format
        if image.shape[2] == 4:
            raise ValueError(f"Got 4-channel image at {image_path}, expected 3-channel RGB")
        
        image = image * image_mask
        cam_info = create_cam_info(uid, param, image, Path(image_path).stem, pattern_idx, image_mask)
        return cam_info
    
    def get_ablation_image_indices(num_view, num_images, view_id):
        """
        Get image indices for ablation studies.

        :param num_view: Number of training views
        :param num_images: Number of training images
        :param view_id: Current view ID (1-25)
        :return: List of image indices for the given view_id
        """
        TOTAL_IMAGES = 100
        MAX_VIEWS = 25
        
        # Validate inputs
        if view_id not in range(1, MAX_VIEWS + 1):
            raise ValueError(f"view_id must be between 1 and {MAX_VIEWS}")
            
        # Mode 1: View ablation (fixed num_images=100, variable num_view)
        if num_images == TOTAL_IMAGES:
            if TOTAL_IMAGES % num_view != 0:
                raise ValueError(f"Number of images ({TOTAL_IMAGES}) must be divisible by number of views ({num_view})")
            
            images_per_view = TOTAL_IMAGES // num_view
            start_image = (view_id - 1) * images_per_view + 1  # 1-based indexing
            end_image = start_image + images_per_view - 1
            
            if end_image > TOTAL_IMAGES:
                raise ValueError(f"End image number ({end_image}) exceeds total images ({TOTAL_IMAGES})")
                
            return list(range(start_image - 1, end_image))  # Convert to 0-based
            
        # Mode 2: Image ablation (fixed num_view=25, variable num_images)
        else:
            if num_view != MAX_VIEWS:
                raise ValueError(f"For image ablation, num_view must be {MAX_VIEWS}")
            if num_images not in [75, 50, 25]:
                raise ValueError(f"num_images must be one of [100, 75, 50, 25], got {num_images}")
                
            # Select subset of images based on pattern
            if num_images == 75:
                selected_indices = [i for i in range(TOTAL_IMAGES) if i % 4 < 3]  # Skip every 4th
            elif num_images == 50:
                selected_indices = [i for i in range(TOTAL_IMAGES) if i % 4 < 2]  # Take first 2 of every 4
            elif num_images == 25:
                selected_indices = [i for i in range(TOTAL_IMAGES) if i % 4 == 0]  # Take every 4th
            else:
                selected_indices = list(range(TOTAL_IMAGES))  # Fallback
                
            # Distribute selected images among views
            if len(selected_indices) % MAX_VIEWS != 0:
                raise ValueError(f"Selected images ({len(selected_indices)}) not divisible by views ({MAX_VIEWS})")
                
            images_per_view = len(selected_indices) // MAX_VIEWS
            start_idx = (view_id - 1) * images_per_view
            end_idx = start_idx + images_per_view
            
            return selected_indices[start_idx:end_idx]

    # Load views
    for param in tqdm(params, desc="Loading views", leave=False, dynamic_ncols=True):
        view_id = param["view_id"]
        # The projector is the specific view with the calib_name
        if view_id in [calib_name]:
            prj_info = ProjectorInfo(
                width=param["width"],
                height=param["height"],
                R=param["R"],
                T=param["T"],
                K=param["K"],
                FovX=param["FovX"],
                FovY=param["FovY"],
                principal_point_ndc=param["principal_point_ndc"],
                w_psf=not wo_psf,
            )
            continue

        images_dir = osp.join(path, "views", f"{int(view_id):02d}", "cam", "raw")
        uid = 0
        # Load image mask for this view
        if wo_mask:
            image_mask = np.ones((param["height"], param["width"], 1), dtype=bool)
        else:
            mask_path = osp.join(images_dir, "mask", "mask.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Image mask {mask_path} not found!")
            image_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            image_mask = (image_mask > 0)[..., None]

        # Load images path list
        images_train_dir = osp.join(images_dir, 'train')
        images_test_dir = osp.join(images_dir, 'test')
        images_train_ls = [osp.join(images_train_dir, path) for path in sorted(listdir(images_train_dir))] if os.path.exists(images_train_dir) else []
        images_test_ls = [osp.join(images_test_dir, path) for path in sorted(listdir(images_test_dir))] if os.path.exists(images_test_dir) else []

        is_train = int(view_id) in train_views
        is_valid = int(view_id) in valid_views
        if not (is_train or is_valid):
            continue

        # Load images
        # training view
        if is_train:
            # Get image indices for this view based on ablation mode
            if num_images == 100:
                # Mode 1: Use relative view ID (position in training views)
                relative_view_id = train_views.index(int(view_id)) + 1
                images_train_ids = get_ablation_image_indices(len(train_views), num_images, relative_view_id)
            else:
                # Mode 2: Use absolute view ID (original view ID)
                images_train_ids = get_ablation_image_indices(len(train_views), num_images, int(view_id))
            
            # Convert 0-based indices to 1-based filenames
            images_train_ls = [osp.join(images_train_dir, f"img_1{image_id+1:03d}.png") for image_id in images_train_ids]

            for image_path in tqdm(images_train_ls, desc="    * Loading training images", leave=False, dynamic_ncols=True):
                uid += 1
                cam_info = load_procams_info(image_path, Path(image_path).stem, patterns_train_dict, loaded_patterns, patterns, image_mask, uid, param)
                infos[(False, False)].append(cam_info)

            # Add black pattern-image pair for "Auto-Zeroing"
            patterns_ref_dir = osp.join(patterns_dir, "ref")
            images_ref_dir = osp.join(images_dir, "ref")
            for pattern_name in ["img_0001"]:
                uid += 1
                image_path = osp.join(images_ref_dir, f"{pattern_name}.png")
                cam_info = load_procams_info(image_path, pattern_name, {pattern_name: osp.join(patterns_ref_dir, f"{pattern_name}.png")}, loaded_patterns, patterns, image_mask, uid, param)
                infos[(False, False)].append(cam_info)

            if is_valid: # Load test images for training view
                for image_path in tqdm(images_test_ls, desc="    * Loading train_valid images", leave=False, dynamic_ncols=True):
                    uid += 1
                    cam_info = load_procams_info(image_path, Path(image_path).stem, patterns_test_dict, loaded_patterns_test, patterns_test, image_mask, uid, param)
                    infos[(False, True)].append(cam_info)
        # or validation view
        else: 
            # Load training images for validation view (Uncomment if needed)
            # for image_path in tqdm(images_train_ls, desc="    * Loading valid_train images", leave=False, dynamic_ncols=True):
            #     uid += 1
            #     cam_info = load_procams_info(image_path, Path(image_path).stem, patterns_train_dict, loaded_patterns, patterns, image_mask, uid, param)
            #     infos[(True, False)].append(cam_info)

            # Load test images for validation view
            for image_path in tqdm(images_test_ls, desc="    * Loading valid_valid images", leave=False, dynamic_ncols=True):
                uid += 1
                cam_info = load_procams_info(image_path, Path(image_path).stem, patterns_test_dict, loaded_patterns_test, patterns_test, image_mask, uid, param)
                infos[(True, True)].append(cam_info)


        
    return *tuple(infos.values()), prj_info, patterns, patterns_test
    
def readColmapSceneInfo(path, eval, calib_name, wo_mask, wo_psf, num_view, max_train, num_images):
    def get_train_view_ids(num_view, max_train):
        """
        Get training view IDs for ISMAR'25 ablation study.

        :param num_view: Number of training views to select
        :param max_train: Maximum number of training views available (25 for ISMAR'25)
        :return: List of training view IDs
        :raises ValueError: if num_view or num_images are not in supported configurations
        """

        # Validate ablation study parameters
        # num_view = 25 is equal to num_images = 100
        if not ((num_view == 25 and num_images in [100, 75, 50, 25]) or 
                (num_images == 100 and num_view in [25, 20, 10, 5, 4, 2])):
            raise ValueError("Invalid ablation configuration.")

        all_views = list(range(1, max_train + 1))
        
        # Define view selection strategies
        if num_view == 25:
            return all_views  # Use all views
        elif num_view == 20:
            return [i for i in all_views if i % 5 != 0]  # Skip every 5th view
        elif num_view == 10:
            return [i for i in all_views if i % 5 in {1, 2}]  # Take first 2 of every 5
        elif num_view == 5:
            return [i for i in all_views if (i - 1) % 5 == 0]  # Take every 5th view
        elif num_view == 4:
            return [1, 11, 16, 21]  
        elif num_view == 2:
            return [1, 21] 
        else:
            raise ValueError(f"Unexpected num_view value: {num_view}")
    
    colmap_dir = osp.join(path, "colmap")
    try:
        cameras_extrinsic_file = osp.join(colmap_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = osp.join(colmap_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(colmap_dir, "sparse/0", "images.txt")
        cameras_intrinsic_file = osp.join(colmap_dir, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    params = []
    for key in tqdm(cam_extrinsics, desc="Pre-processing views from COLMAP", leave=False, dynamic_ncols=True):
        # Parameters 
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        view_id = Path(extr.name).stem.split("_")[0]
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
            raise ValueError("COLMAP camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_y, width)
        K = np.array([[focal_length_x, 0.0, cx], [0.0, focal_length_y, cy], [0.0, 0.0, 1.0]])
        principal_point_ndc = np.array([cx / width, cy / height])
        params.append({"view_id": view_id, "width": width, "height": height, "R": R, "T": T, "FovY": FovY, "FovX": FovX, "K": K, "principal_point_ndc": principal_point_ndc})

    num_params = len(params) 
    print(f"Detected total {num_params} camera models from COLMAP") 

    # Get training view indices
    train_views_idxs = get_train_view_ids(num_view, max_train)

    if eval:
        validation_views_idxs = list(range(max_train+1, num_params)) + [1, 6, 11, 16, 21]
    else:
        validation_views_idxs = []

    print(f"Setting training views: {train_views_idxs} | validation views: {validation_views_idxs}")
    if len(train_views_idxs) > len(params): 
        raise ValueError("Number of training views is greater than the number of views in the scene!")
    train_procams_infos, train_procams_test_infos, test_procams_infos, test_procams_test_infos, prj_info, patterns, patterns_test\
        = readColmapCameras(params=params, path=path, train_views=train_views_idxs, valid_views=validation_views_idxs,
                            calib_name=calib_name, wo_mask=wo_mask, wo_psf=wo_psf, num_images=num_images)
    print(f"Loaded num of train_procams: {len(train_procams_infos)} | num of train_procams_test: {len(train_procams_test_infos)}" +
          f"| num of test_procams: {len(test_procams_infos)} | num of test_procams_test: {len(test_procams_test_infos)}")
    
    nerf_normalization = getNerfppNorm(train_procams_infos)

    ply_path = osp.join(colmap_dir, "sparse/0/points3D.ply")
    bin_path = osp.join(colmap_dir, "sparse/0/points3D.bin")
    txt_path = osp.join(colmap_dir, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_procams_infos,
                           test_cameras=test_procams_infos,
                           train_cameras_test=train_procams_test_infos,
                           test_cameras_test=test_procams_test_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           prj_info=prj_info,
                           patterns=patterns,
                           patterns_test=patterns_test)
    return scene_info

def readNepmapCameras(path, transformsfile, eval, white_background, wo_mask, wo_psf, divide_res=True):
    """ Nepmap Synthetic Dataset """
    def load_cameraInfo(uid, view_id, cam_K, c2w, image_path, pattern_idx, white_background=True, wo_mask=False, divide_res=True):
        image_name = Path(image_path).stem
        image = loadImage(image_path)
        image_height, image_width = image.shape[:2]
        alpha = image[..., 3:4]
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        image = image[..., :3] * alpha + bg * (1 - alpha)
        if wo_mask:
            image_mask = np.ones([image_height, image_width, 1], dtype=bool)
        else:
            image_mask = alpha > 0

        if divide_res:
            # resize image to half resolution following nepmap
            # https://github.com/yoterel/nepmap/blob/master/data/data_loader.py#L352-L366
            cam_K = cam_K // 2
            cam_K[2, 2] = 1.0
            image_width, image_height = image_width // 2, image_height // 2
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            image_mask_tensor = torch.from_numpy(image_mask).permute(2, 0, 1).unsqueeze(0).float()
            image = torch.nn.functional.interpolate(image_tensor, size=(image_width, image_height), mode="bilinear")
            image_mask = torch.nn.functional.interpolate(image_mask_tensor, size=(image_width, image_height), mode="bilinear")
            image = image.squeeze(0).permute(1, 2, 0).numpy()
            image_mask = image_mask.squeeze(0).permute(1, 2, 0).numpy().astype(bool)

        fovx = focal2fov(cam_K[0, 0], image_width)
        fovy = focal2fov(cam_K[1, 1], image_height)
        c2w = np.array(frame["blender_matrix_world"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        principal_point_ndc = np.array([cam_K[0, 2] / image_width, cam_K[1, 2] / image_height])
                        
        return CameraInfo(
            uid=uid,
            view_id=view_id,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            K=cam_K,
            image=image,
            image_name=image_name,
            image_mask=image_mask,
            width=image_width,
            height=image_height,
            principal_point_ndc=principal_point_ndc,
            pattern_idx=pattern_idx,
            pattern_name=image_name
        )

    cam_infos = []
    # Patterns
    patterns = []
    patterns_dir = osp.join(path, "projector")
    patterns_paths = sorted(os.listdir(patterns_dir))
    patterns_paths = [p for p in patterns_paths if Path(p).stem not in ["all_white"]]
    patterns_names = [Path(p).stem for p in patterns_paths]
    for pattern_path in tqdm(patterns_paths, desc="Loading patterns", leave=False, dynamic_ncols=True):
        patterns.append(loadImage(osp.join(patterns_dir, pattern_path)))

    # Projector, cameras, and images
    with open(transformsfile, 'r') as file:
        contents = json.load(file)
        cam_K = np.array(contents["K_cam"])

        ## Projector
        prj_K = np.array(contents["K_proj"])
        prj_c2w = np.array(contents["blender_matrix_world_proj"]) 
        prj_c2w[:3, 1:3] *= -1 # OpenGL to OpenCV
        prj_w2c = np.linalg.inv(prj_c2w)
        prj_R = np.transpose(prj_w2c[:3, :3]) # 3DGS' transpose
        prj_T = prj_w2c[:3, 3]
        prj_height, prj_width = patterns[0].shape[:2]
        prj_fovx = focal2fov(prj_K[0, 0], prj_width)
        prj_fovy = focal2fov(prj_K[1, 1], prj_height)
        prj_principal_point_ndc = np.array([prj_K[0, 2]/ prj_width, prj_K[1, 2] / prj_height])
        prj_info = ProjectorInfo(width=prj_width, height=prj_height, R=prj_R, T=prj_T, K=prj_K,
                                FovX=prj_fovx, FovY=prj_fovy, principal_point_ndc=prj_principal_point_ndc,
                                w_psf=not wo_psf)

        ## Cameras and images
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, desc="Loading training views", leave=False, dynamic_ncols=True)):
            pattern_name = Path(frame["patterns"][0]).stem
            if pattern_name not in ["all_white"]: 
                pattern_idx = patterns_names.index(pattern_name)
                cam_infos.append(load_cameraInfo(uid=idx, view_id=frame["view_id"], cam_K=cam_K, c2w=frame["blender_matrix_world"],
                                                image_path=osp.join(path, frame["file_path"]), pattern_idx=pattern_idx, white_background=white_background,
                                                wo_mask=wo_mask, divide_res=divide_res))

        if eval:
            num_view_cams = len(cam_infos)
            test_dir = path + "_random"
            test_cam_infos = []
            patterns_test = []
            patterns_test_dir = osp.join(test_dir, "projector")
            # test_frames_ls is from https://github.com/yoterel/nepmap/issues/4
            test_frames_ls = [110,74,179,167,50,92,278,272,251,5,221,71,107,17,257,56,182,44,287,269,29,188,77,212,62,83,227,233,14,32,176,68,170,101,173,206]
            transformsfile = osp.join(test_dir, "transforms.json")
            img2texfile = osp.join(test_dir, "img2tex.json")
            with open(transformsfile, 'r') as file:
                with open(img2texfile, 'r') as file2:
                    img2tex = json.load(file2)
                    contents = json.load(file)
                    frames = contents["frames"]
                    idx = 0
                    for i, frame in enumerate(tqdm(frames, desc="Loading test views", leave=False, dynamic_ncols=True)):
                        if int(frame["view_id"]) in test_frames_ls:
                            image_path = osp.join(test_dir, frame["file_path"])
                            image_name = Path(frame["file_path"]).stem
                            pattern_name = img2tex[image_name + ".png"][0][0]
                            patterns_test.append(loadImage(osp.join(patterns_test_dir, pattern_name + ".png")))
                            test_cam_infos.append(load_cameraInfo(uid=idx+num_view_cams, view_id=frame["view_id"], cam_K=cam_K, c2w=frame["blender_matrix_world"],
                                                        image_path=image_path, pattern_idx=idx, white_background=white_background,
                                                        wo_mask=wo_mask, divide_res=divide_res))
                            idx += 1
            
    return cam_infos, test_cam_infos, prj_info, patterns, patterns_test

def readNepmapSceneInfo(path, eval, white_background=True, wo_mask=False, wo_psf=True):
    # GS-ProCams BRDF w/o psf, w/ white_background
    transform_path = osp.join(path, "transforms.json")
    train_cameras, test_cameras_test, prj_info, patterns, patterns_test = readNepmapCameras(path, transform_path, eval, white_background, wo_mask, wo_psf)
    print(f"Loaded num of train_cameras: {len(train_cameras)} | num of test_cameras_test: {len(test_cameras_test)}")

    nerf_normalization = getNerfppNorm(train_cameras)

    ply_path = osp.join(path, "point_cloud.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                                train_cameras=train_cameras,
                                test_cameras=[],
                                train_cameras_test=[],
                                test_cameras_test=test_cameras_test,
                                nerf_normalization=nerf_normalization,
                                ply_path=ply_path,
                                prj_info=prj_info,
                                patterns=patterns,
                                patterns_test=patterns_test)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap" : readColmapSceneInfo,
    "Nepmap" : readNepmapSceneInfo
}