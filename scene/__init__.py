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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, projector_from_prjInfo, projector_to_JSON, patternsTensor_from_numpy
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        eval = True if args.report_interval or args.evaluate else False
        source_path = os.path.join(args.root, "setups", args.setup)
        print(f"Loading scene from {source_path}")
        if os.path.exists(os.path.join(source_path, "colmap")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, eval, args.calib_name, args.wo_mask, args.wo_psf, args.num_view, args.max_train, args.num_images)
            self.type = "Colmap"
        elif os.path.exists(os.path.join(source_path, "img2tex.json")):
            print("Found img2tex.json, assuming Nepmap synthetic data set!")
            scene_info = sceneLoadTypeCallbacks["Nepmap"](source_path, eval, args.white_background, args.wo_mask, args.wo_psf)
            self.type = "Nepmap"
        else:
            assert False, "Could not recognize scene type!"

        self._train_cameras = {}
        self._test_cameras = {}
        self._train_cameras_test = {}
        self._test_cameras_test = {}

        if not self.loaded_iter:
            # Save point cloud to PLY
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # Save camera views info to JSON
            json_cams = []
            camlist = scene_info.train_cameras + scene_info.train_cameras_test + scene_info.test_cameras + scene_info.test_cameras_test
            viewsdict = {}
            for camera in camlist:
                if camera.view_id not in viewsdict:
                    viewsdict[camera.view_id] = camera
            camlist = list(viewsdict.values()) 

            for id, cam_info in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam_info))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            
            # Save projector info to JSON
            json_prj = []
            json_prj.append(projector_to_JSON(scene_info.prj_info))
            with open(os.path.join(self.model_path, "projector.json"), 'w') as file:
                json.dump(json_prj, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        

        self.projector = projector_from_prjInfo(scene_info.prj_info) 
        self.patterns = patternsTensor_from_numpy(scene_info.patterns)

        for resolution_scale in resolution_scales:
            self._train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, "Training Cameras")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", f"iteration_{str(self.loaded_iter)}", "point_cloud.ply"))
            self.projector.load_ckpt(os.path.join(self.model_path, f"procams/iteration_{self.loaded_iter}/procams.ckpt"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Dynamic loading of test cameras and patterns
        self.args = args
        self.patterns_test_info = scene_info.patterns_test
        self.train_cameras_test_info = scene_info.train_cameras_test
        self.test_cameras_info = scene_info.test_cameras
        self.test_cameras_test_info = scene_info.test_cameras_test

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), save_pbr=self.gaussians.pbr)
        ckpt_path = os.path.join(self.model_path, f"procams/iteration_{iteration}/procams.ckpt")
        self.projector.save_ckpt(ckpt_path)

    @property
    def train_cameras_train(self, scale=1.0):
        return self._train_cameras[scale]

    @property
    def train_cameras_test(self, scale=1.0):
        if scale in self._train_cameras_test: return self._train_cameras_test[scale] 
        self._train_cameras_test[scale] = cameraList_from_camInfos(self.train_cameras_test_info, scale, self.args, "Training Cameras Test")
        return self._train_cameras_test[scale]

    @property
    def test_cameras_train(self, scale=1.0):
        if scale in self._test_cameras: return self._test_cameras[scale]
        self._test_cameras[scale] = cameraList_from_camInfos(self.test_cameras_info, scale, self.args, "Test Cameras")
        return self._test_cameras[scale]

    @property
    def test_cameras_test(self, scale=1.0):
        if scale in self._test_cameras_test: return self._test_cameras_test[scale]
        self._test_cameras_test[scale] = cameraList_from_camInfos(self.test_cameras_test_info, scale, self.args, "Test Cameras Test")
        return self._test_cameras_test[scale]
    
    @property
    def patterns_test(self):
        if hasattr(self, '_patterns_test'): return self._patterns_test
        self._patterns_test = patternsTensor_from_numpy(self.patterns_test_info)
        return self._patterns_test
    
