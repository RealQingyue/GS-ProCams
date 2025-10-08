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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._root = ""
        self._setup = ""
        self._model_path = ""
        self.resolution = -1
        self._white_background = False
        self._calib_name = "calib"

        self.data_device = "cuda"
        self.gpu_id = 0
        
        self.novel = False
        # num_view, max_train, num_images, wo_mask, wo_psf are all for ISMAR'25 ablation study
        self.num_view = 25 # viewpoints for train
        self.max_train = 25 # the maximum id of viewpoints for train，then the view_ids for train will be determined by this and num_view
        self.num_images = 100 # the number of images/patterns for train, this will determine the names/ids of the images/patterns for each view
        # In the paper, we only change the num_view or num_images at a time, and keep the another one as default value
        self.wo_mask = False
        self.wo_psf = False

        self.save_interval = 0 # 0 or int
        self.report_interval = 0 # 0 or int
        self.evaluate = False # whether to run evaluation report at the end of training
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.root = os.path.abspath(g.root)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025

        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.normal_lr = 0.01
        self.base_color_lr = 0.01
        self.roughness_lr = 0.01
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        self.lambda_depth_distortion = 0.0
        self.lambda_normal_consistency = 0.0
        self.lambda_dist = 1000.0 
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        self.lambda_mask_entropy = 0.1

        self.PEModel_lr = 1e-3

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.lambda_dssim = 0.2
        self.lambda_roughness_smooth = 0.002
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
