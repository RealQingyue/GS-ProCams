import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrixShift
from utils.system_utils import mkdir_p


class Projector(nn.Module):
    def __init__(self, R, T, K, FoVx, FoVy, height, width, principal_point_ndc, w_psf=True,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ):
        super(Projector, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = height 
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
                                                          width=self.image_width, height=self.image_height, principal_point_ndc=principal_point_ndc).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
       
        self.c2w = (self.world_view_transform.T).inverse()
        self.K = torch.from_numpy(K).float().cuda()
        self.prj_size = torch.tensor([width, height], dtype=torch.float, device=data_device) # (x, y)
        self.pem = self._create_pem(w_psf).to(self.data_device)
    
    def forward(self, pattern):
        ouput = self.pem(pattern.unsqueeze(0)).squeeze(0)
        return ouput

    def _create_pem(self, w_psf):
        return PEModel(self.image_height, self.image_width, 3, 3, w_psf=w_psf, kernel_size=5)

    def training_setup(self, lr):
        self.optimizer = torch.optim.Adam(self.pem.parameters(), lr=lr)
            
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _capture(self):
        capture_dict = self.pem.capture()
        capture_dict["optimizer"] = self.optimizer.state_dict()
        return capture_dict

    def _restore(self, prj_params, weights_only):
        weights = {k: v for k, v in prj_params.items() if "optimizer" not in k}
        self.pem.restore(weights)
        if weights_only: return
        self.training_setup(lr=0.0)
        self.optimizer.load_state_dict(prj_params["optimizer"])
    
    def save_ckpt(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self._capture(), path)
    
    def load_ckpt(self, path, weights_only=False):
        if not os.path.exists(path): raise FileNotFoundError(f'Can not find checkpoint file: {path}')
        ckpt = torch.load(path, weights_only=weights_only, map_location=self.data_device)
        self._restore(ckpt, weights_only=weights_only)
    
class PEModel(nn.Module):
    def __init__(self, H, W, C, C_out, w_psf=True, kernel_size=5):
        super(PEModel, self).__init__()
        self.H = H
        self.W = W 
        self.C = C 
        self.C_out = C_out
        self.gamma = nn.Parameter(torch.tensor(2.2))
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.psf = None if not w_psf else self._create_psf()
        if self.psf is not None: self.psf.apply(lambda x: nn.init.constant_(x.weight, 1 / (kernel_size**2)) if isinstance(x, nn.Conv2d) else None)

    def forward(self, x):
        x = self._srgb_to_linear(x) * self.gain
        if self.psf is not None: x = self._psf(x)
        return x
    
    def _create_psf(self):
        psf = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        psf.apply(lambda x: nn.init.constant_(x.weight, 1 / 25) if isinstance(x, nn.Conv2d) else None)
        return psf

    def _srgb_to_linear(self, x):
        return torch.pow(x, self.gamma)
    
    def _psf(self, x):
        x = self.psf(x.transpose(0, 1)).transpose(0, 1)
        return x

    def capture(self):
        params_dict = {
            "gamma": self.gamma,
            "gain": self.gain,
            "psf": None if self.psf is None else self.psf.state_dict(),
        }
        return params_dict
    
    def restore(self, prj_params):
        self.gamma = prj_params["gamma"]
        self.gain = prj_params["gain"]
        if prj_params["psf"] is not None: 
            if self.psf is None: 
                raise ValueError("Projector psf is None, but parameters contain psf state_dict.")
            self.psf.load_state_dict(prj_params["psf"])
