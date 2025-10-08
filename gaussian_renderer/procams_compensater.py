import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, linear_to_srgb
from utils.point_utils import depths_to_points, warp_points
from . import render
from typing import NamedTuple
from scene import GaussianModel
from scene.projector import Projector
from scene.cameras import MicroCam

class CompensationOptions(NamedTuple):
    max_iter:int = 100
    lr:float = 0.02
    lambda_huber:float = 1.0
    cmp_init:bool = True
    prj_mask:bool = False

class Compensater:
    """ Compensater for GS-ProCams"""
    @torch.no_grad()
    def __init__(self, gaussians:GaussianModel, camera:MicroCam, projector: Projector, cmp_option:CompensationOptions, cam_mask=None, simplified=True, logger=None):
        self.gaussians = gaussians
        self.camera = camera
        self.projector = projector
        self.cmp_option = cmp_option
        self.logger = logger
        self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

        # Close the gradients for gaussians and projector
        for _, param in gaussians.__dict__.items(): 
            if isinstance(param, torch.nn.Parameter): 
                param.requires_grad = False
        for param in projector.parameters():
            param.requires_grad = False
        
        # Initialize the compensation option
        ## compute cam2prj_grid according to the depth map in projector's view
        prj_depth = render(projector, gaussians)['depth']
        prj_surf_pts3d = depths_to_points(projector, prj_depth)
        campts_2d, _ = warp_points(camera, prj_surf_pts3d)
        cam2prj_grid = campts_2d * 2  / torch.tensor([camera.image_width, camera.image_height], dtype=torch.float32, device="cuda") - 1
        
        ## initialize the prj_cmp_opt with desired image
        self.init_grid = None if not self.cmp_option.cmp_init else cam2prj_grid
        
        ## initialize the prj_mask with cam_mask
        if not self.cmp_option.prj_mask:
            self.prj_mask = None 
        else:
            if cam_mask is None: raise ValueError("cam_mask is required when prj_mask is True")
            self.prj_mask =  F.grid_sample(cam_mask.float().unsqueeze(0), cam2prj_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0) > 0
        
        ## pre-compute the BRDF factor
        if not simplified:
            self.simplified = False
            self.brdf_factor = None
            self.residual = None
        else:
            self.simplified = True
            procams_dict_nothing = {
                "projector": self.projector,
                "pattern": torch.ones((3, projector.image_height, projector.image_width), dtype=torch.float32, device="cuda")
            }
            render_pkg = render(self.camera, self.gaussians, pipe=None, bg_color=self.bg_color, procams_dict=procams_dict_nothing)
            self.brdf_factor = render_pkg["brdf_factor"]
            self.residual = render_pkg["render_shs"]
            self.prj2cam_grid = render_pkg["prj2cam_grid"]

    def _init_prj_cmp_opt(self, target):
        """ Initialize the compensated pattern for optimization """
        init_value = torch.tensor(0.1, dtype=target.dtype, device=target.device)
        if self.init_grid is not None:
            prj_cmp_init = 0.5 * F.grid_sample(target.unsqueeze(0), self.init_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()
            prj_cmp_init = torch.where(prj_cmp_init < init_value, init_value, prj_cmp_init)
        else:
            prj_cmp_init = torch.full((3, self.projector.image_height, self.projector.image_width), init_value, dtype=torch.float32, device="cuda", requires_grad=False)

        # apply the prj_mask
        if self.prj_mask is not None: prj_cmp_init = prj_cmp_init * self.prj_mask

        return torch.nn.Parameter(prj_cmp_init)

    def _visualize(self, target):
        fig = plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.subplot(1, 3, 1)
        plt.title("Compensated (rendered)")
        plt.axis('off')
        ax_render = plt.imshow((target).detach().cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.title("Target")
        plt.axis('off')
        plt.imshow((target).detach().cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 3, 3)
        plt.title("Compensated pattern")
        plt.axis('off')
        ax_prj_cmp = plt.imshow(target.detach().cpu().numpy().transpose(1, 2, 0))
        fig.show()
        return fig, ax_render, ax_prj_cmp
    
    def _render(self, prj_cmp):
        if not self.simplified:
            procams_dict = {"projector": self.projector, "pattern": prj_cmp}
            render_image = render(self.camera, self.gaussians, pipe=None, bg_color=self.bg_color, procams_dict=procams_dict)['render']
        else:
            Ip_out = self.projector(prj_cmp)
            Ip_out = F.grid_sample(Ip_out.unsqueeze(0), self.prj2cam_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze() # (3, H, W)
            render_image = self.brdf_factor * Ip_out + self.residual 
            render_image = linear_to_srgb(render_image).clamp(0.0, 1.0)
        return render_image

    def compensate(self, cam_desired, vis=False):
        target = cam_desired
        prj_cmp_opt = self._init_prj_cmp_opt(target)

        optimizer = torch.optim.Adam([prj_cmp_opt], lr=self.cmp_option.lr, weight_decay=0)

        if vis:
            fig, ax_render, ax_prj_cmp = self._visualize(target)

        progress_bar = tqdm(range(1, self.cmp_option.max_iter+1), desc='Compensating', leave=False, dynamic_ncols=True)
        for i in progress_bar:
            prj_cmp = torch.clamp(prj_cmp_opt, 0.0, 1.0)
            render_image = self._render(prj_cmp)
            loss = 0.0
            log_items = {}
            if self.cmp_option.lambda_huber > 0:
                huber = F.smooth_l1_loss(render_image, target)
                loss += self.cmp_option.lambda_huber * huber
                log_items["huber"] = huber.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if vis:
                    Ll1 = l1_loss(render_image, target)
                    Lssim = ssim(render_image, target)
                    log_items = {
                        "l1": Ll1.item(),
                        "ssim": Lssim.item(),
                        "psnr": psnr(render_image, target).mean().item()
                    }
                    
                    ax_render.set_data(render_image.detach().cpu().numpy().transpose(1, 2, 0))
                    ax_prj_cmp.set_data(prj_cmp.detach().cpu().numpy().transpose(1, 2, 0))
                    plt.pause(0.01)
                progress_bar_dict = {item: f"{log_items[item]:.{3}f}" for item in log_items}
                progress_bar.set_postfix(progress_bar_dict)
        
        Ll1 = l1_loss(render_image, target)
        Lssim = ssim(render_image, target)
        log_items = {
            "l1": Ll1.item(),
            "ssim": Lssim.item(),
            "psnr": psnr(render_image, target).mean().item()
        }
    
        if vis:plt.close()
        return prj_cmp, render_image, log_items