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
import cv2
from utils.system_utils import torch_compile

@torch_compile
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch_compile
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img

def vis_normalmap(normalmap):
    normal_transformed = torch.where(normalmap == 0.0, torch.tensor(0.0), normalmap * 0.5 + 0.5)
    return normal_transformed

@torch_compile
def srgb_to_linear(srgb, eps=None):
    """See https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
        eps = torch.tensor(eps, dtype=srgb.dtype, device=srgb.device)
    threshold = 0.04045
    low = srgb / 12.92
    high = ((srgb + 0.055) / 1.055) ** 2.4
    return torch.where(srgb <= threshold, low, high)

@torch_compile
def linear_to_srgb(linear, eps=None):
    """See https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
        eps = torch.tensor(eps, dtype=linear.dtype, device=linear.device)
    threshold = 0.0031308
    low = 12.92 * linear
    high = 1.055 * torch.max(linear, eps) ** (1/2.4) - 0.055
    return torch.where(linear <= threshold, low, high)

def loadImage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image/255.0 # Normalize to [0, 1]
