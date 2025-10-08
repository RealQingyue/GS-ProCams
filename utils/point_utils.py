import torch
from utils.system_utils import torch_compile

@torch_compile
def depths_to_points(view, depthmap):
    c2w = view.c2w
    intrins = view.K
    grid_x, grid_y = torch.meshgrid(torch.arange(view.image_width, device='cuda', dtype=torch.float), torch.arange(view.image_height, device='cuda', dtype=torch.float), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3] # view.camera_center
    points = depthmap.reshape(-1, 1) * rays_d + rays_o 
    return points.reshape(*depthmap.shape[1:3], 3) # pts3d (H, W, 3)

@torch_compile
def points_to_normal(points):
    r"""
    Args:
        view: view camera
        points: 3D point per pixel
    Returns:
        psedo_normal: (H, W, 3)
    """
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output 

@torch_compile
def depth_to_normal(view, depth):
    """
    Args:
        view: view camera
        depth: depthmap 
    Returns:
        psedo_normal: (H, W, 3)
    """
    points = depths_to_points(view, depth)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

@torch_compile
def warp_points(view, points):
    """
    Args:
        view: src view camera
        points: 3D points in dst camera frame (H, W, 3)
    Returns:
        prj2cam_grid (H, W, 2)
    """
    H, W, _ = points.shape
    pts3d = points.view(-1, 3)
    pts3d_homo = torch.cat((pts3d, torch.ones_like(pts3d[:, :1])), dim=-1)

    view_pts3d = pts3d_homo @ view.world_view_transform[:, :3] # (H*W, 4) @ (4, 3) = (H*W, 3)

    uvw = view_pts3d @ view.K.T # (H*W, 3) @ (3, 3) = (H*W, 3)
    uv = uvw[:, :2] / uvw[:, 2:3] # de-homo, raster/pixel space (H*W, 2)
    return uv.view(H, W, 2), view_pts3d.view(H, W, 3)