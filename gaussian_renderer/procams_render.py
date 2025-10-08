import torch
import math
import torch.nn.functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depths_to_points, points_to_normal, warp_points
from utils.loss_utils import l1_loss, ssim,  bilateral_smooth_loss
from utils.image_utils import linear_to_srgb
from collections import namedtuple
from utils.system_utils import torch_compile

PipelineParams = namedtuple("PipelineParams", ["convert_SHs_python", "compute_cov3D_python", "depth_ratio", "debug"])

@torch_compile
def BRDF(light_dir, view_dir, normal, base_color, roughness, F0=0.04):
    r'''
    Args:
        light: Incident light projector (# (3, H, W))
        light_dir: Incident light directions (# (H, W, 3)) 
        view_dir: View directions (# (H, W, 3))
        normal: Surface normals (# (3, H, W))
        base_color: Base color (# (3, H, W))
        roughness: Surface roughness (# (H, W))

    Returns:
        brdf_factor: the shading factor (# (3, H, W))
    '''

    # Diffuse term
    diffuse = base_color / math.pi 

    # Specular term
    L = light_dir.permute(2, 0, 1) # (3, H, W)
    V = view_dir.permute(2, 0, 1) # (3, H, W)
    H = F.normalize(L + V, dim=0) # (3, H, W)
    N = normal 

    NdotV = torch.sum(V * N, dim=0, keepdim=True) # (1, H, W)
    N = N * NdotV.sign() # (3, H, W)

    def _dot(a, b):
        # a, b: (3, H, W)
        # return: (1, H, W)
        return torch.sum(a * b, dim=0, keepdim=True).clamp(1e-6, 1)

    NdotL = _dot(N, L)
    NdotV = _dot(N, V)
    NdotH = _dot(N, H)
    VdotH = _dot(V, H)

    alpha = roughness ** 2
    alpha2 = alpha ** 2
    k = (alpha + 2 * roughness + 1) / 8.0 #（roughness + 1）^2 / 8

    frac = alpha2 * (F0 + (1 - F0) * torch.pow(2.0, ((-5.55473) * VdotH - 6.98316) * VdotH)) # frac_D * frac_F
    nom_D = (math.pi * (NdotH ** 2 * (alpha2 - 1) + 1) ** 2)
    nom_V = 4 * (NdotV * (1 - k) + k) * (NdotL * (1 - k) + k)
    nom = (nom_D * nom_V).clamp(1e-6, 4 * math.pi)
    specular = frac / nom

    # lighting
    brdf_factor = diffuse + specular
    return brdf_factor

def caculate_loss(viewpoint_camera, pc, results, opt):
    log_items = {"num_points": pc.get_xyz.shape[0]}

    image_mask = viewpoint_camera.gt_alpha_mask
    gt_image = viewpoint_camera.original_image

    image = results["render"] 
    surf_normal = results["surf_normal"]
    render_normal = results["render_normal"] 
    base_color = results["base_color"] 
    roughness = results["roughness"] 
    render_dist = results["render_dist"] 
    render_opacity = results["render_alpha"] 

    Ll1 = l1_loss(image, gt_image)
    Lssim = ssim(image, gt_image)

    log_items.update({
        "l1": Ll1,
        "ssim": Lssim,
    })

    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)

    if opt.lambda_roughness_smooth > 0:
        loss_roughness_smooth = bilateral_smooth_loss(roughness, base_color.detach(), image_mask)
        log_items["loss_roughness_smooth"] = loss_roughness_smooth
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

    if opt.normal_consistency_loss > 0:
        normal_consistency_loss = ((1 - (surf_normal * render_normal).sum(dim=0))[None]).mean()
        log_items["normal"] = normal_consistency_loss
        loss = loss + opt.lambda_normal_consistency  * normal_consistency_loss
        
    if opt.lambda_depth_distortion > 0:
        dist_loss = render_dist.mean()
        log_items["dist"] = dist_loss
        loss = loss + opt.lambda_depth_distortion * dist_loss
    
    if opt.lambda_mask_entropy > 0:
        o = render_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        log_items["loss_mask_entropy"] = loss_mask_entropy
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

        
    log_items["loss"] = loss
    return loss, log_items

def render(viewpoint_camera, pc:GaussianModel, pipe=None, bg_color=None, procams_dict=None, scaling_modifier=1.0, override_color=None, training=False, opt=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # If pipe is None, set it to default.
    if pipe is None: pipe = PipelineParams(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    # If background color is not provided, set it to black.
    if bg_color is None: bg_color = torch.zeros(3, device="cuda")
    # If procams_dict is None, only render the scene from the viewpoint_camera.
    if procams_dict is None:
        if training: raise ValueError("procams_dict must be provided for training.")
        projector = pattern = None
    else:
        projector = procams_dict.get("projector")
        pattern = procams_dict.get("pattern")
        if projector is None or pattern is None: raise ValueError("procams_dict must contain both projector and pattern.")
    # If training is True, opt must be provided.
    if training and opt is None: raise ValueError("opt must be provided for training.")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # BRDF attributes
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    features = torch.cat([base_color, roughness], dim=-1)
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    render_shs, radii, render_features, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    render_base_color, render_roughness = torch.split(render_features, [3, 1], dim=0)
    
    render_depth_expected, render_alpha, render_normal, render_depth_median, render_dist \
        = torch.split(allmap[:7], [1, 1, 3, 1, 1], dim=0) 
    
    # Disk normal
    # render_normal_vis (OpenGL Camera Space) # https://github.com/hbb1/2d-gaussian-splatting/issues/68
    render_normal_vis = None if training else -render_normal
    # transform normal from view space to world space
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0, eps=1e-3)

    # Depth
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # Surface normal   
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations
    surf_pts3d = depths_to_points(viewpoint_camera, surf_depth)
    surf_normal = points_to_normal(surf_pts3d).permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()
    # surf_normal_vis (OpenGL Camera Space) # https://github.com/hbb1/2d-gaussian-splatting/issues/68
    surf_normal_vis = None if training else -(surf_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3])).permute(2,0,1)

    # Render the image
    if procams_dict is None: 
        # render the scene from the viewpoint_camera, may be used for visualization
        results = {"depth": surf_depth,
                   "render_normal": render_normal_vis,
                   "surf_normal": surf_normal_vis,
                   "base_color": linear_to_srgb(render_base_color),
                   "render_shs": linear_to_srgb(render_shs),
                   "roughness": render_roughness}
    else:
        # light dirs
        l = F.normalize(projector.camera_center - surf_pts3d, dim=-1) # (H, W, 3) towards projector
        v = F.normalize(viewpoint_camera.camera_center - surf_pts3d, dim=-1) # (H, W, 3) towards camera
        # geometric mapping
        prj_pts2d, _ = warp_points(projector, surf_pts3d) # (H, W, 2/3)
        prj2cam_grid = prj_pts2d * 2  / projector.prj_size - 1 # -> (-1, 1)
        Ip_out = projector(pattern)
        Ip_out = F.grid_sample(Ip_out.unsqueeze(0), prj2cam_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False).squeeze() # (3, H, W)
        # brdf rendering
        brdf_factor = BRDF(l, v, surf_normal, render_base_color, render_roughness)
        render_image = brdf_factor * Ip_out + render_shs
        render_image = linear_to_srgb(render_image).clamp(0.0, 1.0)

        results = {"render": render_image}
        if training:
            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            results.update({"viewspace_points": means2D,
                            "visibility_filter" : radii > 0,
                            "radii": radii,
                            "render_alpha": render_alpha,
                            "render_normal": render_normal,
                            "render_dist": render_dist,
                            "surf_normal": surf_normal,
                            "base_color": render_base_color,
                            "roughness": render_roughness})
            loss, log_items = caculate_loss(viewpoint_camera, pc, results, opt)
            log_items = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in log_items.items()}
            results.update({"loss": loss, "log_items": log_items})
        else:
            # may be used for projector compensation and visualization
            results.update({"Ip_out": Ip_out, 
                            "brdf_factor": brdf_factor,
                            "render_shs": render_shs,
                            "prj2cam_grid": prj2cam_grid})
    return results