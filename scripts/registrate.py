import os
import os.path as osp
import shutil
import argparse
import logging
import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

def run_colmap(args):
    """Run full COLMAP SfM pipeline (feature extraction, matching, mapping, bundle adjust)."""
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else 'colmap'
    use_gpu = 1 if not args.no_gpu else 0

    setup_dir = osp.join(args.root, 'setups', args.setup)
    colmap_dir = osp.join(setup_dir, 'colmap')

    if not args.skip_copy:
        # copy images into colmap/ref/Camera and projector into colmap/ref/Projector
        colmap_prj_dir = osp.join(colmap_dir, 'ref', 'Projector')
        colmap_cam_dir = osp.join(colmap_dir, 'ref', 'Camera')
        os.makedirs(colmap_prj_dir, exist_ok=True)
        os.makedirs(colmap_cam_dir, exist_ok=True)
        prj_cap_path = osp.join(args.root, 'patterns', 'calib', f'{args.texture}{args.extention}')

        views_dir = osp.join(setup_dir, 'views')
        cam_cap_paths = []
        for view_id in range(1, args.views+1):
            cam_cap_dir = osp.join(views_dir, f"{view_id:02d}", 'cam', 'raw', 'calib')
            cam_cap_path = osp.join(cam_cap_dir, f"{args.texture}{args.extention}")
            if not os.path.exists(cam_cap_path):
                logging.error(f"Camera image not found: {cam_cap_path}")
                raise FileNotFoundError(cam_cap_path)
            shutil.copyfile(cam_cap_path, osp.join(colmap_cam_dir, f"{view_id:02d}{args.extention}"))
            cam_cap_paths.append(cam_cap_path)

        prj = osp.join(colmap_prj_dir, f'{args.texture}{args.extention}')
        shutil.copyfile(prj_cap_path, prj)
        print(f"{len(cam_cap_paths)} images copied to {colmap_dir}")

    if not args.skip_matching:
        os.makedirs(colmap_dir + "/sparse", exist_ok=True)

        feat_extracton_cmd = (
            f"{colmap_command} feature_extractor "
            f"--database_path {colmap_dir}/database.db "
            f"--image_path {colmap_dir}/ref "
            f"--ImageReader.single_camera_per_folder 1 "
            f"--ImageReader.camera_model {args.camera} "
            f"--SiftExtraction.use_gpu {use_gpu}"
        )
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
        feat_matching_cmd = (
            f"{colmap_command} exhaustive_matcher "
            f"--database_path {colmap_dir}/database.db "
            f"--SiftMatching.use_gpu {use_gpu}"
        )
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path {colmap_dir}/database.db "
            f"--image_path {colmap_dir}/ref "
            f"--output_path {colmap_dir}/sparse "
            f"--Mapper.ba_global_function_tolerance=0.000001"
        )
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)
        refine_cmd = (
            f"{colmap_command} bundle_adjuster "
            f"--input_path {colmap_dir}/sparse/0 "
            f"--output_path {colmap_dir}/sparse/0 "
            f"--BundleAdjustment.refine_principal_point 1"
        )
        exit_code = os.system(refine_cmd)
        if exit_code != 0:
            logging.error(f"Refine failed with code {exit_code}. Exiting.")
            exit(exit_code)

def register_novel_view(args):
    """Register a novel view into an existing COLMAP sparse model."""
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else 'colmap'
    use_gpu = 1 if not args.no_gpu else 0

    # use provided setup_dir
    setup_dir = osp.join(args.root, 'setups', args.setup)
    colmap_dir = osp.join(setup_dir, 'colmap')

    new_model_path = osp.join(colmap_dir, 'sparse', '0')
    new_img_name = f"{args.view_id:02d}{args.extention}"
    image_list_file = osp.join(new_model_path, 'novel', "image_list.txt")
    os.makedirs(osp.join(new_model_path, 'novel', 'imgs'), exist_ok=True)
    with open(image_list_file, 'w') as f:
        f.write(new_img_name + "\n")

    new_img_src = osp.join(setup_dir, 'views', f"{args.view_id:02d}", 'cam', 'raw', 'calib', args.texture + args.extention)
    new_img_dst = osp.join(new_model_path, 'novel', 'imgs', new_img_name)
    if not os.path.exists(new_img_dst):
        shutil.copyfile(new_img_src, new_img_dst)
    else:
        pass
    print(f"New image {new_img_name} copied to {new_img_dst}")

    if not os.path.exists(osp.join(colmap_dir, "database.db")):
        raise FileNotFoundError("database.db not found")
    if not os.path.exists(image_list_file):
        raise FileNotFoundError("image_list.txt not found")

    feat_extracton_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {colmap_dir}/database.db "
        f"--image_path {os.path.dirname(new_img_dst)} "
        f"--ImageReader.camera_model {args.camera} "
        f"--image_list_path {image_list_file} "
        f"--SiftExtraction.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    feat_matching_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {colmap_dir}/database.db "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    register_cmd = (
        f"{colmap_command} image_registrator "
        f"--database_path {colmap_dir}/database.db "
        f"--Mapper.ba_refine_principal_point 1 "
        f"--input_path {colmap_dir}/sparse/0 "
        f"--output_path {new_model_path}"
    )
    exit_code = os.system(register_cmd)
    if exit_code != 0:
        logging.error(f"Image registration failed with code {exit_code}. Exiting.")
        exit(exit_code)

# threshold surface image and get mask and mask bbox corners
# Kindly borrowed from DeProCams:
# https://github.com/BingyaoHuang/DeProCams/blob/main/src/python/ImgProc.py#L6#L47
def threshDeProCams(im, thresh=None):
    # get rid of negative values
    im[im < 0] = 0

    # threshold im_diff with Otsu's method
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  # !!very important, result of COLOR_RGB2GRAY is different from COLOR_BGR2GRAY
        if im.dtype == 'float32':
            im = np.uint8(im * 255)
            im_in_smooth = cv2.GaussianBlur(im, ksize=(3, 3), sigmaX=1.5)
            if thresh is None:
                # Use Otus's method
                levels = 2
                thresh = threshold_multiotsu(im_in_smooth, levels)
                im_mask = np.digitize(im_in_smooth, bins=thresh) > 0
            else:
                im_mask = im_in_smooth > thresh
    elif im.dtype == np.bool_:  # if already a binary image
        im_mask = im
    elif im.dtype == np.uint8:  # if already a smooothed binary imag
        im_in_smooth = im
        if thresh is None:
                # Use Otus's method
                levels = 3
                thresh = threshold_multiotsu(im_in_smooth, levels)
                im_mask = np.digitize(im_in_smooth, bins=thresh) > 0
        else:
            im_mask = im_in_smooth > thresh
        
    # find the largest contour by area then convert it to convex hull
    # im_contours, contours, hierarchy = cv2.findContours(np.uint8(im_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # only works for OpenCV 3.x
    contours, hierarchy = cv2.findContours(np.uint8(im_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # works for OpenCV 3.x and 4.x
    max_contours = np.concatenate(contours)
    hulls = cv2.convexHull(max_contours)

    im_roi = cv2.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    bbox = cv2.boundingRect(max_contours)
    corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im.shape[0]
    w = im.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1

    return im_mask, im_roi, corners

def make_masks(args):
    """Make masks for all views based on images with white and black projections."""
    setup_dir = osp.join(args.root, 'setups', args.setup)
    views_dir = [osp.join(setup_dir, 'views', f"{view_id:02d}") for view_id in range(1, args.views+1)]
    for view_dir in views_dir:
        cam_dir = osp.join(view_dir, 'cam', 'raw')
        mask_dir = osp.join(cam_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        white = osp.join(cam_dir, 'ref', f'img_0002{args.extention}')
        black = osp.join(cam_dir, 'ref', f'img_0001{args.extention}')
        dst = osp.join(mask_dir, f'mask{args.extention}')

        cam_white = cv2.cvtColor(cv2.imread(white), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cam_black = cv2.cvtColor(cv2.imread(black), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        diff = cam_white - cam_black
        im_mask, _, mask_corners = threshDeProCams(diff)

        im_mask = np.uint8(im_mask > 0) * 255
        # Save the mask to the mask directory
        cv2.imwrite(dst, im_mask)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--root', "-r", required=True, type=str, help='Root folder path')
    args.add_argument('--setup', "-s", required=True, type=str, help='Setup name')
    args.add_argument('--views', type=int, default=25, help='Number of training viewpoints')
    args.add_argument('--view_id', type=int, default=None, help='View ID for novel view registration')
    args.add_argument('--cam_width', type=int, default=800, help='Width of the camera image')
    args.add_argument('--cam_height', type=int, default=800, help='Height of the camera image')
    args.add_argument('--prj_width', type=int, default=800, help='Width of the projector image')
    args.add_argument('--prj_height', type=int, default=800, help='Height of the projector image')
    args.add_argument('--texture', type=str, default='calib', help='Texture name')
    args.add_argument('--extention', type=str, default='.png', help='Image extention')
    args.add_argument('--camera', type=str, default="SIMPLE_PINHOLE", help='COLMAP camera model')
    args.add_argument('--colmap_executable', type=str, default="", help='Path to COLMAP executable')
    args.add_argument('--no_gpu', action='store_true')
    args.add_argument('--skip_matching', action='store_true')
    args.add_argument('--skip_copy', action='store_true')
    args = args.parse_args()
    
    if args.view_id:
        # Novel view registration
        register_novel_view(args)
    else:
        # SfM reconstruction for training
        run_colmap(args)
        # Make mask for each view
        make_masks(args)


