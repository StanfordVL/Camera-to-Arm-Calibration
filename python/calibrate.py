"""
This script runs camera-to-arm calibration.

Author(s): Ajay Mandlekar 

Original Matlab code can be found at:

https://github.com/ZacharyTaylor/Camera-to-Arm-Calibration/blob/master/CalCamArm.m
"""

import os
import glob
import re
import cv2
import numpy as np

from os.path import join as pjoin

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    Key to sort on for natural sorting of a list of strings.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    

### TODO: check this function... do we need cv2.cornerSubPix? ###
def extract_board_corners(img_paths, height, width):
    """
    Helper function to extract checkerboard points from a series of images.

    Args:
        img_paths ([str]): list of paths to images
        height (int): height of checkerboard pattern
        width (int): width of checkerboard pattern

    Returns:
        inds_used ([str]): list of indices for images used
        points ([np.array]): list of points found in each image, in 2D plane
        im_shape ([int]): shape of image, needed downstream
        gray_shape ([int]): shape of gray image, needed downstream
    """

    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    inds_used = []
    points = []
    for i, fname in enumerate(img_paths):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (height, width), None)
        # corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)   

        if ret == True:
            # pattern found
            inds_used.append(i)
            points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (height, width), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    return inds_used, points, img.shape, gray.shape

def calibrate_camera_arm(
    image_folder, 
    arm_mat, 
    square_size,
    board_size,
    verbose=True,
    output_transform_matrix=True,
    max_base_offset=1,
    max_end_offset=1,
    inliers=80,
    err_estimate=True,
    num_bootstraps=100,
    camera_parameters=None,
    base_estimate=np.eye(4),
    end_estimate=np.eye(4),
    save_images=True,
    save_path='output'):
    """
    This function calibrates a camera to work with a robotic arm
    by finding the camera intrinsics and the camera-to-arm base
    transformation matrix using a series of arm poses and corresponding
    images of the arm holding a checkerboard. Assumes that the
    camera is rigidly mounted with respect to the arm's base.

    Args:
        image_folder (str): path to a folder containing N images of the
            robotic arm holding a checkerboard taken by the camera you
            wish to calibrate. The entire board must be visible and
            rigidly attached to the arm. The file names do not matter
            as long as when they are arranged in alphabetical order 
            they correspond to the order of the transformations
            in arm_mat.

        arm_mat (np.array): array of shape (4, 4, N) corresponding
            to the set of base-to-arm transformation matrices.

        square_size (float): the width of one of the checkerboard's 
            squares in mm.

        board_size (int, int): the height and width of the checkerboard
            in terms of number of squares

        verbose (bool): if true, prints stages of calibration

        output_transform_matrix (bool): if true, gives
            results as transformation matrices. if false,
            gives results as 1x6 vector with rotation
            in angle-axis format [x, y, z, rx, ry, rz].

        max_base_offset (float): the maximum possible
            translational offset between the camera
            and the arm base in meters. If in doubt,
            overestimate.

        max_end_offset (float): the maximum possible
            translational offset between the arm's 
            end effector and the chessboard in meters.
            If in doubt, overestimate.

        inliers (float): percent of data to take as
            inliers, helps protect against a misaligned
            board.

        err_estimate (bool): if True, bootstraps the data
            to give estimate of calibration accuracy.

        num_bootstraps (int): number of times to bootstrap
            data, only used if err_estimate is True.

        camera_parameters: if given, the calibration will 
            use these camera parameters (intrinsics). Otherwise,
            they will be estimated from the data.

        base_estimate (np.array): array of shape (4, 4) giving 
            the initial estimate for the camera to the base
            transform

        end_estimate (np.array): array of shape (4, 4) giving
            the initial estimate for the end effector to 
            the checkerboard transform

        save_images (bool): if True, save each image with
            guess poses and final calculated poses displayed

        save_path (str): where to save output images

    Returns:
        base_transform (np.array): the camera to base transform

        end_transform (np.array): the end effector to checkerboard transform

        camera_parameters: the camera intrinsics

        base_transform_std: estimate of std of errors in base_transform

        end_transform_std: estimate of std of errors in end_transform

        pixel_error: mean error of projected inliers in pixels
    """

    # sanity check
    img_paths = [im for im in glob.glob(pjoin(image_folder, "*")) if not os.path.isdir(im)]
    num_images = len(img_paths)
    if verbose:
        print("Starting calibration with {} images in directory {}...".format(num_images, image_folder))
    assert arm_mat.shape[0] == 4 and arm_mat.shape[1] == 4 and arm_mat.shape[2] == num_images, "ERROR: arm data and image folder mismatch"

    ### extract checkerboard patterns ###
    if verbose:
        print("Extracting checkerboards...")
    os.mkdir(save_path)
    square_size /= 1000. # convert to meters

    # run a natural sort on the image paths
    img_paths = sorted(img_paths, key=natural_sort_key)

    # (should do equivalent of Matlab's detectCheckerboardPoints function)
    board_height, board_width = board_size
    inds_used, points, im_shape, gray_shape = extract_board_corners(img_paths, height=board_height, width=board_width)
    num_imgs_used = len(inds_used)
    if num_imgs_used == 0:
        raise Exception("ERROR: no checkerboards found in images...")
    elif num_imgs_used < 10:
        print("WARNING: only {} images found for calibration...".format(num_imgs_used))
    print("Found checkerboard in {} out of {} images.".format(num_imgs_used, num_images))

    # filter out the poses corresponding to unused images
    arm_poses = arm_mat[:, :. inds_used]

    ### estimate camera intrinsics ###

    if verbose:
        print("Finding Camera Parameters...")

    # generate an ideal checkerboard to compare points to
    # (should do equivalent of Matlab's generateCheckerboardPoints)
    world_points = [[j, i] for i in range(height - 1) for j in range(width - 1)]

    # checkerboard points in 3d world coordinates, assuming plane is at Z=0
    world_locs = []
    objp = np.zeros((board_height * board_width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_height, 0:board_width].T.reshape(-1, 2)
    objp *= square_size # scale points by grid spacing
    for _ in points:
        world_locs.append(objp) 

    # estimate camera intrinsics if they have not been provided
    if camera_parameters is None:
        # (should do the equivalent of Matlab's estimateCameraParameters)
        # (https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L236)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_locs, points, gray_shape[::-1], None, None)
        
        ### TODO: do we need to do this part here? ###
        # h, w = im_shape[:2]
        # mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        camera_parameters = np.array(mtx)

        # compute reprojection error
        mean_error = 0
        for i in xrange(len(world_locs)):
            imgpoints2, _ = cv2.projectPoints(world_locs[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total reprojection error: {}".format(mean_error / len(world_locs)))


    ### run optimization to get extrinsics + checkerboard to arm transform ###

    if verbose:
        print("Running optimization...")

    ### TODO: convert the code onwards... (implement T2V, and convex opt and visualization) ###
    ### https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L245 ###

