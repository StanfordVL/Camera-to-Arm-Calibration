#!/usr/bin/env python
"""This script runs camera-to-arm calibration.

Author(s): Ajay Mandlekar 

Original Matlab code can be found at:

https://github.com/ZacharyTaylor/Camera-to-Arm-Calibration/blob/master/CalCamArm.m
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import glob
import re
import time

import cv2
import numpy as np


EPS = np.finfo(float).eps * 4.


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Key to sort on for natural sorting of a list of strings.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    


def find_chessboard_corners(image_paths, height, width):
    """Find checkerboard points from a series of images.

    Args:
        image_paths: List of paths to images.
        height: Height of checkerboard pattern.
        width: Width of checkerboard pattern.

    Returns:
        valid_inds: List of indices for images used.
        image_points: List of points found in each image, in 2D plane.
        image_shape: Shape of image, needed downstream.
        gray_shape: Shape of gray image, needed downstream.
    """
    # TODO: check this function... do we need cv2.cornerSubPix?
    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    valid_inds = []
    image_points = []
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
                gray, (height - 1, width - 1), None)

        if ret is True:
            valid_inds.append(i)
            image_points.append(corners.squeeze())

            # Draw and display the corners
            # cv2.drawChessboardCorners(
            #         img, (height - 1, width - 1), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    image_shape = image.shape
    gray_shape = gray.shape

    return valid_inds, image_points, image_shape, gray_shape


def calibrate(
        image_dir, 
        arm_mat, 
        square_size,
        board_size,
        camera_parameters=None,
        camera_distortion=None,
        base_estimate=np.eye(4),
        end_estimate=np.eye(4),
        verbose=True):
    """
    This function calibrates a camera to work with a robotic arm
    by finding the camera intrinsics and the camera-to-arm base
    transformation matrix using a series of arm poses and corresponding
    images of the arm holding a checkerboard. Assumes that the
    camera is rigidly mounted with respect to the arm's base.

    Args:
        image_dir: path to a folder containing N images of the
            robotic arm holding a checkerboard taken by the camera you
            wish to calibrate. The entire board must be visible and
            rigidly attached to the arm. The file names do not matter
            as long as when they are arranged in alphabetical order 
            they correspond to the order of the transformations
            in arm_mat.
        arm_mat: Array of shape [4, 4, N] corresponding to the set of
            base-to-arm transformation matrices.
        square_size: The width of one of the checkerboard's squares in mm.
        board_size: The height and width tuple of the checkerboard in terms of
            number of squares.
        camera_parameters: if given, the calibration will 
            use these camera parameters (intrinsics). Otherwise,
            they will be estimated from the data.
        camera_distortion: if given, the calibration will
            use these distortion parameters (radial, tangential). 
            Otherwise, they will be estimated from the data.
        verbose: If True, prints stages of calibration.

    Returns:
        base_transform (np.array): the camera to base transform
        end_transform (np.array): the end effector to checkerboard transform
        camera_parameters: the camera intrinsics
        base_transform_std: estimate of std of errors in base_transform
        end_transform_std: estimate of std of errors in end_transform
        pixel_error: mean error of projected inliers in pixels
    """

    # sanity check
    image_pattern = os.path.join(image_dir, '*.png')
    image_paths = glob.glob(image_pattern)
    num_images = len(image_paths)

    # TODO(kuanfang): Double check this.
    image_paths = sorted(image_paths, key=natural_sort_key)

    if verbose:
        print('Starting calibration with %d images in directory %s...'
              % (num_images, image_dir))

    assert (arm_mat.shape[0] == 4 and
            arm_mat.shape[1] == 4 and
            arm_mat.shape[2] == num_images), (
                    'ERROR: arm data and image folder mismatch')

    # Extract checkerboard patterns.
    if verbose:
        print('Extracting checkerboards...')

    # Convert to meters.
    square_size /= 1000.

    # (should do equivalent of Matlab's detectCheckerboardPoints function)
    board_height, board_width = board_size
    valid_inds, image_points, image_shape, gray_shape = find_chessboard_corners(
            image_paths, height=board_height, width=board_width)
    num_valid_images = len(valid_inds)

    if num_valid_images == 0:
        raise ValueError('No checkerboards found in images.')
    elif num_valid_images < 10:
        print('WARNING: only %d images found for calibration...'
              % (num_valid_images))
    else:
        print('Found %d checkerboards out of %d images.'
              % (num_valid_images, num_images))

    # Filter out the poses corresponding to unused images.
    arm_poses = arm_mat[:, :, valid_inds]  # NOQA

    # Checkerboard points in 3d world coordinates, assuming plane is at z=0.
    height_to_use = board_height - 1
    width_to_use = board_width - 1
    object_points = np.zeros((height_to_use * width_to_use, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:height_to_use, 0:width_to_use].T.reshape(
            -1, 2)
    object_points *= square_size  # Scale points by grid spacing.
    object_points = [object_points] * len(image_points)

    # estimate camera intrinsics if they have not been provided
    if camera_parameters is None or camera_distortion is None:
        # Estimate camera intrinsics.
        tic = time.time()
        if verbose:
            print('Etsimating camera intrinsics...')

        # Equivalent of Matlab's estimateCameraParameters:
        # https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L236
        _, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray_shape[::-1], None, None)

        toc = time.time()
        print('Finish in %.3f sec' % (toc - tic))
        
        # TODO(ajay): Do we need to do this part here?
        # h, w = im_shape[:2]
        # mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        camera_intrinsics = np.array(intrinsics)
        print('camera_intrinsics:')
        print(camera_intrinsics)

        # should be [k1, k2, p1, p2[, k3[, k4, k5, k6]]] 
        # the latter ones might not be returned, in which case they are 0.
        camera_distortion = np.array(distortion)[0]
        print('camera_distortion')
        print(camera_distortion)

        # compute reprojection error
        mean_error = 0
        for i in range(len(object_points)):
            image_points_2, _ = cv2.projectPoints(
                    object_points[i], rvecs[i], tvecs[i],
                    intrinsics, distortion)
            error = cv2.norm(
                    image_points[i], image_points_2.squeeze(), cv2.NORM_L2)
            error /= len(image_points_2)
            mean_error += error

        mean_error /= float(len(object_points))

        print('Total reprojection error: %.2f' % (mean_error))


if __name__ == '__main__':

    image_dir = './calib_data/images'
    arm_mat = np.load('./calib_data/poseMat.npy')

    # hardcoded for debugging
    camera_parameters = None
    camera_distortion = None
    # camera_parameters = np.array([
    #     [636.21627486, 0., 310.36756783],
    #     [0., 638.1934484, 236.25505728],
    #     [0., 0., 1.]])
    # camera_distortion = np.array(
    #         [0.13085502, -0.36697504, -0.00276427, -0.00270943, 0.53815114])

    calibrate(
        image_dir=image_dir, 
        arm_mat=arm_mat, 
        square_size=24.7,
        board_size=(7, 10),
        camera_parameters=camera_parameters,
        camera_distortion=camera_distortion,
        verbose=True,
    )
