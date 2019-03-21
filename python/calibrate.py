#!/usr/bin/env python
"""This script runs camera-to-arm calibration.

Author(s): Ajay Mandlekar 

Original Matlab code can be found at:

https://github.com/ZacharyTaylor/Camera-to-Arm-Calibration/blob/master/CalCamArm.m
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
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


def find_chessboard_corners(image_paths, height, width, mode='rgb'):
    """Find checkerboard points from a series of images.

    Args:
        image_paths: List of paths to images.
        height: Height of checkerboard pattern.
        width: Width of checkerboard pattern.
        mode: Either `rgb` or `ir`.

    Returns:
        image_points: List of points found in each image, in 2D plane.
        image_size: Size of image.
        valid_inds: List of indices for images used.
    """
    # TODO: check this function... do we need cv2.cornerSubPix?
    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_points = []
    image_size = None
    valid_inds = []
    for i, path in enumerate(image_paths):
        if mode == 'rgb':
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                    gray, (height - 1, width - 1),
                    cv2.CALIB_CB_FAST_CHECK)
        elif mode == 'ir':
            image = cv2.imread(path)
            image = cv2.resize(image, None, 2.0, 2.0, cv2.INTER_CUBIC)
            ir_min = np.min(image)
            ir_max = np.max(image)
            ir_converted = (image - ir_min) / (ir_max - ir_min) * 255
            ir_converted = ir_converted.astype(np.uint8)
            ret, corners = cv2.findChessboardCorners(
                    ir_converted, (height - 1, width - 1),
                    cv2.CALIB_CB_ADAPTIVE_THRESH)
        else:
            raise ValueError('Unrecognized mode: %r' % (mode))

        if image_size is None:
            image_size = (image.shape[0], image.shape[1])
        else:
            assert (image_size[0] == image.shape[0] and
                    image_size[1] == image.shape[1]), (
                    'Image size is not consistent.')

        if ret is True:
            valid_inds.append(i)
            image_points.append(corners.squeeze())

            # Draw and display the corners
            # cv2.drawChessboardCorners(
            #         img, (height - 1, width - 1), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
        else:
            image_points.append(None)

    # cv2.destroyAllWindows()

    return image_points, image_size, valid_inds 


def get_image_points(
        image_dir, 
        square_size,
        board_size,
        mode,
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
        square_size: The width of one of the checkerboard's squares in mm.
        board_size: The height and width tuple of the checkerboard in terms of
            number of squares.
        mode: Either `rgb or `ir`.
        verbose: If True, prints stages of calibration.

    Returns:
        image_points: List of found points on loaded images.
        image_size: Size of the image.
        valid_inds: Indices of loaded images which have checker board found..
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

    # Extract checkerboard patterns.
    if verbose:
        print('Extracting checkerboards...')

    # Convert to meters.
    square_size /= 1000.

    # (should do equivalent of Matlab's detectCheckerboardPoints function)
    board_height, board_width = board_size
    image_points, image_size, valid_inds = find_chessboard_corners(
            image_paths, height=board_height, width=board_width, mode=mode)

    num_valid_images = len(valid_inds)
    if num_valid_images == 0:
        raise ValueError('No checkerboards found in images.')
    elif num_valid_images < 10:
        print('WARNING: only %d images found for calibration...'
              % (num_valid_images))
    else:
        print('Found %d checkerboards out of %d images.'
              % (num_valid_images, num_images))

    return image_points, image_size, valid_inds


def get_object_points(
        square_size,
        board_size):
    """Checkerboard points in 3d world coordinates, assuming plane is at z=0.
    """
    board_height, board_width = board_size
    height_to_use = board_height - 1
    width_to_use = board_width - 1
    object_points = np.zeros((height_to_use * width_to_use, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:height_to_use, 0:width_to_use].T.reshape(
            -1, 2)
    object_points *= square_size  # Scale points by grid spacing.
    return object_points


def calibrate_intrinsics(
        image_dir, 
        square_size,
        board_size,
        mode,
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
        square_size: The width of one of the checkerboard's squares in mm.
        board_size: The height and width tuple of the checkerboard in terms of
            number of squares.
        mode: Either `rgb or `ir`.
        verbose: If True, prints stages of calibration.

    Returns:
        camera_intrinsics: The camera intrinsics.
        camera_distortion: The camera distortion parameters.
    """
    image_points, image_size, _ = get_image_points(
            image_dir, square_size, board_size, mode=mode, verbose=verbose)
    image_points = [value for value in image_points if value is not None]

    object_points = get_object_points(square_size, board_size)
    object_points = [object_points] * len(image_points)

    # Estimate camera intrinsics.
    tic = time.time()
    if verbose:
        print('Etsimating camera intrinsics...')

    # Equivalent of Matlab's estimateCameraParameters:
    # https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L236
    _, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size[::-1], None, None)

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

    return camera_intrinsics, camera_distortion


def calibrate_extrinsics(
        rgb_dir, 
        ir_dir, 
        square_size,
        board_size,
        rgb_intrinsics=None,
        rgb_distortion=None,
        ir_intrinsics=None,
        ir_distortion=None,
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
        square_size: The width of one of the checkerboard's squares in mm.
        board_size: The height and width tuple of the checkerboard in terms of
            number of squares.
        verbose: If True, prints stages of calibration.

    Returns:
        camera_intrinsics: The camera intrinsics.
        camera_distortion: The camera distortion parameters.
    """
    rgb_points, rgb_size, rgb_valid = get_image_points(
            rgb_dir, square_size, board_size, mode='rgb', verbose=verbose)
    ir_points, ir_size, ir_valid = get_image_points(
            ir_dir, square_size, board_size, mode='ir', verbose=verbose)
    object_points = get_object_points(square_size, board_size)

    assert len(rgb_points) == len(ir_points)

    if rgb_intrinsics is None or rgb_distortion is None:
        if verbose:
            print('Etsimating RGB camera intrinsics...')
        tic = time.time()
        _rgb_points = [value for value in rgb_points if value is not None]
        _object_points = [object_points] * len(_rgb_points)
        _, rgb_intrinsics, rgb_distortion, _, _ = cv2.calibrateCamera(
                _object_points, _rgb_points, rgb_size[::-1], None, None)
        toc = time.time()
        print('Finish in %.3f sec' % (toc - tic))

    if ir_intrinsics is None or ir_distortion is None:
        if verbose:
            print('Etsimating ir camera intrinsics...')
        tic = time.time()
        _ir_points = [value for value in ir_points if value is not None]
        _object_points = [object_points] * len(_ir_points)
        _, ir_intrinsics, ir_distortion, _, _ = cv2.calibrateCamera(
                _object_points, _ir_points, ir_size[::-1], None, None)
        toc = time.time()
        print('Finish in %.3f sec' % (toc - tic))

    new_rgb_points = []
    new_ir_points = []
    for i in range(len(rgb_points)):
        if rgb_points[i] is not None and ir_points[i] is not None:
            new_rgb_points.append(rgb_points[i])
            new_ir_points.append(ir_points[i])
    rgb_points = new_rgb_points
    ir_points = new_ir_points

    if verbose:
        print('Found checkerboards in %d pairs of images.' % len(rgb_points))

    object_points = [object_points] * len(rgb_points)

    (_, rgb_intrinsics, rgb_distortion, ir_intrinsics, ir_distortion,
     R, T, E, F) = cv2.stereoCalibrate(
        object_points, rgb_points, ir_points,
        np.array(rgb_intrinsics), np.array(rgb_distortion),
        np.array(ir_intrinsics), np.array(ir_distortion),
        rgb_size)

    print('rgb_intrinsics:')
    print(rgb_intrinsics)
    print('rgb_distortion:')
    print(rgb_distortion)
    print('ir_intrinsics:')
    print(ir_intrinsics)
    print('ir_distortion:')
    print(ir_distortion)

    print('R:')
    print(R)
    print('T:')
    print(T)

    return R, T


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--mode',
            dest='mode',
            type=str,
            help='Either `rgb`, `ir` or `sync`.',
            required=True)

    parser.add_argument(
            '--rgb',
            dest='rgb_dir',
            type=str,
            help='Directory of the RGB images.',
            default=None)

    parser.add_argument(
            '--ir',
            dest='ir_dir',
            type=str,
            help='Directory of the IR images.',
            default=None)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # hardcoded for debugging
    # camera_intrinsics = np.array([
    #     [636.21627486, 0., 310.36756783],
    #     [0., 638.1934484, 236.25505728],
    #     [0., 0., 1.]])
    # camera_distortion = np.array(
    #         [0.13085502, -0.36697504, -0.00276427, -0.00270943, 0.53815114])

    args = parse_args()

    if args.mode == 'rgb':
        calibrate_intrinsics(
            image_dir=args.rgb_dir, 
            square_size=24.7,
            board_size=(7, 9),
            mode=args.mode,
            verbose=True,
        )
    elif args.mode == 'ir':
        calibrate_intrinsics(
            image_dir=args.ir_dir, 
            square_size=24.7,
            board_size=(7, 9),
            mode=args.mode,
            verbose=True,
        )
    elif args.mode == 'sync':
        rgb_intrinsics = [
                [1.05865391e+03, 0., 9.81659694e+02],
                [0., 1.06207063e+03, 5.49504441e+02],
                [0., 0., 1.]]
        rgb_distortion = [
                0.05187516, -0.10796201, 0.00166304, 0.00258503, 0.05098001]
        ir_intrinsics = [
                [731.00351522, 0., 522.95355393],
                [0., 732.67193276, 423.19366342],
                [0., 0., 1.]]
        ir_distortion = [
                -0.04270708, 0.0657837, 0.00645785, 0.00570504, -0.05387679]
        calibrate_extrinsics(
            rgb_dir=args.rgb_dir, 
            ir_dir=args.ir_dir, 
            square_size=24.7,
            board_size=(7, 9),
            rgb_intrinsics=rgb_intrinsics,
            rgb_distortion=rgb_distortion,
            ir_intrinsics=ir_intrinsics,
            ir_distortion=ir_distortion,
            verbose=True,
        )
    else:
        raise ValueError
