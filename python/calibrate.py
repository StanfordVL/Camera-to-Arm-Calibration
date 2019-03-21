"""
This script runs camera-to-arm calibration.

Author(s): Ajay Mandlekar 

Original Matlab code can be found at:

https://github.com/ZacharyTaylor/Camera-to-Arm-Calibration/blob/master/CalCamArm.m
"""

import os
import sys
import shutil
import json
import time
import glob
import math
import re
import cv2
import scipy.optimize
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os.path import join as pjoin

from pyquaternion import Quaternion

EPS = np.finfo(float).eps * 4.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """
    Key to sort on for natural sorting of a list of strings.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]    

def convert_quat(q, to='xyzw'):
    """
    Converts quaternion from one convention to another. 
    The convention to convert TO is specified as an optional argument. 
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    :param q: a 4-dim numpy array corresponding to a quaternion
    :param to: a string, either 'xyzw' or 'wxyz', determining 
               which convention to convert to.
    """
    if to == 'xyzw':
        return q[[1, 2, 3, 0]]
    elif to == 'wxyz':
        return q[[3, 0, 1, 2]]
    else:
        raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")

def transform_to_vector(transform):
    """
    This function converts a 4x4 transform matrix to a 6D vector
    representation where the first 3 coordinates are translation
    and the next three correspond to the axis-angle representation
    of the rotation with the unit vector corresponding to the axis
    multiplied by the angle.
    """
    vec = np.zeros(6)
    vec[:3] = transform[:3, 3]
    q = mat2quat(transform[:3, :3])
    q = convert_quat(q, to='wxyz')
    q = Quaternion(q)
    vec[3:] = q.angle * q.axis
    return vec

def vector_to_transform(vec):
    """
    This function converts a 6D representation of a transformation
    to a 4x4 transform matrix.
    """
    t = vec[:3]
    angle = np.linalg.norm(vec[3:])
    if angle != 0:
        axis = vec[3:] / angle
    else:
        axis = vec[3:]
    q = quaternion_about_axis(angle, axis)
    return pose2mat((t, q))

def vector_std_to_transform_std(vec, std):
    """
    This function converts a 6D representation of a transformation
    and its standard deviation to a 4x4 transform matrix and
    an entry-wise standard deviation 4x4 matrix.
    """

    # use 1000 samples to estimate std
    transform = vector_to_transform(vec)
    transform_std = np.zeros((4, 4, 1000))
    for i in range(1000):
        transform_std[:, :, i] = vector_to_transform(vec + std * np.random.randn(6))
    transform_std = np.std(transform_std, ddof=1, axis=2)
    return transform, transform_std


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = np.linalg.norm(q)
    if qlen > EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return convert_quat(q, to='xyzw')

def mat2quat(rmat, precise=False):
    """
    Convert given rotation matrix to quaternion
    :param rmat: 3x3 rotation matrix
    :param precise: If isprecise is True,
    the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    :return: vec4 float quaternion angles
    """
    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if precise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q[[1,2,3,0]]

def quat2mat(quaternion):
    """
    Convert given quaternion to matrix
    :param quaternion: vec4 float angles
    :return: 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float32, copy=True)[[3,0,1,2]]
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])

def pose2mat(pose):
    """
    Convert pose to homogeneous matrix
    :param pose: a (pos, orn) tuple where
    pos is vec3 float cartesian, and
    orn is vec4 float quaternion.
    :return:
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat

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
        ret, corners = cv2.findChessboardCorners(gray, (height - 1, width - 1), None)
        # corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)  

        if ret == True:
            # pattern found
            inds_used.append(i)
            points.append(corners.squeeze())

            ### TODO: put a verbose flag here, to put this in... ###

    #         # Draw and display the corners
    #         cv2.drawChessboardCorners(img, (height - 1, width - 1), corners, ret)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(500)
    # cv2.destroyAllWindows()

    return inds_used, points, img.shape, gray.shape

def project_error(
    points, 
    camera_parameters, 
    camera_distortion,
    world_points, 
    arm_poses, 
    inliers, 
    estimated_parameters):
    """
    Projects checkerbooards onto camera frame and finds error between projected and
    actual position.

    Args:
        points (np.array): numpy array of shape (N, 2, M) that gives the positions
            of the N checkerboard points in the M images

        camera_parameters (np.array): camera intrinsics

        camera_distortion (np.array): camera distortion parameters

        world_points (np.array): numpy array of shape (N, 2) giving the world position
            of the checkerboard points before being transformed

        arm_poses (np.array): array of shape (4, 4, M) corresponding
            to the set of base-to-arm transformation matrices per image

        inliers (float): percent of data to take as
            inliers, helps protect against a misaligned
            board.

        estimated_parameters (np.array): the estimated parameter values
            given as a 13-dim vector. The first 6 dimensions are the
            transform from the base to the camera, the next 6 dimensions 
            are the transform from the end effector to the board, and the 
            final parameter is the square size.

    Returns:
        error (float): mean projection error of the inliers in pixels

        projection (np.array): array of shape (N, 2, M) of the positions of the
            N checkerboard points in the M images

        proj_estimate (np.array): array of shape (12, 2, M) of the position of
            [base origin, basex, basey, basez,tcp origin, tcp x, tcp y, tcp z, 
            grid x, grid y, grid z] in each image
    """

    # extract transforms from estimate
    base_transform = vector_to_transform(estimated_parameters[:6])
    grip_transform = vector_to_transform(estimated_parameters[6:12])
    square_size = estimated_parameters[12]

    ### TODO: wtf is going on with the line below? implementation copied from matlab ###

    N = points.shape[0] # number of checkerboard points
    M = points.shape[2] # number of images
    assert(world_points.shape[0] == N)
    assert(arm_poses.shape[2] == M)

    # add square size to chessboard
    # note: world_points is now shape (4, N)
    world_points = square_size * np.array(world_points)
    world_points = np.concatenate([world_points, 
        np.zeros((N, 1)),
        np.ones((N, 1))], axis=1).T

    ### TODO: should this be a transpose? ###
    K_cam = camera_parameters.T 
    assert(K_cam.shape[0] == 3 and K_cam.shape[1] == 3)

    projection = np.zeros_like(points)
    proj_estimate = np.zeros((12, 2, M))

    # organize distortion parameters

    # should be [k1, k2, p1, p2[, k3[, k4, k5, k6]]] 
    # the latter ones might not be returned, in which case they should be 0.
    k = [camera_distortion[0], camera_distortion[1]]
    p = [camera_distortion[2], camera_distortion[3]]
    if len(camera_distortion > 4):
        k.append(camera_distortion[4])
    else:
        k.append(0.)
    k = np.array(k)
    p = np.array(p)

    ### TODO: I don't know what this code is doing down here, but I'm trying to keep it as close to Matlab implementation ###
    axis_length = 0.1
    axis = np.array([
        [0., 0., 0., 1.],
        [axis_length, 0., 0., 1.],
        [0., axis_length, 0., 1.],
        [0., 0., axis_length, 1.]
        ])

    error = np.empty((N, M))
    error[:] = np.nan

    mat_4d_to_3d = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])

    # loop over arm poses
    for i in range(M):
        # transform chessboard points from the world into the image
        res1 = mat_4d_to_3d.dot(base_transform.dot(arm_poses[:, :, i].dot(grip_transform.dot(world_points)))).T # shape (N, 3)
        res2 = mat_4d_to_3d.dot(base_transform.dot(axis)).T # shape (4, 3)
        res3 = mat_4d_to_3d.dot(base_transform.dot(arm_poses[:, :, i].dot(axis))).T # shape (4, 3)
        res4 = mat_4d_to_3d.dot(base_transform.dot(arm_poses[:, :, i].dot(grip_transform.dot(axis)))).T # shape (4, 3)
        projected = np.concatenate([res1, res2, res3, res4], axis=0) # shape (N + 12, 3)
        x = projected[:, 0] / projected[:, 2]
        y = projected[:, 1] / projected[:, 2]
        r2 = x * x + y * y

        # find tangential distortion
        xTD = 2 * p[0] * x * y + p[1] * (r2 + 2 * x * x)
        yTD = p[0] * (r2 + 2 * y * y) + 2 * p[1] * x * y

        # find radial distortion
        rad_dist = (1. + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2)
        xRD = x * rad_dist
        yRD = y * rad_dist

        # recombine and include camera intrinsics
        x_comb = np.expand_dims(xTD + xRD, 1)
        y_comb = np.expand_dims(yTD + yRD, 1)
        projected = np.concatenate([x_comb, y_comb, np.ones((x.shape[0], 1))], axis=1) # shape (N + 12, 3)
        projected = K_cam.dot(projected.T) # shape (3, N + 12)
        projected = projected[:2, :].T # shape (N + 12, 2)

        projection[:, :, i] = projected[:-12, :] # shape (N, 2)
        proj_estimate[:, :, i] = projected[-12:, :] # shape (12, 2)

        # find error in projection
        error[:, i] = np.sum(np.square(projection[:, :, i] - points[:, :, i]), axis=1) # shape (N,)

    ### TODO: why is this NaN check necessary? ###

    # remove invalid points
    error = error[~np.isnan(error)]

    # drop the largest values as outliers and take mean
    error.sort(axis=-1)
    cutoff = math.floor((inliers / 100.) * error.shape[0])
    error = np.mean(error[:cutoff])

    # return RMS error in pixels
    error = np.sqrt(error)
    return error, projection, proj_estimate


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
    camera_distortion=None,
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

        camera_parameters (np.array): if given, the calibration will 
            use these camera parameters (intrinsics). Otherwise,
            they will be estimated from the data.

        camera_distortion (np.array): if given, the calibration will
            use these distortion parameters (radial, tangential). 
            Otherwise, they will be estimated from the data.

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
    start_time = time.time()

    # sanity check
    img_paths = [im for im in glob.glob(pjoin(image_folder, "*")) if not os.path.isdir(im)]
    num_images = len(img_paths)
    if verbose:
        print("Starting calibration with {} images in directory {}...".format(num_images, image_folder))
    assert arm_mat.shape[0] == 4 and arm_mat.shape[1] == 4 and arm_mat.shape[2] == num_images, "ERROR: arm data and image folder mismatch"

    ### extract checkerboard patterns ###
    if verbose:
        print("Extracting checkerboards...")
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
    arm_poses = arm_mat[:, :, inds_used]

    ### estimate camera intrinsics ###

    # checkerboard points in 3d world coordinates, assuming plane is at Z=0
    world_locs = []
    height_to_use = board_height - 1
    width_to_use = board_width - 1
    objp = np.zeros((height_to_use * width_to_use, 3), np.float32)
    objp[:,:2] = np.mgrid[0:height_to_use, 0:width_to_use].T.reshape(-1, 2)
    objp *= square_size # scale points by grid spacing
    for _ in points:
        world_locs.append(objp) 

    # estimate camera intrinsics if they have not been provided
    if camera_parameters is None or camera_distortion is None:
        print("Camera Parameters not provided. Finding Camera Parameters...")
        # (should do the equivalent of Matlab's estimateCameraParameters)
        # (https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L236)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_locs, points, gray_shape[::-1], None, None)
        
        ### TODO: do we need to do this part here? ###
        # h, w = im_shape[:2]
        # mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        camera_parameters = np.array(mtx)
        print("camera_parameters: (shape {})".format(camera_parameters.shape))
        print(camera_parameters)

        # should be [k1, k2, p1, p2[, k3[, k4, k5, k6]]] 
        # the latter ones might not be returned, in which case they are 0.
        camera_distortion = np.array(dist)[0]
        print("camera_distortion: (shape {})".format(camera_distortion.shape))
        print(camera_distortion)

        # compute reprojection error
        mean_error = 0
        for i in range(len(world_locs)):
            imgpoints2, _ = cv2.projectPoints(world_locs[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(points[i], imgpoints2.squeeze(), cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total reprojection error: {}".format(mean_error / len(world_locs)))


    ### run optimization to get extrinsics + checkerboard to arm transform ###
    if verbose:
        print("Running optimization...")

    ### TODO: convert the code onwards... (implement convex opt and visualization) ###

    base_estimate = transform_to_vector(base_estimate)
    end_estimate = transform_to_vector(end_estimate)

    # set up search ranges
    base_range = np.array([max_base_offset, max_base_offset, max_base_offset, np.pi, np.pi, np.pi])
    grip_range = np.array([max_end_offset, max_end_offset, max_end_offset, np.pi, np.pi, np.pi])
    square_range = 0.001
    ranges = np.concatenate([base_range, grip_range, [square_range]])

    initial = np.concatenate([base_estimate, end_estimate, [square_size]])
    ub = initial + ranges
    lb = initial - ranges

    # generate an ideal checkerboard to compare points to
    # (should do equivalent of Matlab's generateCheckerboardPoints)
    world_points = [[j, i] for i in range(height_to_use) for j in range(width_to_use)]

    print("init points shape: {} {}".format(len(points), points[0].shape))
    points=np.array(points).transpose((1, 2, 0))
    print("new points shape: {} {}".format(len(points), points[0].shape))

    # function to optimize: returns RMS pixel error of reprojection
    opt_func = lambda x : project_error(
        points=points, 
        camera_parameters=camera_parameters, 
        camera_distortion=camera_distortion,
        world_points=np.array(world_points),
        arm_poses=arm_poses,
        inliers=inliers,
        estimated_parameters=x)[0]

    ### TODO: experiment with different methods here (cvxpy, different method argument w/o bound, etc) ###
    
    bounds = (
        (lb[0], ub[0]),
        (lb[1], ub[1]),
        (lb[2], ub[2]),
        (lb[3], ub[3]),
        (lb[4], ub[4]),
        (lb[5], ub[5]),
        (lb[6], ub[6]),
        (lb[7], ub[7]),
        (lb[8], ub[8]),
        (lb[9], ub[9]),
        (lb[10], ub[10]),
        (lb[11], ub[11]),
        (lb[12], ub[12]),
    )
    res = scipy.optimize.minimize(opt_func,
        x0=initial,
        method='L-BFGS-B',
        bounds=bounds
    )

    assert(res.success)
    solution = res.x
    pixel_error = res.fun

    if pixel_error > 10:
        print("WARNING: Average projection error found to be {} pixels.".format(pixel_error))
        print("This large error is a strong indicator that something has gone wrong")
        print("Check input parameters, number of images correctly processed, and try tuning the input parameters.")


    ### TODO: code for saving images, bootstrapping, and saving parameters... ###
    ### https://github.com/StanfordVL/Camera-to-Arm-Calibration/blob/master/CalCamArm_app.m#L280 ###

    if save_images:
        if verbose:
            print("Saving images with initial guess and result...")

        _, projection_guess, projected_initial = project_error(
            points=points, 
            camera_parameters=camera_parameters, 
            camera_distortion=camera_distortion,
            world_points=np.array(world_points),
            arm_poses=arm_poses,
            inliers=inliers,
            estimated_parameters=initial)

        _, projection_solution, projected_solution = project_error(
            points=points, 
            camera_parameters=camera_parameters, 
            camera_distortion=camera_distortion,
            world_points=np.array(world_points),
            arm_poses=arm_poses,
            inliers=inliers,
            estimated_parameters=solution)

        # dump images with visualization
        if os.path.isdir(save_path):
            if (sys.version_info > (3, 0)):
                out = input("WARNING: will delete old contents of save location {}. Press enter to proceed".format(save_path))
            else:
                out = raw_input("WARNING: will delete old contents of save location {}. Press enter to proceed".format(save_path))
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            os.mkdir(save_path)

        num_images = projection_guess.shape[2]
        for i in range(num_images):

            # create figure
            plt.figure()

            # draw original image
            im_path = img_paths[inds_used[i]]
            img = cv2.imread(im_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

            plt.scatter(points[:, 0, i], points[:, 1, i], c='r', marker='o')
            plt.scatter(projection_guess[:, 0, i], projection_guess[:, 1, i], c='b', marker='+')
            plt.scatter(projection_solution[:, 0, i], projection_solution[:, 1, i], c='g', marker='d')

            plt.scatter(projected_initial[0, 0, i], projected_initial[0, 1, i], c='c', marker='^')
            plt.scatter(projected_initial[4, 0, i], projected_initial[4, 1, i], c='c', marker='v')
            plt.scatter(projected_initial[8, 0, i], projected_initial[8, 1, i], c='c', marker='<')

            plt.scatter(projected_solution[0, 0, i], projected_solution[0, 1, i], c='m', marker='^')
            plt.scatter(projected_solution[4, 0, i], projected_solution[4, 1, i], c='m', marker='v')
            plt.scatter(projected_solution[8, 0, i], projected_solution[8, 1, i], c='m', marker='<')

            color = 'rgb'
            for j in range(2, 5):
                for k in range(3):
                    plt.plot(
                        [projected_initial[4 * k, 0, i], projected_initial[4 * k + j - 1, 0, i]],
                        [projected_initial[4 * k, 1, i], projected_initial[4 * k + j - 1, 1, i]],
                        color[j - 2]
                    )
            for j in range(2, 5):
                for k in range(3):
                    plt.plot(
                        [projected_solution[4 * k, 0, i], projected_solution[4 * k + j - 1, 0, i]],
                        [projected_solution[4 * k, 1, i], projected_solution[4 * k + j - 1, 1, i]],
                        color[j - 2]
                    )
            plt.legend(['detected', 'guess', 'solution', 'base initial guess', 'tcp initial guess',
                'grid initial guess', 'base solution', 'tcp solution', 'grid solution'])
            
            ### TODO: convert this snippet of matlab code ###
            #  annotation('textbox', [0,0,.10,.10], 'String',...
            # 'Red is x, Green is y, Blue is z', 'FitBoxToText','on',...
            # 'BackgroundColor','white','Color','red')
            # plt.annotate()

            plt.savefig(pjoin(save_path, "outputImage{}.png".format(i)))
            plt.close()

        if verbose:
            print("Done saving images.")

    # bootstrap estimates
    assert(num_images == arm_poses.shape[2])
    if err_estimate:
        boot_solutions = np.zeros((num_bootstraps, initial.shape[0]))
        if verbose:
            print("Running Bootstrap Opimization...")
        for i in range(num_bootstraps):
            # sample points
            sampled_inds = np.random.choice(range(num_images), num_images)
            boot_arm_poses = arm_poses[:, :, sampled_inds]
            boot_points = points[:, :, sampled_inds]

            # function to optimize: returns RMS pixel error of reprojection
            boot_func = lambda x : project_error(
                points=boot_points, 
                camera_parameters=camera_parameters, 
                camera_distortion=camera_distortion,
                world_points=np.array(world_points),
                arm_poses=boot_arm_poses,
                inliers=inliers,
                estimated_parameters=x)[0]

            # optimize
            res = scipy.optimize.minimize(boot_func,
                x0=initial,
                method='L-BFGS-B',
                bounds=bounds
            )
            assert(res.success)
            boot_solutions[i, :] = res.x

            if verbose:
                print("Bootsrap Opt {} out of {}".format(i + 1, num_bootstraps))
    else:
        boot_solutions = np.zeros((1, 13))

    # convert solutions to transformation matrices
    base_transform = solution[:6]
    base_transform_std = np.std(boot_solutions[:, :6], ddof=1, axis=0)
    end_transform = solution[6:12]
    end_transform_std = np.std(boot_solutions[:, 6:12], ddof=1, axis=0)
    estimated_square_size = solution[12]
    estimated_square_size_std = np.std(boot_solutions[:, 12], ddof=1)

    if output_transform_matrix:
        base_transform, base_transform_std = vector_std_to_transform_std(base_transform, base_transform_std)
        end_transform, end_transform_std = vector_std_to_transform_std(end_transform, end_transform_std)
    
    if verbose:
        print("Calibration completed in {} seconds with a mean error of {} pixels.".format(time.time() - start_time, pixel_error))

    # save solutions
    if not os.path.isdir(save_path):
        os.makedir(save_path)

    solutions_to_save = dict(
        base_transform=base_transform, 
        base_transform_std=base_transform_std,
        end_transform=end_transform,
        end_transform_std=end_transform_std,
        camera_parameters=camera_parameters,
        camera_distortion=camera_distortion,
        pixel_error=np.array([pixel_error]),
        estimated_square_size=np.array([estimated_square_size]),
        estimated_square_size_std=np.array([estimated_square_size_std]))
    np.savez(pjoin(save_path, "calib.npz"), **solutions_to_save)

    if verbose:
        print("Computed solutions")
        print(json.dumps(solutions_to_save, indent=4))

if __name__ == "__main__":

    image_folder = "./calib_data/images"
    arm_mat = np.load("./calib_data/poseMat.npy")

    # hardcoded for debugging
    # camera_parameters = None
    # camera_distortion = None
    camera_parameters = np.array([
        [636.21627486, 0., 310.36756783],
        [0., 638.1934484, 236.25505728],
        [0., 0., 1.]])
    camera_distortion = np.array([0.13085502, -0.36697504, -0.00276427, -0.00270943, 0.53815114])

    calibrate_camera_arm(
        image_folder=image_folder, 
        arm_mat=arm_mat, 
        square_size=24.7, # in mm
        board_size=(7, 10),
        verbose=True,
        output_transform_matrix=True,
        max_base_offset=1,
        max_end_offset=1,
        inliers=80,
        err_estimate=True,
        num_bootstraps=100,
        camera_parameters=camera_parameters,
        camera_distortion=camera_distortion,
        base_estimate=np.eye(4),
        end_estimate=np.eye(4),
        save_images=True,
        save_path='output',
    )


