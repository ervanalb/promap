import numpy as np
import scipy.interpolate

def compute_inverse_and_disparity(x, y, proj_width, proj_height, quantile=0.7, z=4):
    """This function computes the disparity of a decoded camera image.
    It also can be used to filter outliers from the original image.
    """

    (cam_h, cam_w) = x.shape

    goal_x = np.tile(np.arange(0, cam_w), cam_h)
    goal_y = np.repeat(np.arange(0, cam_h), cam_w)
    goals = np.vstack((goal_x, goal_y, np.ones(cam_w * cam_h))).T

    actuals = np.vstack((x.ravel(), y.ravel())).T

    keep = np.any(actuals != 0, axis=1)
    goals = goals[keep]
    actuals = actuals[keep]

    result = np.linalg.lstsq(goals, actuals, rcond=None)
    xform = result[0]
    # Only keep the best ~70% of the data, like a good scientist
    residuals = np.dot(goals, xform) - actuals
    x_keep = np.power(residuals[:,0], 2).argsort().argsort() < quantile * len(residuals)
    y_keep = np.power(residuals[:,1], 2).argsort().argsort() < quantile * len(residuals)
    keep = np.logical_and(x_keep, y_keep)

    # linreg again
    result = np.linalg.lstsq(goals[keep], actuals[keep], rcond=None)
    xform = result[0]

    # compute disparity
    residuals = np.dot(goals, xform) - actuals # TODO perhaps this should be the other way?
    disparity = np.linalg.norm(residuals, axis=1)
    found = disparity > 0
    stdev = np.std(disparity[found])
    found = disparity < z * stdev # only keep things within ~4 stdevs of the mean

    all_proj_x = np.tile(np.arange(0, proj_width), proj_height)
    all_proj_y = np.repeat(np.arange(0, proj_height), proj_width)
    all_proj_pts = np.vstack((all_proj_x, all_proj_y)).T

    # Create the inverse image
    xs = scipy.interpolate.griddata(actuals[found], goals[found][:,0], all_proj_pts, method="linear", fill_value=0)
    ys = scipy.interpolate.griddata(actuals[found], goals[found][:,1], all_proj_pts, method="linear", fill_value=0)
    x_img = xs.reshape((proj_height, proj_width))
    y_img = ys.reshape((proj_height, proj_width))

    # Create the inverse disparity map
    inv_disparity = scipy.interpolate.griddata(actuals[found], disparity[found], all_proj_pts, method="linear", fill_value=0)
    d_img = inv_disparity.reshape((proj_height, proj_width))

    return ((x_img, y_img), d_img)
