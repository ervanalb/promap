import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal

x = np.load("x.npy")
y = np.load("y.npy")
w = np.load("w.npy")
b = np.load("b.npy")

width = 1920
height = 1080

(im_height, im_width) = x.shape

goal_x = np.tile(np.arange(0, im_width), im_height)
goal_y = np.repeat(np.arange(0, im_height), im_width)
goals = np.vstack((goal_x, goal_y, np.ones(im_width * im_height))).T

actuals = np.vstack((x.ravel(), y.ravel())).T

keep = np.any(actuals != 0, axis=1)
goals = goals[keep]
actuals = actuals[keep]
print(goals)
print(actuals)
print(len(actuals))

result = np.linalg.lstsq(goals, actuals, rcond=None)
xform = result[0]
# Only keep the best 70% of the data, like a good scientist
residuals = np.dot(goals, xform) - actuals
x_keep = np.power(residuals[:,0], 2).argsort().argsort() < 0.7 * len(residuals)
y_keep = np.power(residuals[:,1], 2).argsort().argsort() < 0.7 * len(residuals)
keep = np.logical_and(x_keep, y_keep)

# linreg again
result = np.linalg.lstsq(goals[keep], actuals[keep], rcond=None)
xform = result[0]

# compute disparity
residuals = np.dot(goals, xform) - actuals # TODO perhaps this should be the other way?
print(residuals)
disparity = np.linalg.norm(residuals, axis=1)
print(disparity)
found = disparity > 0
stdev = np.std(disparity[found])
found = disparity < 4 * stdev # only keep things within 4 stdevs of the mean

all_x = np.tile(np.arange(0, width), height)
all_y = np.repeat(np.arange(0, height), width)
all_pts = np.vstack((all_x, all_y)).T

# Compute disparity from the camera's perspective
all_x = np.tile(np.arange(0, im_width), im_height)
all_y = np.repeat(np.arange(0, im_height), im_width)
all_img_pts = np.vstack((all_x, all_y)).T
disparity_im = scipy.interpolate.griddata(goals[found][:,0:2], disparity[found], all_img_pts, method="linear", fill_value=0)
disparity_im = disparity_im.reshape(im_height, im_width)
plt.imshow(disparity_im)
plt.show()

def interp(grid):
    return scipy.interpolate.RegularGridInterpolator([np.arange(d) for d in grid.shape], grid)

# Use this line for disparity computation
#values = scipy.interpolate.griddata(actuals[found], disparity[found], all_pts, method="linear", fill_value=0)

# Use this chunk for color computation
xs = scipy.interpolate.griddata(actuals[found], goals[found][:,0], all_pts, method="linear", fill_value=0)
ys = scipy.interpolate.griddata(actuals[found], goals[found][:,1], all_pts, method="linear", fill_value=0)
img_pts = np.vstack((xs, ys)).T
#disparity = disparity.reshape((im_height, im_width))
#values = interp(disparity_im)(img_pts[:,1::-1])
values = interp(w)(img_pts[:,1::-1])

#THRESHOLD = 1
#
#from scipy.interpolate.interpnd import _ndim_coords_from_arrays
#from scipy.spatial import cKDTree
#
## Construct kd-tree, functionality copied from scipy.interpolate
#tree = cKDTree(actuals[found])
#xi = _ndim_coords_from_arrays(all_pts, ndim=2)
#dists, indexes = tree.query(xi, k=2)
#dists = dists[:,1]
#
## Copy original result but mask missing values with NaNs
values_masked = values[:]
#values_masked[dists > THRESHOLD] = np.nan

proj_disparity = values_masked.reshape((height, width))

#proj_disparity_filt = scipy.signal.medfilt(proj_disparity, 5)
proj_disparity_filt = proj_disparity

#proj_disparity = np.zeros((height, width))
#proj_disparity[all_pts[:, 1], all_pts[:, 0]] = values

#proj_disparity = np.zeros((height, width))
#proj_disparity[actuals[:,1], actuals[:,0]] = disparity
#proj_disparity_filt = np.zeros((height, width))

#kernel_size = 5
#print("Filtering...")
#for y in range(height):
#    print(int(y / height * 100), "%")
#    for x in range(width):
#        proj_disparity_filt[y, x] = proj_disparity[y, x]
#        if proj_disparity_filt[y, x] > 0:
#            continue
#        for k in range(1, kernel_size):
#            kernel = proj_disparity[max(0, y-k):min(height, y+k+1), max(0, x-k):min(width, x+k+1)]
#            kernel = kernel.ravel()
#            kernel = kernel[kernel > 0]
#            if len(kernel) == 0:
#                continue
#            result = np.mean(kernel)
#            proj_disparity_filt[y, x] = result
#            if len(kernel) > 2:
#                break

#proj_disparity_filt %= 10
plt.imshow(proj_disparity_filt, cmap=plt.get_cmap('gray'))
plt.show()
import sys
sys.exit(0)

x_corrected = np.zeros((height, width))
y_corrected = np.zeros((height, width))

proj_goal_x = np.tile(np.arange(0, width), height)
proj_goal_y = np.repeat(np.arange(0, height), width)
proj_goals = np.vstack((proj_goal_x, proj_goal_y, np.ones(width * height))).T

# Compute inverse of the transformation matrix
print(xform)
xform = np.hstack((xform, np.array([0, 0, 1])[:,None]))
print(xform)
xform_inv = np.linalg.inv(xform)
xform_inv = xform_inv[:,0:2]

im_points = np.dot(proj_goals, xform_inv)

colors_x = interp(x)(im_points[1::-1]).reshape(height, width)
colors_y = interp(y)(im_points[1::-1]).reshape(height, width)
#colors_y = y[int_points[:,1], int_points[:,0]].reshape(height, width)

plt.imshow(colors_x)
plt.imshow(colors_y)
plt.show()

print(im_points)
