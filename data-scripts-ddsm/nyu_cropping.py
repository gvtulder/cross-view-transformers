# Cross-view transformers for multi-view analysis of unregistered medical images
# Copyright (C) 2021 Gijs van Tulder / Radboud University, the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import skimage.measure
import skimage.morphology
import skimage.transform
import scipy.ndimage.morphology

import calc_optimal_centers

#
# Crops mammography images using the method described in
# https://cs.nyu.edu/~kgeras/reports/datav1.0.pdf
#
# See also https://github.com/nyukat/breast_cancer_classifier/
#

N_ITER = 100
N_THRESHOLD = -0.999
N_BUFFER = 50


def find_crop_center(x, crop_size=[1024, 1024], side='left', view='cc'):
    if side == 'right':
        # flip left-right
        x = x[:, ::-1]

    # find nonzero pixels
    x_nonzero = x > N_THRESHOLD

    # erode N_ITER times
    x_obj_mask = scipy.ndimage.morphology.binary_erosion(x_nonzero, iterations=N_ITER)

    # find the largest connected component
    x_obj_mask, n_objects = skimage.morphology.label(x_obj_mask, return_num=True)
    largest_object_label = np.argmax(np.sum(x_obj_mask == label + 1) for label in range(n_objects))

    # mask the largest connected component
    x_obj_mask = (x_obj_mask == largest_object_label + 1)

    # dilate N_ITER times
    x_obj_mask = scipy.ndimage.morphology.binary_dilation(x_obj_mask, iterations=N_ITER)

    # find bounding box
    rprops = skimage.measure.regionprops(x_obj_mask.astype(int))[0]

    # add buffer
    ymin = max(0, min(rprops.bbox[0] - N_BUFFER, x_obj_mask.shape[0]))
    xmin = max(0, min(rprops.bbox[1] - N_BUFFER, x_obj_mask.shape[1]))
    ymax = max(0, min(rprops.bbox[2] + N_BUFFER, x_obj_mask.shape[0]))
    xmax = max(0, min(rprops.bbox[3] + N_BUFFER, x_obj_mask.shape[1]))

    # add constraints based on view
    if view == 'cc':
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(rightmost_x=xmax)
    else:
        assert view == 'mlo'
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(rightmost_x=xmax,
                                                                                     bottommost_y=ymax)

    # compute optimal center
    wininfo = calc_optimal_centers.get_image_optimal_window_info(
        image=x_obj_mask,
        com=np.array(x_obj_mask.shape) // 2,
        window_dim=np.array(crop_size),
        tl_br_constraint=tl_br_constraint)

    best_center_y = wininfo['best_center_y']
    best_center_x = wininfo['best_center_x']

    if side == 'right':
        # flip left-right
        best_center_x = x.shape[1] - best_center_x

    return (best_center_y, best_center_x), rprops.bbox, wininfo
