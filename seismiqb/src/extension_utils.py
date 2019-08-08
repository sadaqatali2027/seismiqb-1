""" Utility functions. """
import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm
from skimage.measure import label, regionprops
from numba import njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt

from ..batchflow import B, V, C, L, F, D, P, R
# from seismiqb.batchflow.models.tf import DenseNetFC
# from seismiqb import SeismicCropBatch, SeismicGeometry, SeismicCubeset
from .utils import create_mask, _get_horizons, convert_to_numba_dict

def make_data_extension(batch, **kwargs):
    data_x = []
    for i, cube in enumerate(batch.images):
        cut_mask_ = batch.cut_masks[i]
        data_x.append(np.concatenate([cube, cut_mask_], axis=-1))

    data_y = []

    for cube in batch.masks:
        data_y.append(cube)
    return {"feed_dict": {'images': data_x,
                          'masks': data_y}}

def update_horizon_dict(first, second):
    """ Left merge two dicts. """
    for k, v in second.items():
        if not k in first:
            first.update({k: v})
    return first

def plot_extension_history(next_predict_pipeline, btch):
    """ Function to show single extension step. """
    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(1, 5, 1)
    plt.imshow(btch.images[0][..., 0].T)
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 2)
    plt.imshow(btch.masks[0][:, :, 0, 0].T)
    plt.title('true mask')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 3)
    plt.imshow(btch.cut_masks[0][..., 0].T)
    plt.title('Created mask using predictions')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 4)
    plt.imshow(next_predict_pipeline.get_variable('result_preds')[0][:, :, 0].T)
    plt.title('predictions')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 5)
    plt.imshow(btch.cut_masks[0][..., 0].T + next_predict_pipeline.get_variable('result_preds')[0][:, :, 0].T)
    plt.title('overlap')

    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)
    plt.show()


def make_grid_info(grid_array, cube_name, crop_shape):
    """ Create grid info based on the grid array with lower left coordinates of the crops. """
    grid_array = np.array(grid_array)
    offsets = np.array([min(grid_array[:, 0]),
                        min(grid_array[:, 1]),
                        min(grid_array[:, 2])])
    grid_array = grid_array[:, :].astype(int) - offsets

    # this is not ilines/xlines coords
    ilines_range = [np.min(grid_array[:, 0]), np.max(grid_array[:, 0]) + 1]
    xlines_range = [np.min(grid_array[:, 1]), np.max(grid_array[:, 1]) + crop_shape[1]]
    h_range = [np.min(grid_array[:, 2]), np.max(grid_array[:, 2]) + crop_shape[2]]
    predict_shape = (ilines_range[1] - ilines_range[0],
                     xlines_range[1] - xlines_range[0],
                     h_range[1] - h_range[0])

    grid_info = {'grid_array': grid_array[:, :],
         'predict_shape': predict_shape,
         'crop_shape': crop_shape,
         'cube_name': cube_name,
         'range': [ilines_range, xlines_range, h_range],
         'offsets': offsets}
    return grid_info

def compute_next_points(points, prediction, crop_shape, strides_candidates, WIDTH):
    compared_slices_ = []
    compared_slices_.append(np.sum(prediction[:WIDTH, crop_shape[1] - WIDTH:]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] - WIDTH:, crop_shape[1] - WIDTH:]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] // 2 - WIDTH // 2:crop_shape[2] // 2 + WIDTH // 2,
                                              crop_shape[1] - WIDTH:]))
    compared_slices_.append(np.sum(prediction[:WIDTH ** 2 // crop_shape[1], :]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] - WIDTH ** 2 // crop_shape[1]:, :]))
    stride = strides_candidates[np.argmax(np.array(compared_slices_))]
    points = [sum(x) for x in zip(points, stride)]
    return points, compared_slices_
