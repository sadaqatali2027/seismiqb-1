import os
import sys
import logging

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('..')
from seismiqb.batchflow import Dataset, Pipeline, FilesIndex, Batch
from seismiqb.batchflow import B, V, C, L, F, D, P, R
from seismiqb.batchflow.models.tf import TFModel, DenseNetFC
from seismiqb.batchflow.models.tf.layers import conv_block
from seismiqb import SeismicCropBatch, SeismicGeometry, SeismicCubeset
from seismiqb.src.utils import create_mask, _get_horizons, convert_to_numba_dict


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
    plt.imshow(btch.data_crops[0][..., 0].T)
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 2)
    plt.imshow(btch.mask_crops[0][:, :, 0, 0].T)
    plt.title('true mask')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 3)
    plt.imshow(btch.cut_mask_crops[0][..., 0].T)
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
    plt.imshow(btch.cut_mask_crops[0][..., 0].T + next_predict_pipeline.get_variable('result_preds')[0][:, :, 0].T)
    plt.title('overlap')

    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 64, 10));
    ax.set_yticks(np.arange(0, 64, 10));
    ax.grid(color='w', linestyle='-', linewidth=.5)
    plt.show()

    def load_prior_mask(ds, points, crop_shape=[2, 64, 64], cube_index=0, show_prior_mask=False):
        """Save prior mask to a cubeset attribute `prior_mask`.
        Parameters
        ----------
        ds : Cubeset
        points : points to crop from
        crop_shape
        cube_index : int
        show_prior_mask : bool
        """
        ds_points = np.array([[ds.indices[cube_index], *points, None]])[:, :4]

        start_predict_pipeline = (Pipeline()
                            .load_component(src=[D('geometries'), D('labels')],
                                            dst=['geometries', 'labels'])
                            .crop(points=ds_points, shape=crop_shape)
                            .load_cubes(dst='data_crops')
                            .create_masks(dst='mask_crops', width=1, single_horizon=True, src_labels='labels')
                            .rotate_axes(src=['data_crops', 'mask_crops'])
                            .add_axis(src='mask_crops', dst='mask_crops')
                            .scale(mode='normalize', src='data_crops')) << ds

        batch = start_predict_pipeline.next_batch(3, n_epochs=None)

        if show_prior_mask:
            plt.imshow(batch.mask_crops[0][:, :, 0, 0].T)
            plt.show()

        i_shift, x_shift, h_shift = [slices[0] for slices in batch.slices[0]]
        transforms = [lambda i_: ds.geometries[ds.indices[cube_index]].ilines[i_ + i_shift],
                      lambda x_: ds.geometries[ds.indices[cube_index]].xlines[x_ + x_shift],
                      lambda h_: h_ + h_shift]
        ds.get_point_cloud(np.moveaxis(batch.mask_crops[0][:, :, :1, 0], -1, 0),
                           threshold=0.5, dst='prior_mask', coordinates=None, transforms=transforms, separate=True)
        if len(ds.prior_mask[0]) == 0:
            raise ValueError("Prior mask is empty")
        numba_horizon = convert_to_numba_dict(ds.prior_mask[0])
        ds.prior_mask = {ds.indices[cube_index]: numba_horizon}
        return ds

def make_grid_info(grid_array, cube_index, crop_shape):
    """ Create grid info based on the grid array with lower left coordinates of the crops. """
    grid_array = np.array(grid_array)
    offsets = np.array([min(grid_array[:, 0]),
                        min(grid_array[:, 1]),
                        min(grid_array[:, 2])])
    grid_array = grid_array[:, :].astype(int) - offsets

    # this is not ilines/xlines coords
    ilines_range = [np.min(grid_array[:, 0]), np.max(grid_array[:, 0]) + 1]
    xlines_range = [np.min(grid_array[:, 1]), np.max(grid_array[:, 1]) + CROP_SHAPE[1]]
    h_range = [np.min(grid_array[:, 2]), np.max(grid_array[:, 2]) + CROP_SHAPE[2]]
    predict_shape = (ilines_range[1] - ilines_range[0],
                     xlines_range[1] - xlines_range[0],
                     h_range[1] - h_range[0])

    grid_info = {'grid_array': grid_array[:, :],
         'predict_shape': predict_shape,
         'crop_shape': crop_shape,
         'cube_name': ds.indices[cube_index],
         'range': [ilines_range, xlines_range, h_range],
         'offsets': offsets}
    return grid_info

def make_slice_prediction(ds, points, crop_shape, max_iters=10, WIDTH = 10, STRIDE = 32,
                          cube_index=0, threshold=0.02, show_count=None, large_axis='xline', mode='left'):
    """ Extend horizon on one slice.
    
    Parameters
    ----------
    points : tuple or list
        lower left coordinates of the starting crop.
    crop_shape
    max_iters : int
        max_number of extension steps. If we meet end of the cube we will make less steps.
    WIDTH : int
        width of compared windows
    STRIDE
    cube_index
    threshold : float
        threshold for predicted mask
    show_count : int
        Number of extension steps to show
    large_axis :
        If 'xline',
    mode : str
        if left increase next point's line coordinates otherwise decrease it.

    Returns
    -------
        ds : Cubeset
            Cubeset with updated `predicted_labels` attribute.
        grid_info : dict
            grid info based on the grid array with lower left coordinates of the crops
    """
    show_count = max_iters if show_count is None else show_count

    geom = ds.geometries[ds.indices[cube_index]]
    grid_array = []
    if large_axis == 'iline':
        axes = (1, 0, 2)
        strides_candidates = [[0, 0, -STRIDE], [0, 0, STRIDE], [STRIDE, 0, 0]]
    else:
        axes = (0, 1, 2)
        strides_candidates = [[0, 0, -STRIDE], [0, 0, STRIDE], [0, STRIDE, 0]]
        strides_candidates[2] = [0, -STRIDE, 0] if mode != 'left' else strides_candidates[2]
    load_components_ppl = (Pipeline()
                            .load_component(src=[D('geometries'), D('labels')],
                                            dst=['geometries', 'labels'])
                            .add_components('predicted_labels'))
    predict_ppl = (Pipeline()
                    .load_component(src=[D('predicted_labels')], dst=['predicted_labels'])
                    .load_cubes(dst='data_crops')
                    .create_masks(dst='mask_crops', width=1, single_horizon=True, src_labels='labels')
                    .create_masks(dst='cut_mask_crops', width=1, single_horizon=True, src_labels='predicted_labels')
                    .apply_transform(np.transpose, axes=axes, src=['data_crops', 'mask_crops', 'cut_mask_crops'])
                    .rotate_axes(src=['data_crops', 'mask_crops', 'cut_mask_crops'])
                    .scale(mode='normalize', src='data_crops')
                    .add_axis(src='mask_crops', dst='mask_crops')
                    .import_model('extension', train_pipeline)
                    .init_variable('result_preds', init_on_each_run=list())
                    .predict_model('extension', fetches='sigmoid',
                                   make_data=make_data_extension,
                                   save_to=V('result_preds', mode='e')))

    for i in range(max_iters):
        if (points[0] + crop_shape[0] > geom.ilines_len or 
            points[1] + crop_shape[1] > geom.xlines_len or points[2] + crop_shape[2] > geom.depth):
            print("End of the cube")
            break
        grid_array.append(points)
        ds_points = np.array([[ds.indices[cube_index], *points, None]])[:, :4]
        crop_ppl = Pipeline().crop(points=ds_points, shape=crop_shape, passdown='predicted_labels')
                              
        next_predict_pipeline = (load_components_ppl + crop_ppl + predict_ppl) << ds
        btch = next_predict_pipeline.next_batch(3, n_epochs=None)
        result = next_predict_pipeline.get_variable('result_preds')[0]

        # transform cube coordinates to ilines-xlines
        i_shift, x_shift, h_shift = [slices[0] for slices in btch.slices[0]]
        transforms = [lambda i_: ds.geometries[ds.indices[cube_index]].ilines[i_ + i_shift], lambda x_: ds.geometries[ds.indices[cube_index]].xlines[x_ + x_shift],
                      lambda h_: h_ + h_shift]
        if large_axis == 'iline':
            ds.get_point_cloud(np.moveaxis(result, -1, 1), threshold=threshold, dst='predicted_mask', coordinates=None,
                               separate=True, transforms=transforms)
        else:
            ds.get_point_cloud(np.moveaxis(result, -1, 0), threshold=threshold, dst='predicted_mask', coordinates=None,
                               separate=True, transforms=transforms)
        try:
            numba_horizons = convert_to_numba_dict(ds.predicted_mask[0])
        except IndexError:
            print('Empty predicted mask')
            break
        assembled_horizon_dict = update_horizon_dict(ds.predicted_labels[ds.indices[cube_index]],
                                                     numba_horizons)
        ds.predicted_labels = {ds.indices[cube_index]: assembled_horizon_dict}

        # compute next points:
        compared_slices_ = []
        prediction = result[:, :, 0]
        compared_slices_.append(np.sum(prediction[:, :WIDTH]))
        compared_slices_.append(np.sum(prediction[:, crop_shape[1] - WIDTH]))
        compared_slices_.append(np.sum(prediction[crop_shape[2] - WIDTH:, :]))
        stride = strides_candidates[np.argmax(np.array(compared_slices_))]
        points = [sum(x) for x in zip(points, stride)]

        if i < show_count:
            print('----------------')
            print(i)
            print('argmax ', np.argmax(np.array(compared_slices_)))
            print('next stride ', stride)
            print('selected next points ', points)
            plot_extension_history(next_predict_pipeline, btch)

        if len(ds.predicted_labels) == 0:
            break

    # assemble grid_info
    grid_info = make_grid_info(grid_array, cube_index, crop_shape)
    return ds, grid_info

def ds_compute_metrics(ds, time_interval=2.5, cube_index=0):
    predicted_hor = ds.predicted_labels[ds.indices[cube_index]]

    hor = predicted_hor
    labels = ds.labels[ds.indices[cube_index]]

    res, not_present = [], 0
    vals, vals_true = [], []

    for key, val in hor.items():
        if labels.get(key) is not None:
            true_horizonts = labels[key]
            diff = abs(true_horizonts - (val[0]+1))
            idx = np.argmin(diff)

            res.append(diff[idx])
            vals_true.append(true_horizonts[idx])
            vals.append(val)
        else:
            not_present += 1

    print('Mean value/std of error:                  {:8.7} / {:8.7}'.format(np.mean(res), np.std(res)))
    print('Horizont length:                          {}'.format(len(hor)))
    print('Rate in 5 ms window:                      {:8.7}'.format(sum(np.array(res) < time_interval) / len(res)))
    print('Average height/std of true horizont:      {:8.7}'.format(np.mean(vals_true)))
    print('Average height/std of predicted horizont: {:8.7}'.format(np.mean(vals)))
    print('Number of values that were labeled by model and not labeled by experts: {}'.format(not_present))

    plt.title('Distribution of errors')
    _ = plt.hist(res, bins=100)
    
def make_area_prediction(ds, prior_mask_range, crop_shape, max_iters=30, show_count=0, cube_index=0):
    """ Predict for cube 3 based on predefined traversal order """
    _ilines, _xlines, _hs = prior_mask_range
    all_grid_info = []
    for iline in range(*_ilines, 1):
        points = [iline, _xlines[0], _hs[0]]
        next_points = [iline, _xlines[0] + 32, _hs[0]]
        ds = load_prior_mask(ds, points, cube_index=cube_index, show_prior_mask=False)

        dict_update = ds.prior_mask[ds.indices[cube_index]]
        if hasattr(ds, 'predicted_labels'):
            dict_update = update_horizon_dict(ds.predicted_labels[ds.indices[cube_index]], ds.prior_mask[ds.indices[cube_index]])
        ds.predicted_labels = {ds.indices[cube_index]: dict_update}

        ds, grid_info = make_slice_prediction(ds, next_points, crop_shape=crop_shape, cube_index=cube_index, max_iters=10, show_count=0)
        all_grid_info.append(grid_info)
    print("==============================")
    for xline in range(_xlines[0], _xlines[0] + 32):
        new_points = _ilines[0], xline, _hs[0]
        new_crop_shape = [crop_shape[1], crop_shape[0], crop_shape[2]]
        ds, grid_info = make_slice_prediction(ds, new_points, crop_shape=new_crop_shape, cube_index=cube_index, large_axis='iline',
                                              max_iters=max_iters, show_count=0, threshold=0.001)
        all_grid_info.append(grid_info)

    print("==============================")
    for iline in range(_ilines[1], ds.geometries['P_cube'].ilines_len + ds.geometries['P_cube'].ilines_offset):
        new_points = [iline, _xlines[0], _hs[0]]
        crop_shape = crop_shape
        ds, grid_info = make_slice_prediction(ds, new_points, crop_shape=crop_shape, cube_index=cube_index, large_axis='xline',
                                              max_iters=max_iters, show_count=5, threshold=0.001)
        all_grid_info.append(grid_info)
    print("==============================")
    for iline in range(_ilines[1], ds.geometries['P_cube'].ilines_len + ds.geometries['P_cube'].ilines_offset):
        new_points = [iline, _xlines[0], _hs[0]]
        crop_shape = crop_shape
        ds, grid_info = make_slice_prediction(ds, new_points, crop_shape=crop_shape, cube_index=cube_index, large_axis='xline',
                                              max_iters=max_iters, show_count=5, mode='right', threshold=0.001)
        all_grid_info.append(grid_info)
    return ds

def show_saved_horizon(ds, start_ranges, grid_info, btch, update_dict=True):
    """ To be improved """
    geom = ds.geometries[ds.indices[0]]
    start_point = start_ranges[0][0], start_ranges[1][0], start_ranges[2][0]

    next_points = np.array([[ds.indices[0], *start_point, None]])[:, :4]

    if update_dict:
        i_shift, x_shift, h_shift = np.minimum(np.array(grid_info['offsets']), np.array(start_point))
        print('i_shift, x_shift, h_shift ', i_shift, x_shift, h_shift)
        transforms = [lambda i_: ds.geometries[ds.indices[0]].ilines[i_ + i_shift], lambda x_: ds.geometries[ds.indices[0]].xlines[x_ + x_shift],
                      lambda h_: h_ + h_shift]
        ds.get_point_cloud(btch.extension_assembled_pred, threshold=0.7, dst='predicted_labels', coordinates=None,
                           separate=True, transforms=transforms)
        numba_horizons = convert_to_numba_dict(ds.predicted_labels[0])
        ds.predicted_labels = {ds.indices[0]: numba_horizons}
    else:
        print('will use saved dict')

    load_components_ppl = (Pipeline()
                            .load_component(src=[D('geometries'), D('labels')],
                                            dst=['geometries', 'labels'])
                            .add_components('predicted_labels')
                            .crop(points=next_points, shape=grid_info['predict_shape'], passdown='predicted_labels')
                            .load_component(src=[D('predicted_labels')],
                                            dst=['predicted_labels'])
                            .load_cubes(dst='data_crops')
                            .create_masks(dst='mask_crops', width=2, single_horizon=True, src_labels='labels')
                            .create_masks(dst='cut_mask_crops', width=1, single_horizon=True,
                                          src_labels='predicted_labels'))
    batch = (load_components_ppl << ds).next_batch(3)
    
    plt.figure(figsize=(30, 20))
    plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.imshow(batch.mask_crops[0][0].T, cmap="Blues")
    plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
    plt.show()

    plt.figure(figsize=(30, 20))
    plt.imshow(batch.cut_mask_crops[0][0].T, cmap="Blues")
    plt.imshow(batch.mask_crops[0][0].T, cmap="Greens", alpha=0.5)

    plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
    plt.show()
    
    return ds