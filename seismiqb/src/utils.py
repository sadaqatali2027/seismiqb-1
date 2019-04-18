""" Utility functions. """
import numpy as np
import segyio
from tqdm import tqdm
import pandas as pd

import numba
from numba.typed import Dict
from numba import types
from numba import njit


FILL_VALUE = -999



def repair(path_cube, geometry, path_save,
           i_low=0, i_high=-2, x_low=0, x_high=-2):
    """ Cuts unnecessary inlines/xlines from cube. """
    with segyio.open(path_cube, 'r', strict=False) as src:
        src.mmap()
        spec = segyio.spec()
        spec.sorting = int(src.sorting)
        spec.format = int(src.format)
        spec.samples = range(geometry.depth)
        spec.ilines = geometry.ilines[i_low:i_high]
        spec.xlines = geometry.xlines[x_low:x_high]

        with segyio.create(path_save, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]

            c = 0
            for il_ in tqdm(spec.ilines):
                for xl_ in spec.xlines:
                    tr_ = geometry.il_xl_trace[(il_, xl_)]
                    dst.header[c] = src.header[tr_]
                    dst.header[c][segyio.TraceField.FieldRecord] = il_
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_FILE] = il_

                    dst.header[c][segyio.TraceField.TraceNumber] = xl_ - geometry.xlines_offset
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_LINE] = xl_ - geometry.xlines_offset
                    dst.trace[c] = src.trace[tr_]
                    c += 1
            dst.bin = src.bin
            dst.bin = {segyio.BinField.Traces: c}

    # Check that repaired cube can be opened in 'strict' mode
    with segyio.open(path_save, 'r', strict=True) as _:
        pass


def read_point_cloud(paths, default=None, order=('iline', 'xline', 'height'), transforms=None, **kwargs):
    """ Read point cloud of labels from files using pandas.

    Parameters
    ----------
    paths : str or tuple or list
        array-like of paths to files containing point clouds (table of floats with several columns).
    default : dict
        dict containing arguments of pandas parser; will be used for parsing all supplied files.
    order : array-like
        specifies the order of columns of the resulting array.
    transforms : array-like
        contains list of vectorized transforms. Each transform is applied to a column of the resulting array.
    **kwargs
        file-specific arguments of pandas parser in format `{paths[0]: args_0, paths[1]: args_1, ...}`.
        Each dict updates `default`-args and then usef for parsing a specific file.

    Returns
    -------
    ndarray
        resulting point-cloud.
    """
    paths = (paths, ) if isinstance(paths, str) else paths

    # default params of pandas-parser
    if default is None:
        default = dict(sep='\s+', names=['xline', 'iline', 'height']) #pylint: disable=anomalous-backslash-in-string

    # read point clouds
    point_clouds = []
    for path in paths:
        copy = default.copy()
        copy.update(kwargs.get(path, dict()))
        cloud = pd.read_csv(path, **copy)
        point_clouds.append(cloud.loc[:, order].values)

    points = np.concatenate(point_clouds)

    # apply transforms
    if transforms is not None:
        for i in range(points.shape[-1]):
            points[:, i] = transforms[i](points[:, i])

    return points


def make_labels_dict(point_cloud):
    """ Make labels-dict using cloud of points.

    Parameters
    ----------
    point_cloud : array
        array `(n_points, 3)`, contains point cloud of labels in format `(x, y, z)`.

    Returns
    -------
    numba.Dict
        dict of labels `{(x, y): [z_1, z_2, ...]}`.
    """
    # round and cast
    ilines_xlines = np.round(point_cloud[:, :2]).astype(np.int64)

    # make dict-types
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]

    # find max array-length
    counts = Dict.empty(key_type, types.int64)

    @njit
    def fill_counts_get_max(counts, ilines_xlines):
        max_count = 0
        for i in range(len(ilines_xlines)):
            il, xl = ilines_xlines[i, :2]
            if counts.get((il, xl)) is None:
                counts[(il, xl)] = 0

            counts[(il, xl)] += 1
            if counts[(il, xl)] > max_count:
                max_count = counts[(il, xl)]

        return max_count

    max_count = fill_counts_get_max(counts, ilines_xlines)

    # put key-value pairs into numba-dict
    labels = Dict.empty(key_type, value_type)

    @njit
    def fill_labels(labels, counts, ilines_xlines, max_count):
        # zero-out the counts
        for k in counts.keys():
            counts[k] = 0

        # fill labels-dict
        for i in range(len(ilines_xlines)):
            il, xl = ilines_xlines[i, :2]
            if labels.get((il, xl)) is None:
                labels[(il, xl)] = np.full((max_count, ), FILL_VALUE, np.int64)

            labels[(il, xl)][counts[(il, xl)]] = point_cloud[i, 2]
            counts[(il, xl)] += 1

    fill_labels(labels, counts, ilines_xlines, max_count)
    return labels


@njit
def create_mask(ilines_, xlines_, hs_,
                il_xl_h, geom_ilines, geom_xlines, geom_depth,
                mode, width):
    """ Jit-decorated function for fast mask creation from point cloud data stored in numba.typed.Dict.
    This function is usually called inside SeismicCropBatch's method load_masks.
    """
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))

    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = geom_ilines[iline_], geom_xlines[xline_]
            if il_xl_h.get((il_, xl_)) is None:
                continue
            m_temp = np.zeros(geom_depth)
            if mode == 'horizon':
                for height_ in il_xl_h[(il_, xl_)]:
                    m_temp[max(0, height_ - width):min(height_ + width, geom_depth)] += 1
            elif mode == 'stratum':
                current_col = 1
                start = 0
                sorted_heights = sorted(il_xl_h[(il_, xl_)])
                for height_ in sorted_heights:
                    if start > hs_[-1]:
                        break
                    m_temp[start:height_ + 1] = current_col
                    start = height_ + 1
                    current_col += 1
                    m_temp[sorted_heights[-1] + 1:min(hs_[-1] + 1, geom_depth)] = current_col
            else:
                raise ValueError('Mode should be either `horizon` or `stratum`')
            mask[i, j, :] = m_temp[hs_]
    return mask


@njit
def _count_nonzeros(array):
    """ Definetely not empty docstring. """
    count = 0
    for i in array:
        if i != 0:
            count += 1
    return max(count, 1)


@njit
def _aggregate(array_crops, crop_shape, array_grid,
               template, aggr_func):
    """ Definetely not empty docstring. """
    total = len(array_grid)

    for il in range(template.shape[0]):
        for xl in range(template.shape[1]):
            for h in range(template.shape[2]):

                temp_arr = np.zeros(total)
                for i in range(total):
                    il_crop = array_grid[i, 0]
                    xl_crop = array_grid[i, 1]
                    h_crop = array_grid[i, 2]

                    if 0 <= (il - il_crop) < crop_shape[0] and \
                       0 <= (xl - xl_crop) < crop_shape[1] and \
                       0 <= (h - h_crop) < crop_shape[2]:
                        dot = array_crops[i, xl-xl_crop, h-h_crop, il-il_crop] # crops are in (xl, h, il) order
                        temp_arr[i] = dot

                template[il, xl, h] = aggr_func(temp_arr)
    return template


def cube_predict(dataset, pipeline, model_name,
                 crop_shape, ilines_range, xlines_range, h_range,
                 strides=None, batch_size=16, cube_number=0,
                 aug_pipeline=None, mode='avg'):
    """ Definetely not empty docstring. """

    ### PART ONE. GRID
    cube_name = dataset.indices[cube_number]
    geom = dataset.geometries[cube_name]

    strides = strides or crop_shape

    i_low = min(geom.ilines_len-crop_shape[0], ilines_range[0])
    i_high = min(geom.ilines_len-crop_shape[0], ilines_range[1])

    x_low = min(geom.xlines_len-crop_shape[1], xlines_range[0])
    x_high = min(geom.xlines_len-crop_shape[1], xlines_range[1])

    h_low = min(geom.depth-crop_shape[2], h_range[0])
    h_high = min(geom.depth-crop_shape[2], h_range[1])

    ilines_range = np.arange(i_low, i_high+1, strides[0])
    xlines_range = np.arange(x_low, x_high+1, strides[1])
    h_range = np.arange(h_low, h_high+1, strides[2])

    grid = []
    for il in ilines_range:
        for xl in xlines_range:
            for h in h_range:
                point = [cube_name, il, xl, h]
                grid.append(point)
    grid = np.array(grid, dtype=object)


    ### PART TWO. PREDICTION
    aug_pipeline = aug_pipeline or Pipeline().scale(mode='normalize', src='data_crops')
    model_pipeline = (Pipeline()
                      .import_model(model_name, pipeline)
                      .init_variable('result', init_on_each_run=list())
                      .predict_model(model_name,
                                     fetches=['cubes', 'masks', 'predictions', 'loss'],
                                     make_data=make_data,
                                     save_to=V('result'), mode='a'))

    img_crops, mask_crops, pred_crops = [], [], []
    for i in range(0, len(grid), batch_size):

        points = grid[i:i+batch_size]
        load_pipeline = (Pipeline()
                         .load_component(src=[D('geometries'), D('labels')],
                                         dst=['geometries', 'labels'])
                         .crop(points=points, shape=crop_shape)
                         .load_cubes(dst='data_crops')
                         .load_masks(dst='mask_crops')
                         )

        predict_pipeline = (load_pipeline + aug_pipeline + model_pipeline) << dataset
        predict_pipeline.next_batch(2, n_epochs=None)

        img_crops.extend(predict_pipeline.get_variable('result')[0][0])
        mask_crops.extend(predict_pipeline.get_variable('result')[0][1])
        pred_crops.extend(predict_pipeline.get_variable('result')[0][2])


    ### PART THREE. AGGREGATION
    if mode == 'avg':
        @njit
        def _callable(array):
            return np.sum(array) / _count_nonzeros(array)
    elif mode == 'max':
        @njit
        def _callable(array):
            return np.max(array)
    elif isinstance(mode, numba.targets.registry.CPUDispatcher):
        _callable = mode

    template = np.zeros((i_high-i_low+crop_shape[0],
                         x_high-x_low+crop_shape[1],
                         h_high-h_low+crop_shape[2]))
    array_grid = grid[:, 1:].astype(int) - [min(ilines_range), min(xlines_range), min(h_range)]


    img_crops = np.array(img_crops)
    img_full = _aggregate(img_crops, crop_shape, array_grid,
                          np.zeros_like(template), aggr_func=_callable)
    img_full = img_full[:(i_high-i_low), :(x_high-x_low), :(h_high-h_low)]


    mask_crops = np.squeeze(np.array(mask_crops), axis=-1)
    mask_full = _aggregate(mask_crops, crop_shape, array_grid,
                           np.zeros_like(template), aggr_func=_callable)
    mask_full = mask_full[:(i_high-i_low), :(x_high-x_low), :(h_high-h_low)]


    pred_crops = np.squeeze(np.array(pred_crops), axis=-1)
    pred_full = _aggregate(pred_crops, crop_shape, array_grid,
                           np.zeros_like(template), aggr_func=_callable)
    pred_full = pred_full[:(i_high-i_low), :(x_high-x_low), :(h_high-h_low)]
    return img_full, mask_full, pred_full
