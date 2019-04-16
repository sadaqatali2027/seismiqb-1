""" Utility functions. """
import numpy as np
import segyio
from tqdm import tqdm
import pandas as pd
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
    """ Read point cloud of horizont-labels from files.
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


def cube_predict(pipeline, model_name, dataset,
                 crop_shape, ilines_range, xlines_range, h_range,
                 stride=None, cube_number=0, batch_size=16, aggregation_function=None):

    cube_name = dataset.indices[cube_number]
    geom = dataset.geometries[cube_name]

    if stride is None:
        stride = crop_shape

    if isinstance(ilines_range, (list, tuple)):
        i_low = min(geom.ilines_len-crop_shape[0], ilines_range[0])
        i_high = min(geom.ilines_len-crop_shape[0], ilines_range[1])

    if isinstance(xlines_range, (list, tuple)):
        x_low = min(geom.xlines_len-crop_shape[1], xlines_range[0])
        x_high = min(geom.xlines_len-crop_shape[1], xlines_range[1])

    if isinstance(h_range, (list, tuple)):
        h_low = min(geom.depth-crop_shape[2], h_range[0])
        h_high = min(geom.depth-crop_shape[2], h_range[1])

    ilines_range = np.arange(i_low, i_high+1, stride[0])
    xlines_range = np.arange(x_low, x_high+1, stride[1])
    h_range = np.arange(h_low, h_high+1, stride[2])

    grid = []
    grid_true = []
    for il in ilines_range:
        for xl in xlines_range:
            for h in h_range:
                point = [cube_name, il/2000, xl/2000, h/2000]
                grid.append(point)
                grid_true.append([cube_name, il, xl, h])
    grid = np.array(grid, dtype=object)
    print('RANGES::')
    print(ilines_range)
    print(xlines_range)
    print(h_range)
    print('Length of grid: ', len(grid))

    img_crops, mask_crops = [], []
    pred_crops = []
    for i in range(0, len(grid), batch_size):
        points = grid[i:i+batch_size]

        predict_pipeline = (Pipeline()
                            .load_component(src=[D('geometries'), D('labels')],
                                            dst=['geometries', 'labels'])
                            .crop(points=points, shape=crop_shape)
                            .load_cubes(dst='data_crops')
                            .load_masks(dst='mask_crops')
                            .import_model(model_name, pipeline)
                            .init_variable('result', init_on_each_run=list())
                            .predict_model(model_name,
                                           fetches=['cubes', 'masks', 'predictions', 'loss'],
                                           make_data=make_data,
                                           save_to=V('result'), mode='a')
                         ) << dataset
        predict_pipeline.next_batch(1, n_epochs=None)

        img_crops.extend(predict_pipeline.get_variable('result')[0][0])
        mask_crops.extend(predict_pipeline.get_variable('result')[0][1])
        pred_crops.extend(predict_pipeline.get_variable('result')[0][2])

    img_full = np.zeros((i_high+crop_shape[0], x_high+crop_shape[1], h_high+crop_shape[2]))
    for i, point in enumerate(grid_true):
        img_full[point[1]:point[1]+crop_shape[0], point[2]:point[2]+crop_shape[1], point[3]:point[3]+crop_shape[2]] += img_crops[i].T

    return img_full
