""" Utility functions. """
import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm
from skimage.measure import label, regionprops
from numba import njit, types
from numba.typed import Dict


FILL_VALUE = -999


def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

    Notes
    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.

    Parameters
    ----------
    path : str
        Location of original .sgy cube.

    geometry : SeismicGeometry
        Infered information about original cube.

    path_save : str
        Place to save subcube.

    i_range : array-like
        Ilines to include in subcube.

    x_range : array-like
        Xlines to include in subcube.
    """

    i_low, i_high = i_range[0], i_range[-1]
    x_low, x_high = x_range[0], x_range[-1]

    with segyio.open(path, 'r', strict=False) as src:
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


def convert_point_cloud(path, path_save, names=None, order=None):
    """ Change set of columns in file with point cloud labels.
    Usually is used to remove redundant columns.

    Parameters
    ----------
    path : str
        Path to file to convert.

    path_save : str
        Path for the new file to be saved to.

    names : str or sequence of str
        Names of columns in the original file. Default is Petrel's export format, which is
        ('_', '_', 'iline', '_', '_', 'xline', 'cdp_x', 'cdp_y', 'height'), where `_` symbol stands for
        redundant keywords like `INLINE`.

    order : str or sequence of str
        Names and order of columns to keep. Default is ('iline', 'xline', 'cdp_x', 'cdp_y', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'cdp_x', 'cdp_y', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    to_save = df.loc[:, order]
    to_save.to_csv(path_save, sep=' ', index=False, header=False)


def read_point_cloud(paths, names=None, order=None, transforms=None, **kwargs):
    """ Read point cloud of labels from files using pandas.

    Parameters
    ----------
    paths : str or tuple or list
        array-like of paths to files containing point clouds (table of floats with several columns).
    names : sequence
        sequence of column names in files.
    order : array-like
        specifies the order of columns to keep in the resulting array.
    transforms : array-like
        contains list of vectorized transforms. Each transform is applied to a column of the resulting array.
    **kwargs
        file-specific arguments of pandas parser.

    Returns
    -------
    ndarray
        resulting point-cloud. First three columns contain `(x, y, z)`-coords while the last one stores
        horizon-number.
    """
    #pylint: disable=anomalous-backslash-in-string
    paths = [paths] if isinstance(paths, str) else paths

    # default params of pandas-parser
    names = names or ['iline', 'xline', 'cdp_x', 'cdp_y', 'height']
    order = order or ['cdp_y', 'cdp_x', 'height']

    # read point clouds
    point_clouds = []
    for ix, path in enumerate(paths):
        cloud = pd.read_csv(path, sep='\s+', names=names, usecols=set(order), **kwargs)

        temp = np.hstack([cloud.loc[:, order].values,
                          np.ones((cloud.shape[0], 1)) * ix])
        point_clouds.append(temp)

    points = np.concatenate(point_clouds)

    # apply transforms
    if transforms is not None:
        for i in range(points.shape[-1] - 1):
            points[:, i] = transforms[i](points[:, i])

    return points


def make_labels_dict(point_cloud):
    """ Make labels-dict using cloud of points.

    Parameters
    ----------
    point_cloud : array
        array `(n_points, 4)`, contains point cloud of labels in format `(x, y, z, horizon_number)`.

    Returns
    -------
    numba.Dict
        dict of labels `{(x, y): [z_1, z_2, ...]}`.
    """
    # round and cast
    ilines_xlines = np.round(point_cloud[:, :2]).astype(np.int64)
    max_count = int(point_cloud[-1, -1]) + 1

    # make typed Dict
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    labels = Dict.empty(key_type, value_type)

    @njit
    def fill_labels(labels, ilines_xlines, point_cloud, max_count):
        """ Fill in labels-dict.
        """
        for i in range(len(ilines_xlines)):
            il, xl = ilines_xlines[i, :2]
            if labels.get((il, xl)) is None:
                labels[(il, xl)] = np.full((max_count, ), FILL_VALUE, np.int64)

            idx = int(point_cloud[i, 3])
            labels[(il, xl)][idx] = point_cloud[i, 2]

    fill_labels(labels, ilines_xlines, point_cloud, max_count)
    return labels


@njit
def create_mask(ilines_, xlines_, hs_,
                il_xl_h, geom_ilines, geom_xlines, geom_depth,
                mode, width):
    """ Jit-accelerated function for fast mask creation from point cloud data stored in numba.typed.Dict.
    This function is usually called inside SeismicCropBatch's method `load_masks`.
    """
    #pylint: disable=too-many-nested-blocks
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))

    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = geom_ilines[iline_], geom_xlines[xline_]
            if il_xl_h.get((il_, xl_)) is None:
                continue
            m_temp = np.zeros(geom_depth)
            if mode == 'horizon':
                for height_ in il_xl_h[(il_, xl_)]:
                    if height_ != FILL_VALUE:
                        m_temp[max(0, height_ - width):min(height_ + width, geom_depth)] = 1
            elif mode == 'stratum':
                current_col = 1
                start = 0
                sorted_heights = sorted(il_xl_h[(il_, xl_)])
                for height_ in sorted_heights:
                    if height_ == FILL_VALUE:
                        height_ = start
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


def _get_horizons(mask, threshold, averaging, transforms, separate=False):
    """ Compute horizons from a mask.
    """
    mask_ = np.zeros_like(mask, np.int32)
    mask_[mask >= threshold] = 1

    # get regions
    labels = label(mask_)
    regions = regionprops(labels)

    # make horizons-structure
    horizons = dict() if not separate else []
    for n_horizon, region in enumerate(regions):
        if separate:
            horizons.append(dict())

        # compute horizon-height for each inline-xline
        coords = region.coords
        coords = pd.DataFrame(coords, columns=['iline', 'xline', 'height'])
        horizon_ = getattr(coords.groupby(['iline', 'xline']), averaging)()

        # separate the columns
        ilines = horizon_.index.get_level_values('iline').values
        xlines = horizon_.index.get_level_values('xline').values
        heights = horizon_.values

        # transform each column
        ilines_ = transforms[0](ilines)
        xlines_ = transforms[1](xlines)
        heights_ = np.ravel(transforms[2](heights))

        if separate:
            for key, h in zip(zip(ilines_, xlines_), heights_):
                horizons[n_horizon][key] = h
        else:
            for key, h in zip(zip(ilines_, xlines_), heights_):
                if key in horizons:
                    horizons[key].append(h)
                else:
                    horizons[key] = [h]

    return horizons


def dump_horizon(horizon, geometry, path_save, offset=1):
    """ Save horizon as point cloud.

    Parameters
    ----------
    horizon : dict
        Mapping from pairs (iline, xline) to height.

    geometry : SeismicGeometry
        Information about cube

    path_save : str
        Path for the horizon to be saved to.

    offset : int, float
        Shift horizont before applying inverse transform.
        Usually is used to take into account different numeration bases:
        Petrel uses 1-based numeration, whilst Python uses 0-based numberation.
    """
    ixh = []
    for (i, x), h in horizon.items():
        ixh.append([i, x, h])
    ixh = np.asarray(ixh)

    cdp_xy = geometry.lines_to_abs(ixh)

    h = (ixh[:, -1] + offset) * geometry.sample_rate + geometry.delay

    data = np.hstack([ixh[:, :2], cdp_xy[:, :2]])
    data[:, -1] += 1 # take into account that initial horizonts are 1-based

    df = pd.DataFrame(data, columns=['iline', 'xline', 'cdp_y', 'cdp_x', 'height'])
    df.sort_values(['iline', 'xline'], inplace=True)
    df.to_csv(path_save, sep=' ', columns=['iline', 'xline', 'cdp_x', 'cdp_y', 'height'],
              index=False, header=False)


@njit
def count_nonfill(array):
    """ Jit-accelerated function to count non-fill elements. """
    count = 0
    for i in array:
        if i != FILL_VALUE:
            count += 1
    return count


@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape):
    """ Jit-accelerated function to glue together crops according to grid.
    At positions, where different crops overlap, only the maximum value is saved.
    This function is usually called inside SeismicCropBatch's method `assemble_crops`.
    """
    #pylint: disable=assignment-from-no-return
    total = len(array_grid)
    background = np.zeros(predict_shape)

    for i in range(total):
        il, xl, h = array_grid[i, :]
        il_end = min(background.shape[0], il+crop_shape[0])
        xl_end = min(background.shape[1], xl+crop_shape[1])
        h_end = min(background.shape[2], h+crop_shape[2])

        crop = np.transpose(array_crops[i], (2, 0, 1))
        crop = crop[:(il_end-il), :(xl_end-xl), :(h_end-h)]
        previous = background[il:il_end, xl:xl_end, h:h_end]
        background[il:il_end, xl:xl_end, h:h_end] = np.maximum(crop, previous)
    return background


@njit(parallel=True)
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.

    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        ticks_ = ticks[i]
        if p <= ticks_[0]:
            values[i] = ticks_[0]
        elif p >= ticks_[-1]:
            values[i] = ticks_[-1]
        else:
            ix = np.searchsorted(ticks_, p)

            if abs(ticks_[ix] - p) <= abs(ticks_[ix-1] - p):
                values[i] = ticks_[ix]
            else:
                values[i] = ticks_[ix-1]
    return values
