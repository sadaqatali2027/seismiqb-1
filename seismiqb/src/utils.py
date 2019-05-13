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
        """ Fill in counts-dict.
        """
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
        """ Fill in labels-dict.
        """
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
    """ Jit-accelerated function for fast mask creation from point cloud data stored in numba.typed.Dict.
    This function is usually called inside SeismicCropBatch's method `load_masks`.
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
                    m_temp[max(0, height_ - width):min(height_ + width, geom_depth)] = 1
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

        for i, x in horizon_.index.values:
            il_xl = (transforms[0](i), transforms[1](x))
            height = transforms[2](horizon_.loc[(i, x), 'height'])

            if separate:
                horizons[n_horizon][il_xl] = height
            else:
                if il_xl in horizons:
                    horizons[il_xl].append(height)
                else:
                    horizons[il_xl] = [height]

    return horizons


@njit
def count_nonzeros(array):
    """ Jit-accelerated function to count non-zero elements. Faster than numpy version. """
    count = 0
    for i in array:
        if i != 0:
            count += 1
    return count


@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape, aggr_func):
    """ Jit-accelerated function to glue together crops according to grid.
    This function is usually called inside SeismicCropBatch's method `assemble_crops`.
    """
    total = len(array_grid)
    background = np.zeros(predict_shape)

    for il in range(background.shape[0]):
        for xl in range(background.shape[1]):
            for h in range(background.shape[2]):

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

                background[il, xl, h] = aggr_func(temp_arr)
    return background


@njit
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to nearest value from the other.
    Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.

    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from `ticks`.
    """
    for i, p in enumerate(values):
        if p <= ticks[0]:
            values[i] = ticks[0]
        elif p >= ticks[-1]:
            values[i] = ticks[-1]
        else:
            ix = np.searchsorted(ticks, p)

            if abs(ticks[ix] - p) <= abs(ticks[ix-1] - p):
                values[i] = ticks[ix]
            else:
                values[i] = ticks[ix-1]
    return values
