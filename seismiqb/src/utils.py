""" Utility functions. """
import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, selem
from numba import njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt


FILL_VALUE = -999
FILL_VALUE_A = -999999


def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

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

    Notes
    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.
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



def convert_point_cloud(path, path_save, names=None, order=None, transform=None):
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
        Names and order of columns to keep. Default is ('iline', 'xline', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    data = df.loc[:, order]
    if transform:
        data = data.apply(transform)
    data.to_csv(path_save, sep=' ', index=False, header=False)


def read_point_cloud(paths, names=None, order=None, **kwargs):
    """ Read point cloud of labels from files using pandas.

    Parameters
    ----------
    paths : str or tuple or list
        array-like of paths to files containing point clouds (table of floats with several columns).
    names : sequence
        sequence of column names in files.
    order : array-like
        specifies the order of columns to keep in the resulting array.
    **kwargs
        additional keyword-arguments of pandas parser.

    Returns
    -------
    ndarray
        resulting point-cloud. First three columns contain (iline, xline, height) while the last one stores
        horizon number.
    """
    #pylint: disable=anomalous-backslash-in-string
    paths = [paths] if isinstance(paths, str) else paths

    # read point clouds
    point_clouds = []
    for ix, path in enumerate(paths):
        names = names or _get_default_names(path)
        order = order or names
        cloud = pd.read_csv(path, sep='\s+', names=names, usecols=set(order), **kwargs)

        temp = np.hstack([cloud.loc[:, order].values,
                          np.ones((cloud.shape[0], 1)) * ix])
        point_clouds.append(temp)
    point_cloud = np.concatenate(point_clouds)
    return np.rint(point_cloud).astype(np.int64)

def _get_default_names(path):
    with open(path) as file:
        line = file.readline().split(' ')
    if len(line) == 3:
        return ['iline', 'xline', 'height']
    return ['iline', 'xline', 's_point', 'e_point']


@njit
def filter_point_cloud(point_cloud, filtering_matrix, ilines_offset, xlines_offset):
    """ Remove entries corresponding to zero traces.

    Parameters
    ----------
    point_cloud : ndarray
        Point cloud with labels.
    filtering_matrix : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions to remove.
    ilines_offset, xlines_offset : int
        Offsets of numeration.
    """
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(point_cloud), dtype=np.int32)

    for i in range(len(point_cloud)):
        il, xl = point_cloud[i, 0], point_cloud[i, 1]
        if filtering_matrix[il-ilines_offset, xl-xlines_offset] == 1:
            mask[i] = 0
    return point_cloud[mask == 1, :]



def make_labels_dict(point_cloud):
    """ Make labels-dict using cloud of points.

    Parameters
    ----------
    point_cloud : array
        array `(n_points, 4)`, contains point cloud of labels in format `(il, xl, height, horizon_number)`.

    Returns
    -------
    numba.Dict
        dict of labels `{(il, xl): [h_1, h_2, ...]}`.
    """
    # round and cast
    point_cloud = np.rint(point_cloud).astype(np.int64)
    max_count = point_cloud[-1, -1] + 1

    # make typed Dict
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    labels = Dict.empty(key_type, value_type)

    @njit
    def fill_labels(labels, point_cloud, max_count):
        """ Fill in labels-dict.
        """
        for i in range(len(point_cloud)):
            il, xl = point_cloud[i, :2]
            if labels.get((il, xl)) is None:
                labels[(il, xl)] = np.full((max_count, ), FILL_VALUE, np.int64)

            idx = int(point_cloud[i, 3])
            labels[(il, xl)][idx] = point_cloud[i, 2]

    fill_labels(labels, point_cloud, max_count)
    return labels


def make_labels_dict_f(point_cloud):
    """ Make labels-dict using cloud of points.

    Parameters
    ----------
    point_cloud : array
        array `(n_points, 5)`, contains point cloud of labels in format
        `(il, xl, start_height, end_height, horizon_number)`.

    Returns
    -------
    numba.Dict
        dict of labels `{(il, xl): (start_points, end_points, class_labels)}`.
    """
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64
    counts = Dict.empty(key_type, value_type)

    @njit
    def _fill_counts(counts, point_cloud):
        """ Count number of facies on each trace. """
        for i in range(point_cloud.shape[0]):
            key = (point_cloud[i, 0], point_cloud[i, 1])

            if counts.get(key) is None:
                counts[key] = 1
            else:
                counts[key] += 1
        return counts
    counts = _fill_counts(counts, point_cloud)

    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.Tuple((types.int64[:], types.int64[:], types.int64[:]))
    labels = Dict.empty(key_type, value_type)

    def _fill_labels(labels, counts, point_cloud):
        """ Fill in actual labels.
        Every value in the dictionary is a tuple of three arrays: starting points, ending points and class labels.
        """
        for i in range(point_cloud.shape[0]):
            key = (point_cloud[i, 0], point_cloud[i, 1])
            s_point, e_point, c = point_cloud[i, 2], point_cloud[i, 3], point_cloud[i, 4]

            if key not in labels:
                count = counts[key]
                s_points = np.full((count, ), FILL_VALUE, np.int64)
                e_points = np.full((count, ), FILL_VALUE, np.int64)
                c_array = np.full((count, ), FILL_VALUE, np.int64)

                s_points[0] = s_point
                e_points[0] = e_point
                c_array[0] = c

                labels[key] = (s_points, e_points, c_array)
            else:
                (s_points, e_points, c_array) = labels[key]
                for j, point in enumerate(s_points):
                    if point == FILL_VALUE:
                        idx = j
                        break

                s_points[idx] = s_point
                e_points[idx] = e_point
                c_array[idx] = c
                labels[key] = (s_points, e_points, c_array)
        return labels

    labels = _fill_labels(labels, counts, point_cloud)
    return labels


@njit
def filter_labels(labels, filtering_matrix, ilines_offset, xlines_offset):
    """ Remove (inplace) keys from labels dictionary according to filtering_matrix.

    Parameters
    ----------
    labels : dict
        Dictionary with keys in (iline, xline) format.
    filtering_matrix : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions to remove.
    ilines_offset, xlines_offset : int
        Offsets of numeration.
    """
    n_zeros = int(np.sum(filtering_matrix))
    if n_zeros > 0:
        c = 0
        to_remove = np.zeros((n_zeros, 2), dtype=np.int64)

        for il, xl in labels.keys():
            if filtering_matrix[il - ilines_offset, xl - xlines_offset] == 1:
                to_remove[c, 0] = il
                to_remove[c, 1] = xl
                c = c + 1

        for i in range(c):
            key = (to_remove[i, 0], to_remove[i, 1])
            if key in labels: # for some reason that is necessary
                labels.pop(key)



@njit
def create_mask(ilines_, xlines_, hs_,
                il_xl_h, ilines_offset, xlines_offset, geom_depth,
                mode, width, single_horizon=False):
    """ Jit-accelerated function for fast mask creation for seismic horizons.
    This function is usually called inside SeismicCropBatch's method `create_masks`.
    """
    #pylint: disable=line-too-long, too-many-nested-blocks, too-many-branches
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))
    if single_horizon:
        single_idx = -1

    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = iline_ + ilines_offset, xline_ + xlines_offset
            if il_xl_h.get((il_, xl_)) is None:
                continue
            m_temp = np.zeros(geom_depth)
            if mode == 'horizon':
                filtered_idx = [idx for idx, height_ in enumerate(il_xl_h[(il_, xl_)])
                                if height_ != FILL_VALUE]
                filtered_idx = [idx for idx in filtered_idx
                                if il_xl_h[(il_, xl_)][idx] > hs_[0] and il_xl_h[(il_, xl_)][idx] < hs_[-1]]
                if len(filtered_idx) == 0:
                    continue
                if single_horizon:
                    if single_idx == -1:
                        single_idx = np.random.choice(filtered_idx)
                        single_idx = filtered_idx[np.random.randint(len(filtered_idx))]
                    value = il_xl_h[(il_, xl_)][single_idx]
                    m_temp[max(0, value - width):min(value + width, geom_depth)] = 1
                else:
                    for idx in filtered_idx:
                        m_temp[max(0, il_xl_h[(il_, xl_)][idx] - width):min(il_xl_h[(il_, xl_)][idx] + width, geom_depth)] = 1
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


@njit
def create_mask_f(ilines_, xlines_, hs_, il_xl_h, ilines_offset, xlines_offset, geom_depth):
    """ Jit-accelerated function for fast mask creation for seismic facies.
    This function is usually called inside SeismicCropBatch's method `create_masks`.
    """
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))

    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = iline_ + ilines_offset, xline_ + xlines_offset
            if il_xl_h.get((il_, xl_)) is None:
                continue

            m_temp = np.zeros(geom_depth)
            value = il_xl_h.get((il_, xl_))
            s_points, e_points, classes = value

            for s_p, e_p, c in zip(s_points, e_points, classes):
                m_temp[max(0, s_p):min(e_p+1, geom_depth)] = c+1
            mask[i, j, :] = m_temp[hs_]
    return mask


def check_if_joinable(horizon_1, horizon_2, border_margin=1, height_margin=1):
    """ Check whether a pair of horizons can be stitched together.
    """
    # check whether the horizons have overlap in covered area
    keyset_1, keyset_2 = set(horizon_1.keys()), set(horizon_2.keys())
    shared_keys = keyset_1 & keyset_2

    if len(shared_keys) > 0:
        # horizons have area overlap. check if they can be joined
        ctr_same, ctr_differ = 0, 0
        for key in shared_keys:
            if np.isclose(horizon_1.get(key)[0], horizon_2.get(key)[0]):
                ctr_same += 1
        else:
            ctr_differ += 1
        if ctr_same == 0:
            # horizon-areas overlap but the horizon itself differ
            return False
        elif ctr_differ > 0:
            # horizon diverge so they cannot be merged
            return False
        else:
            # horizons can be merged just fine
            return True
    else:
        # horizons don't have area overlap
        # we still have to check whether they are adjacent

        # find shared horizons-border
        xs_1, xs_2 = [{x for x, y in keyset} for keyset in [keyset_1, keyset_2]]
        ys_1, ys_2 = [{y for x, y in keyset} for keyset in [keyset_1, keyset_2]]
        min_x, min_y = min(xs_1 | xs_2), min(ys_1 | ys_2)
        max_x, max_y = max(xs_1 | xs_2), max(ys_1 | ys_2)
        area_1 = np.zeros(shape=(max_x - min_x + 1, max_y - min_y + 1))
        area_2 = np.zeros_like(area_1)

        # put each horizon on respective area-copy
        for keyset, area in zip([keyset_1, keyset_2], [area_1, area_2]):
            for x, y in keyset:
                area[x - min_x, y - min_y] = 1

        # apply dilation to each horizon-area to determine borders-intersection
        area_1 = binary_dilation(area_1, selem.diamond(border_margin))
        borders = area_1 * area_2
        border_pairs = np.argwhere(borders > 0)
        if len(border_pairs) == 0:
            # horizons don't have adjacent borders so cannot be joined
            return False
        else:
            for x, y in border_pairs:
                x, y = x + min_x, y + min_y
                height_2 = horizon_2.get((x, y))
                # get first horizon-height selecting suitable bordering pixel
                for key in ([(x + i, y) for i in range(-border_margin, border_margin + 1)] +
                            [(x, y + i) for i in range(-border_margin, border_margin + 1)]):
                    height_1 = horizon_1.get(key)
                    if height_1 is not None:
                        break
                height_1 = [-999.0] if height_1 is None else height_1

                # determine whether the horizon-heights are close
                if np.abs(height_1[0] - height_2[0]) <= height_margin:
                    return True
            return False


def merge_horizons(horizon_1, horizon_2):
    pass


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
                horizons[n_horizon][key] = [h]
        else:
            for key, h in zip(zip(ilines_, xlines_), heights_):
                if key in horizons:
                    horizons[key].append(h)
                else:
                    horizons[key] = [h]
    return horizons


def dump_horizon(horizon, geometry, path_save, idx=None, offset=1):
    """ Save horizon in a point cloud format.

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
        if idx is not None:
            h = h[idx]
        ixh.append([i, x, h])
    ixh = np.asarray(ixh)

    ixh[:, -1] = (ixh[:, -1] + offset) * geometry.sample_rate + geometry.delay

    df = pd.DataFrame(ixh, columns=['iline', 'xline', 'height'])
    df.sort_values(['iline', 'xline'], inplace=True)
    df.to_csv(path_save, sep=' ', columns=['iline', 'xline', 'height'],
              index=False, header=False)


def compare_horizons(dict_1, dict_2, printer=print, plot=False, sample_rate=1, offset=0):
    """ Compare two horizons in dictionary format.

    Parameters
    ----------
    dict_1, dict_2 : dict
        Mappings from (iline, xline) to heights. Value can be either array or one number.
    printer : callable
        Function to output results with, for example `print` or `log.info`.
    plot : bool
        Whether to plot histogram of errors.
    sample_rate : number
        Frequency of taking measures. Used to normalize 5ms window.
    offset : number
        Value to shift horizon up. Can be used to take into account different counting bases.
    """
    differences = []
    not_present_1, not_present_2 = 0, 0
    vals_1, vals_2 = [], []

    for key, val_1 in dict_1.items():
        try:
            val_1 = val_1[0]
        except IndexError:
            pass

        val_2 = dict_2.get(key)
        if val_2 is not None:
            diff_ = abs(val_2 - val_1 - offset)
            idx = np.argmin(diff_)
            diff = diff_[idx]
            differences.append(diff)

            vals_1.append(val_1)
            vals_2.append(val_2[idx])
        else:
            not_present_1 += 1

    for key, val_2 in dict_2.items():
        if dict_1.get(key) is None:
            not_present_2 += 1

    printer('First horizont length:                    {}'.format(len(dict_1)))
    printer('Second horizont length:                   {}'.format(len(dict_2)))
    printer('Mean value/std of error:                  {:8.7} / {:8.7}' \
            .format(np.mean(differences), np.std(differences)))
    printer('Number in 5 ms window:                    {}' \
            .format(np.sum(np.array(differences) <= 5/sample_rate)))
    printer('Rate in 5 ms window:                      {:8.7}' \
            .format(np.sum(np.array(differences) <= 5/sample_rate) / len(differences)))

    printer('Average height of FIRST horizont:         {:8.7}'.format(np.mean(vals_1)))
    printer('Average height of SECOND horizont:        {:8.7}'.format(np.mean(vals_2)))

    printer('In the FIRST, but not in the SECOND:      {}'.format(not_present_1))
    printer('In the SECOND, but not in the FIRST:      {}'.format(not_present_2))
    printer('\n\n')

    if plot:
        plt.title('Distribution of errors', fontdict={'fontsize': 15})
        _ = plt.hist(differences, bins=100)



@njit
def count_nonfill(array):
    """ Jit-accelerated function to count non-fill elements. """
    count = 0
    for i in array:
        if i != FILL_VALUE:
            count += 1
    return count


@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape, order):
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

        crop = np.transpose(array_crops[i], order)
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



@njit
def update_minmax(array, val_min, val_max, matrix, il, xl, ilines_offset, xlines_offset):
    """ Get both min and max values in just one pass through array.
    Simultaneously updates (inplace) matrix if the trace is filled with zeros.
    """
    maximum = array[0]
    minimum = array[0]
    for i in array[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i

    if (minimum == 0) and (maximum == 0):
        matrix[il - ilines_offset, xl - xlines_offset] = 1

    if minimum < val_min:
        val_min = minimum
    if maximum > val_max:
        val_max = maximum

    return val_min, val_max, matrix



def horizon_to_depth_map(labels, geom, horizon_idx=0, offset=0):
    """ Converts labels-dictionary to matrix of depths.

    Parameters
    ----------
    labels : dict
        Labeled horizon.
    horizon_idx : int
        Index of item inside `labels` values.
    offset : number
        Value to add to each entry in matrix.
    """
    @njit
    def _horizon_to_depth_map(labels, i_offset, x_offset, i_len, x_len, horizon_idx=0, offset=0):
        depth_map = np.full((i_len, x_len), FILL_VALUE_A)

        for il in range(i_len):
            for xl in range(x_len):
                key = (il+i_offset, xl+x_offset)
                value = labels.get(key)
                if value is not None:
                    h = value[horizon_idx]
                    if h != FILL_VALUE:
                        h += int(np.rint(offset))
                        depth_map[il, xl] = h
        return depth_map

    return _horizon_to_depth_map(labels, geom.ilines_offset, geom.xlines_offset,
                                 geom.ilines_len, geom.xlines_len,
                                 horizon_idx, offset)


def depth_map_to_labels(depth_map, geom, labels=None, horizon_idx=0):
    """ Converts matrix of depths back into dictionary.
    Can also be used to replace dictionary values with updated ones.

    Parameters
    ----------
    depth_map : array
        Matrix of depths.
    labels : dict, optional
        If None, then new numba dictionary is created.
        If dict, then values are written into that dict.
    horizon_idx : int
        Index of value to replace in passed `labels`.
    """
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    max_count = len(list(labels.values())[0]) if labels else 1

    horizon_idx = horizon_idx if labels else 0
    labels = labels or Dict.empty(key_type, value_type)

    @njit
    def _depth_map_to_labels(depth_map, i_offset, x_offset, labels, horizon_idx, max_count):
        i_len, x_len = depth_map.shape

        for il in range(i_len):
            for xl in range(x_len):
                key = (il+i_offset, xl+x_offset)
                value = depth_map[il, xl]
                if labels.get(key) is None:
                    labels[key] = np.full((max_count, ), FILL_VALUE, np.int64)
                labels[key][horizon_idx] = value
        return labels

    return _depth_map_to_labels(depth_map, geom.ilines_offset, geom.xlines_offset, labels, horizon_idx, max_count)



def get_horizon_amplitudes(labels, geom, horizon_idx=0, window=3, offset=0, scale=False):
    """ Get values from the cube along the horizon.

    Parameters
    ----------
    labels : dict
        Labeled horizon.
    horizon_idx : int
        Index of item inside `labels` values.
    window : int
        Width of data to cut.
    offset : int
        Value to add to each entry in matrix.
    scale : bool, callable
        If bool, then values are scaled to [0, 1] range.
        If callable, then it is applied to iline-oriented slices of data from the cube.
    """
    low = window // 2
    high = max(window - low, 0)

    h5py_cube = geom.h5py_file['cube']
    i_offset, x_offset = geom.ilines_offset, geom.xlines_offset
    i_len, x_len = geom.ilines_len, geom.xlines_len
    scale_val = (geom.value_max - geom.value_min)

    background = np.zeros((i_len, x_len, window))
    depth_map = np.full((geom.ilines_len, geom.xlines_len), FILL_VALUE_A)

    @njit
    def _update(background, depth_map, slide, labels,
                xlines_len, il, ilines_offset, xlines_offset,
                low, high, horizon_idx, offset):
        for xl in range(xlines_len):
            key = (il+ilines_offset, xl+xlines_offset)
            arr = labels.get(key)
            if arr is not None:
                h = arr[horizon_idx]
                if h != FILL_VALUE:
                    h += int(np.rint(offset))
                    depth_map[il, xl] = h
                value = slide[xl, h-low:h+high]
                background[il, xl, :] = value
        return background, depth_map

    for il in range(i_len):
        slide = h5py_cube[il, :, :]

        if callable(scale):
            slide = scale(slide)
        elif scale is True:
            slide -= geom.value_min
            slide *= (1 / scale_val)
        # here we can insert even more transforms!
        background, depth_map = _update(background, depth_map, slide, labels,
                                        x_len, il, i_offset, x_offset,
                                        low, high, horizon_idx, offset)
    background = np.squeeze(background)
    return background, depth_map


@njit
def compute_corrs(data):
    """ Compute average correlation between each column in data and nearest traces. """
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for i in range(i_range):
        for x in range(x_range):
            trace = data[i, x, :]
            if np.max(trace) == np.min(trace):
                continue

            s, c = 0.0, 0

            if i > 0:
                trace_1 = data[i-1, x, :]
                if not np.max(trace_1) == np.min(trace_1):
                    s += np.corrcoef(trace, trace_1)[0, 1]
                    c += 1
            if i < i_range-1:
                trace_2 = data[i+1, x, :]
                if not np.max(trace_2) == np.min(trace_2):
                    s += np.corrcoef(trace, trace_2)[0, 1]
                    c += 1
            if x > 0:
                trace_3 = data[i, x-1, :]
                if not np.max(trace_3) == np.min(trace_3):
                    s += np.corrcoef(trace, trace_3)[0, 1]
                    c += 1
            if x < x_range-1:
                trace_4 = data[i, x+1, :]
                if not np.max(trace_4) == np.min(trace_4):
                    s += np.corrcoef(trace, trace_4)[0, 1]
                    c += 1

            if c != 0:
                corrs[i, x] = s / c
    return corrs
