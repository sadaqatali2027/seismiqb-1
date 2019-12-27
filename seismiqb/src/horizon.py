""" Functions to work with horizons:
    * Converting between formats (dictionary, depth-map, mask)
    * Saving to txt file
    * Comparing horizons and evaluating metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import njit, types
from numba.typed import Dict
from skimage.measure import label, regionprops
from scipy.signal import hilbert, medfilt

from ._const import FILL_VALUE, FILL_VALUE_MAP
from .utils import compute_running_mean



def mask_to_horizon(mask, threshold, averaging, transforms, separate=False):
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



def convert_to_numba_dict(labels):
    """ Convert a dict to a Numba-typed dict.

    Parameters
    ----------
    labels : dict
        Dictionary with a special format:
            Keys must be tuples of length 2 with int64 values;
            Values must be arrays.

    Returns
    -------
    A Numba-typed dict.
    """
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    type_labels = Dict.empty(key_type, value_type)
    for key, value in labels.items():
        type_labels[key] = np.asarray(np.rint(value), dtype=np.int64)
    return type_labels



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
        depth_map = np.full((i_len, x_len), FILL_VALUE_MAP)

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



def get_horizon_amplitudes(labels, geom, horizon_idx=0, window=3, offset=0, scale=False, chunk_size=128):
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
    chunk_size : int
        Size of data along height axis processed at a time.
    """
    low = window // 2
    high = max(window - low, 0)

    h5py_cube = geom.h5py_file['cube_h']
    i_offset, x_offset = geom.ilines_offset, geom.xlines_offset
    i_len, x_len = geom.ilines_len, geom.xlines_len
    scale_val = (geom.value_max - geom.value_min)


    horizon_min, horizon_max = _find_min_max(labels, horizon_idx)
    chunk_size = min(chunk_size, horizon_max - horizon_min + window)

    background = np.zeros((i_len, x_len, window))
    depth_map = np.full((geom.ilines_len, geom.xlines_len), FILL_VALUE_MAP)

    for h_start in range(horizon_min - low, horizon_max + high, chunk_size):
        h_end = min(h_start + chunk_size, horizon_max + high)
        data_chunk = h5py_cube[h_start:h_end, :, :]

        if callable(scale):
            data_chunk = scale(data_chunk)
        elif scale is True:
            data_chunk -= geom.value_min
            data_chunk *= (1 / scale_val)

        background, depth_map = _update(background, depth_map, data_chunk, labels, horizon_idx, i_offset, x_offset,
                                        low, high, window, h_start, h_end, chunk_size, offset)

    background = np.squeeze(background)
    return background, depth_map

@njit
def _find_min_max(labels, horizon_idx):
    """ Fast way of finding minimum and maximum of horizon depth inside labels dictionary. """
    min_, max_ = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    for value in labels.values():
        h = value[horizon_idx]
        if h > max_:
            max_ = h
        if h < min_:
            min_ = h
    return min_, max_

@njit
def _update(background, depth_map, data, labels, horizon_idx, ilines_offset, xlines_offset,
            low, high, window, h_start, h_end, chunk_size, offset):
    """ Jit-accelerated function of cutting window of amplitudes along the horizon. """
    for key, value in labels.items():
        h = value[horizon_idx]
        if h != FILL_VALUE:
            h += offset
        h_low, h_high = h - low, h + high

        if h_start <= h_low < h_high < h_end: # window is completely inside the chunk
            il, xl = key[0] - ilines_offset, key[1] - xlines_offset
            idx_start = h_low - h_start
            idx_end = h_high - h_start
            background[il, xl, :] = data[idx_start:idx_end, il, xl]
            depth_map[il, xl] = h

        elif h_start < h_low <= h_end: # window pierces the chunk from below
            il, xl = key[0] - ilines_offset, key[1] - xlines_offset
            idx_start = h_low - h_start
            background[il, xl, 0:(chunk_size - idx_start)] = data[idx_start:min(chunk_size, idx_start+window),
                                                                  il, xl]
            depth_map[il, xl] = h

        elif h_start <= h_high < h_end: # window pierces the chunk from above
            il, xl = key[0] - ilines_offset, key[1] - xlines_offset
            idx_end = h_high - h_start
            if idx_end != 0:
                background[il, xl, -idx_end:] = data[max(0, idx_end-window):idx_end, il, xl]
            else:
                background[il, xl, 0] = data[0, il, xl]
            depth_map[il, xl] = h
    return background, depth_map



def compute_local_corrs(data, zero_traces, locality=4):
    """ Compute average correlation between each column in data and nearest traces.

    Parameters
    ----------
    data : ndarray
        Amplitudes along the horizon of shape (n_ilines, n_xlines, window).
    zero_traces : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions to remove.
    locality : {4, 8}
        Defines number of nearest traces to average correlations from.
    """
    if locality == 4:
        locs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    elif locality == 8:
        locs = [[-1, -1], [0, -1], [1, -1],
                [-1, 0], [1, 0],
                [-1, 1], [0, 1], [1, 1]]
    locs = np.array(locs)

    bad_traces = np.copy(zero_traces)
    bad_traces[np.std(data, axis=-1) == 0.0] = 1
    return _compute_local_corrs(data, bad_traces, locs)

@njit
def _compute_local_corrs(data, bad_traces, locs):
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for il in range(i_range):
        for xl in range(x_range):
            if bad_traces[il, xl] == 0:
                trace = data[il, xl, :]

                s, c = 0.0, 0
                for i in range(len(locs)):
                    il_, xl_ = il + locs[i][0], xl + locs[i][1]

                    if (0 < il_ < i_range) and (0 < xl_ < x_range):
                        if bad_traces[il_, xl_] == 0:
                            trace_ = data[il_, xl_, :]
                            s += np.corrcoef(trace, trace_)[0, 1]
                            c += 1
                if c != 0:
                    corrs[il, xl] = s / c
    return corrs


def compute_support_corrs(data, supports, zero_traces, safe_strip=0, line_no=None):
    """ Compute correlations with support traces.

    Parameters
    ----------
    data : ndarray
        Amplitudes along the horizon of shape (n_ilines, n_xlines, window).
    zero_traces : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions to remove.
    supports : int, sequence, ndarray or str
        Defines mode of generating support traces.
        If int, then that number of random non-zero traces positions are generated.
        If sequence or ndarray, then must be of shape (N, 2) and is used as positions of support traces.
        If str, then must defines either `iline` or `xline` mode. In each respective one, given iline/xline is used
        to generate supports.
    safe_strip : int
        Used only for `int` mode of `supports` parameter and defines minimum distance from borders for sampled points.
    line_no : int
        Used only for `str` mode of `supports` parameter to define exact iline/xline to use.
    """
    bad_traces = np.copy(zero_traces)
    bad_traces[np.std(data, axis=-1) == 0.0] = 1

    if isinstance(supports, (int, tuple, list)):
        if isinstance(supports, int):
            if safe_strip:
                bad_traces[:, :safe_strip], bad_traces[:, -safe_strip:] = 1, 1
                bad_traces[:safe_strip, :], bad_traces[-safe_strip:, :] = 1, 1

            non_zero_traces = np.where(bad_traces == 0)
            indices = np.random.choice(len(non_zero_traces[0]), supports)
            supports = np.array([non_zero_traces[0][indices], non_zero_traces[1][indices]]).T

        elif isinstance(supports, (tuple, list)):
            if min(len(item) == 2 for item in supports) is False:
                raise ValueError('Each of `supports` sequence must contain coordinate of trace (il, xl). ')
            supports = np.array(supports)

        return _compute_support_corrs(data, supports, bad_traces)

    if isinstance(supports, str):
        if supports.startswith('i'):
            support_il = line_no or data.shape[0] // 2
            return _compute_iline_corrs(data, support_il, bad_traces)

        if supports.startswith('x'):
            support_xl = line_no or data.shape[1] // 2
            return _compute_xline_corrs(data, support_xl, bad_traces)
    raise ValueError('`Supports` must be either int, sequence, ndarray or string. ')

@njit
def _compute_support_corrs(data, supports, bad_traces):
    """ Jit-accelerated function to compute correlations with a number of support traces. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    stds = np.zeros((n_supports,))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]
        stds[i] = np.std(support_traces[i, :])

    corrs = np.zeros((i_range, x_range, n_supports))

    for il in range(i_range):
        for xl in range(x_range):
            if bad_traces[il, xl] == 0:
                trace = data[il, xl, :]

                for i in range(n_supports):
                    corrs[il, xl, i] = np.corrcoef(trace, support_traces[i, :])[0, 1]
    return corrs

@njit
def _compute_iline_corrs(data, support_il, bad_traces):
    """ Jit-accelerated function to compute correlations along given iline"""
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for xl in range(x_range):
        if bad_traces[support_il, xl] == 0:
            support_trace = data[support_il, xl, :]

            for il in range(i_range):
                if bad_traces[il, xl] == 0:
                    trace = data[il, xl, :]
                    corrs[il, xl] = np.corrcoef(trace, support_trace)[0, 1]
    return corrs

@njit
def _compute_xline_corrs(data, support_xl, bad_traces):
    """ Jit-accelerated function to compute correlations along given xline"""
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for il in range(i_range):
        if bad_traces[il, support_xl] == 0:
            support_trace = data[il, support_xl, :]

            for xl in range(x_range):
                if bad_traces[il, xl] == 0:
                    trace = data[il, xl, :]
                    corrs[il, xl] = np.corrcoef(trace, support_trace)[0, 1]
    return corrs


def compute_hilbert(data, depth_map, mode='median', kernel_size=3, eps=1e-5):
    """ Compute phase along the horizon. """
    analytic = hilbert(data, axis=-1)
    phase = (np.angle(analytic))
    phase = phase % (2 * np.pi) - np.pi
    phase[depth_map == FILL_VALUE_MAP, :] = 0

    horizon_phase = phase[:, :, phase.shape[-1] // 2]
    horizon_phase = correct_pi(horizon_phase, eps)

    if mode == 'mean':
        median_phase = compute_running_mean(horizon_phase, kernel_size)
    else:
        median_phase = medfilt(horizon_phase, kernel_size)
    median_phase[depth_map == FILL_VALUE_MAP] = 0

    img = np.minimum(median_phase - horizon_phase, 2 * np.pi + horizon_phase - median_phase)
    img[depth_map == FILL_VALUE_MAP] = 0
    img = np.where(img < -np.pi, img + 2 * np. pi, img)

    metrics = np.zeros((*img.shape, 2+data.shape[2]))
    metrics[:, :, 0] = img
    metrics[:, :, 1] = median_phase
    metrics[:, :, 2:] = phase
    return metrics


@njit
def correct_pi(horizon_phase, eps):
    """ Jit-accelerated function to <>. """
    for i in range(horizon_phase.shape[0]):
        prev = horizon_phase[i, 0]
        for j in range(1, horizon_phase.shape[1] - 1):
            if np.abs(np.abs(prev) - np.pi) <= eps and np.abs(np.abs(horizon_phase[i, j + 1]) - np.pi) <= eps:
                horizon_phase[i, j] = prev
            prev = horizon_phase[i, j]
    return horizon_phase
