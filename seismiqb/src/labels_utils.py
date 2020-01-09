""" Helper functions to load labels in different formats from txt-like files. """
import numpy as np
import pandas as pd

from numba import njit, types
from numba.typed import Dict

from ._const import FILL_VALUE



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
    point_cloud = np.rint(point_cloud).astype(np.int64)

    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    labels = Dict.empty(key_type, value_type)
    labels = _fill_labels(labels, point_cloud, point_cloud[-1, -1] + 1)
    return labels

@njit
def _fill_labels(labels, point_cloud, n_horizons):
    """ Fill in labels-dict.
    Every value in the dictionary is array of the fixed (number of horizons) length, containing depth of each surface.
    """
    for i in range(len(point_cloud)):
        il, xl = point_cloud[i, :2]
        if labels.get((il, xl)) is None:
            labels[(il, xl)] = np.full((n_horizons, ), FILL_VALUE, np.int64)

        idx = int(point_cloud[i, 3])
        labels[(il, xl)][idx] = point_cloud[i, 2]
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
    counts = _fill_counts_f(counts, point_cloud)

    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.Tuple((types.int64[:], types.int64[:], types.int64[:]))
    labels = Dict.empty(key_type, value_type)
    labels = _fill_labels_f(labels, counts, point_cloud)
    return labels

@njit
def _fill_counts_f(counts, point_cloud):
    """ Count number of facies on each trace. """
    for i in range(point_cloud.shape[0]):
        key = (point_cloud[i, 0], point_cloud[i, 1])

        if counts.get(key) is None:
            counts[key] = 1
        else:
            counts[key] += 1
    return counts

def _fill_labels_f(labels, counts, point_cloud):
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
