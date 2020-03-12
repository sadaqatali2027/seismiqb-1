""" Contains facies-related functions. """
import numpy as np
from numba import njit, types
from numba.typed import Dict


from .horizon import BaseLabel


FILL_VALUE = -999999



class GeoBody(BaseLabel):
    """ Class for 3D bodies inside the cube: rivers or facies. """

    def from_file(self, path):
        """ Labels must be loadable from file on disk. """

    def from_points(self, array):
        """ Labels must be loadable from array of needed format. """

    def add_to_mask(self, mask, mask_bbox, **kwargs):
        """ Labels must be able to create masks for training. """



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
