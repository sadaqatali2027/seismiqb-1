""" Functions to work with horizons:
    * Converting between formats (dictionary, depth-map, mask)
    * Saving to txt file
    * Comparing horizons and evaluating metrics
"""
#pylint: disable=too-many-lines
import os
from copy import copy
from functools import wraps
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import njit
from skimage.measure import label, regionprops
from scipy.signal import hilbert, medfilt

from ..batchflow import HistoSampler

from ._const import FILL_VALUE, FILL_VALUE_MAP
from .geometry import SeismicGeometry
from .utils import compute_running_mean, lru_cache
from .plot_utils import plot_image




class BaseLabel(ABC):
    """ Base class for labels. """
    @abstractmethod
    def fromfile(self, path):
        """ Labels must be loadable from file on disk. """

    @abstractmethod
    def frompoints(self, array):
        """ Labels must be loadable from array of needed format. """

    @abstractmethod
    def add_to_mask(self, mask, mask_bbox, **kwargs):
        """ Labels must be able to create masks for training. """



def synchronizable(base='matrix'):
    """ Marks function as one that can be synchronized.
    Under the hood, uses `synchronize` method with desired `base` before and/or after the function call.
    If the horizon is already in sync ('synchronized' attribute evaluates to True), no sync is performed.

    Parameters
    ----------
    base : 'matrix' or 'array'
        Base of synchronization: if 'matrix', then 'array' is synchronized, and vice versa.
    before : bool
        Whether to synchronize before the function call.
    after : bool
        Whether to synchronize after the function call.
    """
    def _synchronizable(func):
        @wraps(func)
        def wrapped(self, *args, before=True, after=True, **kwargs):
            if before and not self.synchronized:
                self.synchronize(base)
                self.synchronized = True
                if self.debug:
                    self.debug_update()

            result = func(self, *args, **kwargs)

            if after and not self.synchronized:
                self.synchronize(base)
                self.synchronized = True
                if self.debug:
                    self.debug_update()
            return result
        return wrapped
    return _synchronizable



class Horizon(BaseLabel):
    """ Finally, we've made a separate class for horizon storing..
    """
    #pylint: disable=too-many-public-methods

    # CHARISMA: default seismic format of storing surfaces inside the volume
    CHARISMA_SPEC = ['INLINE', '_', 'iline', 'XLINE', '_', 'xline', 'cdp_x', 'cdp_y', 'height']

    # REDUCED_CHARISMA: CHARISMA without redundant columns and spaces
    REDUCED_CHARISMA_SPEC = ['iline', 'xline', 'height']

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999


    def __init__(self, storage, geometry=None, debug=False, debug_attrs=None, **kwargs):
        # Meta information
        self.path = None
        self.name = None
        self.format = None

        # Main storage and coordinates of local system inside the cubic one
        self.bbox = None
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.matrix = None

        # Additional way of storing the underlying data
        self.points = None

        # Heights information
        self.h_min, self.h_max = None, None
        self.h_mean, self.h_std = None, None

        # Attributes from geometry
        if geometry:
            if isinstance(geometry, str):
                geometry = SeismicGeometry(geometry)
                geometry.load()
            self.geometry = geometry
            self.cube_name = geometry.name
            self.cube_shape = geometry.cube_shape

        self.sampler = None
        # Store copies of attributes
        self.debug = debug
        self.debug_attrs = debug_attrs or ['matrix', 'points']

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, dict):
            # mapping from (iline, xline) to (height)
            self.format = 'dict'

        elif isinstance(storage, np.ndarray):
            if storage.ndim == 2 and storage.shape[1] == 3:
                # array with row in (iline, xline, height) format
                self.format = 'points'

            elif storage.ndim == 2 and (storage.shape == self.cube_shape[:-1]).all():
                # matrix of (iline, xline) shape with every value being height
                self.format = 'fullmatrix'

            elif storage.ndim == 2:
                # matrix of (iline, xline) shape with every value being height
                self.format = 'matrix'

            elif storage.ndim >= 3:
                # model prediction: 3+ dimensional mask with horizon surface
                self.format = 'mask'

        getattr(self, 'from{}'.format(self.format))(storage, **kwargs)
        self.synchronized = True


    # Debug methods
    def debug_update(self):
        """ Create copies of all 'debug_attributes' to use for later debug. """
        for attr in self.debug_attrs:
            setattr(self, 'copy_{}'.format(attr), copy(getattr(self, attr)))

    def debug_check(self):
        """ Assert that current storages are the same as their saved copies. """
        for attr in self.debug_attrs:
            assert (getattr(self, attr) == getattr(self, 'copy_{}'.format(attr))).all(), \
            "Attribute `{}` is not the same as it's saved copy!".format(attr)


    # Coordinate transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] = self.geometry.ilines_transform(array[:, 0])
        array[:, 1] = self.geometry.xlines_transform(array[:, 1])
        array[:, 2] = self.geometry.height_transform(array[:, 2])
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array[:, 0] = self.geometry.ilines_reverse(array[:, 0])
        array[:, 1] = self.geometry.xlines_reverse(array[:, 1])
        array[:, 2] = self.geometry.height_reverse(array[:, 2])
        return array


    # Initialization from different containers
    def frompoints(self, points, transform=False, **kwargs):
        """ Base initialization: from point cloud array of (N, 3) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: two spatial coordinates and depth.
        transform : bool
            Whether transform from line coordinates (ilines, xlines) to cubic system.
        """
        _ = kwargs

        # Transform to cubic coordinates, if needed
        if transform:
            points = self.lines_to_cubic(points)
        self.points = np.rint(points).astype(np.int32)

        # Collect stats on separate axes
        self.i_min, self.x_min, self.h_min = np.min(self.points, axis=0).astype(np.int32)
        self.i_max, self.x_max, self.h_max = np.max(self.points, axis=0).astype(np.int32)

        self.h_mean = np.mean(self.points[:, -1])
        self.h_std = np.std(self.points[:, -1])

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max]],
                             dtype=np.int32)

        # Convert array of (N, 3) shape to depth map (matrix)
        copy_points = np.copy(self.points)
        copy_points[:, 0] -= self.i_min
        copy_points[:, 1] -= self.x_min
        matrix = np.full((self.i_length, self.x_length), self.FILL_VALUE, np.int32)

        matrix[copy_points[:, 0], copy_points[:, 1]] = copy_points[:, 2]
        self.matrix = matrix

        if self.debug:
            self.debug_update()


    def fromfile(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file. """
        _ = kwargs

        self.path = path
        self.name = os.path.basename(path)
        points = self.file_to_points(path)
        self.frompoints(points, transform)

    @lru_cache(1024, storage=os.environ.get('SEISMIQB_HORIZON_CACHEDIR'), anchor=True)
    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path) as file:
            line_len = len(file.readline().split(' '))
        if line_len == 3:
            names = Horizon.REDUCED_CHARISMA_SPEC
        elif line_len >= 9:
            names = Horizon.CHARISMA_SPEC
        else:
            raise ValueError('Horizon labels must be in CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep='\s+', names=names, usecols=Horizon.COLUMNS)
        df.sort_values(Horizon.COLUMNS, inplace=True)
        return df.values


    def frommatrix(self, matrix, i_min, x_min, transform=False, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        points = self.matrix_to_points(matrix)
        points[:, 0] += i_min
        points[:, 1] += x_min
        self.frompoints(points, transform=transform)


    def fromfullmatrix(self, matrix, transform=False, **kwargs):
        """ Init from matrix that covers the whole cube. """
        _ = kwargs

        points = self.matrix_to_points(matrix)
        self.frompoints(points, transform=transform)

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.asarray(matrix != Horizon.FILL_VALUE).nonzero()
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


    def frommask(self, array, order=None, **kwargs):
        """ Init from mask with drawn surface. """
        _ = kwargs, array, order


    def fromdict(self, dictionary, transform=True, **kwargs):
        """ Init from mapping from (iline, xline) to depths. """
        _ = kwargs, dictionary, transform

        points = self.dict_to_points(dictionary)
        self.frompoints(points, transform=transform)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    # Functions to use to change the horizon
    def synchronize(self, base='matrix'):
        """ Synchronize the whole horizon instance with base storage. """
        if not self.synchronized:
            if base == 'matrix':
                self.frommatrix(self.matrix, self.i_min, self.x_min)
            elif base == 'points':
                self.frompoints(self.points)
            self.synchronized = True


    @synchronizable('matrix')
    def apply_to_matrix(self, function, **kwargs):
        """ Apply passed function to matrix storage. Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to matrix storage directly.
            Can return either new_matrix, new_i_min, new_x_min or new_matrix only.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        result = function(self.matrix, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            matrix, i_min, x_min = result
        else:
            matrix, i_min, x_min = result, self.i_min, self.x_min
        self.matrix, self.i_min, self.x_min = matrix, i_min, x_min

        self.synchronized = False


    @synchronizable('points')
    def apply_to_points(self, function, **kwargs):
        """ Apply passed function to points storage. Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to points storage directly.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        self.points = function(self.points, **kwargs)
        self.synchronized = False


    def filter_points(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from points storage."""
        filtering_matrix = filtering_matrix or self.geometry.zero_traces

        def filtering_function(points, **kwds):
            _ = kwds
            @njit
            def _filtering_function(points, filtering_matrix):
                #pylint: disable=consider-using-enumerate
                mask = np.ones(len(points), dtype=np.int32)

                for i in range(len(points)):
                    il, xl = points[i, 0], points[i, 1]
                    if filtering_matrix[il, xl] == 1:
                        mask[i] = 0
                return points[mask == 1, :]
            return _filtering_function(points, filtering_matrix)

        self.apply_to_points(filtering_function, before=False, after=True, **kwargs)


    def filter_matrix(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from matrix storage."""
        filtering_matrix = filtering_matrix or self.geometry.zero_traces
        idx_i, idx_x = np.asarray(filtering_matrix[self.i_min:self.i_max + 1,
                                                   self.x_min:self.x_max + 1] == 1).nonzero()
        def filtering_function(matrix, **kwds):
            _ = kwds
            matrix[idx_i, idx_x] = self.FILL_VALUE
            return matrix

        self.apply_to_matrix(filtering_function, before=False, after=True, **kwargs)


    # Horizon usage: point/mask generation
    def create_sampler(self, bins=None, **kwargs):
        """ Create sampler based on horizon location.

        Parameters
        ----------
        bins : sequence
            Size of ticks alongs each respective axis.
        """
        _ = kwargs
        default_bins = self.cube_shape // np.array([5, 20, 20])
        bins = bins or default_bins
        self.sampler = HistoSampler(np.histogramdd(self.points/self.cube_shape, bins=bins))


    def add_to_mask(self, mask, mask_bbox, width):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        mask_bbox : ndarray
            Bounding box of a mask: minimum/maximum iline, xline and depth.
        width : int
            Width of an added horizon.
        """
        low = width // 2
        high = max(width - low, 0)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max, mask_x_max)
        overlap = self.matrix[i_min - self.i_min:i_max - self.i_min,
                              x_min - self.x_min:x_max - self.x_min]

        # Coordinates of points to use in overlap local system
        idx_i, idx_x = np.asarray((overlap != self.FILL_VALUE) &
                                  (overlap >= mask_h_min + low) &
                                  (overlap <= mask_h_max - high)).nonzero()
        heights = overlap[idx_i, idx_x]

        # Convert coordinates to mask local system
        idx_i += i_min - mask_i_min
        idx_x += x_min - mask_x_min
        heights -= (mask_h_min + low)

        for _ in range(width):
            mask[idx_i, idx_x, heights] = 1
            heights += 1
        return mask


    # Horizon evaluation
    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        return len(self) / (np.prod(self.matrix.shape) - np.sum(self.geometry.zero_traces))



    @lru_cache(2)
    def metric(self, **kwargs):
        """ Doc. """

    def compare_to(self, other):
        """ Compare two horizons on simple numbers. """
        if not isinstance(other, Horizon):
            raise TypeError('One can compare horizon only to another horizon. ')

        results_dict = {
            'length_self': len(self),
            'length_other': len(other),
            'mean_self': self.h_mean,
            'mean_other': other.h_mean,
        }
        return results_dict

    def common_background(self, other):
        """ Doc. """
        common_i_min, common_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        common_x_min, common_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((common_i_max - common_i_min + 1, common_x_max - common_x_min + 1), dtype=np.int32)
        merge_dict = {
            'background': background,
            'counts': np.copy(background),
            'i_min': common_i_min, 'i_max': common_i_max,
            'x_min': common_x_min, 'x_max': common_x_max,
        }
        return merge_dict

    def put_on_background(self, merge_dict):
        """ Doc. """
        i_start, x_start = self.i_min - merge_dict['i_min'], self.x_min - merge_dict['x_min']
        i_end, x_end = i_start + self.i_length, x_start + self.x_length

        background = merge_dict['background']
        background[i_start:i_end, x_start:x_end] = self.matrix

        counts = merge_dict['counts']
        counts[i_start:i_end, x_start:x_end] += (self.matrix != self.FILL_VALUE).astype(np.int32)
        return merge_dict



    def __getitem__(self, point):
        """ Get value in point in cubic coordinates. """
        point = point[0] - self.i_min, point[1] - self.x_min
        return self.matrix[point]

    def __len__(self):
        """ Number of labeled traces. """
        return len(np.asarray(self.matrix != self.FILL_VALUE).nonzero()[0])


    def dump_horizon(self, path, add_height=True):
        """ Save horizon points on disc.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        self.synchronize()
        values = self.cubic_to_lines(copy(self.points))

        df = pd.DataFrame(values, columns=self.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)

        path = path if not add_height else '{}_#{}'.format(path, self.h_mean)
        df.to_csv(path, sep=' ', columns=self.COLUMNS,
                  index=False, header=False)


    # Methods of (visual) representation of a horizon
    def __repr__(self):
        return f"""<horizon {self.name} for {self.cube_name} at {hex(id(self))}>"""

    def __str__(self):
        return f"""Horizon {self.name} for {self.cube_name} loaded from {self.format}
Ilines from {self.i_min} to {self.i_max}
Xlines from {self.x_min} to {self.x_max}
Heights from {self.h_min} to {self.h_max}, mean is {self.h_mean:5.5}, std is {self.h_std:4.4}
Currently synchronized: {self.synchronized}; In debug mode: {self.debug}; At {hex(id(self))}
"""


    def show(self, **kwargs):
        """ Nice visualization of a horizon depth map. """
        copy_matrix = copy(self.matrix).astype(np.float32)
        copy_matrix[copy_matrix == self.FILL_VALUE] = np.nan
        plot_image(copy_matrix, 'Depth map of {} on {}'.format(self.name, self.cube_name),
                   cmap='viridis_r', **kwargs)

    def show_on_full(self, **kwargs):
        """ Nice visualization of a horizon depth map with respect to the whole cube. """
        background = np.full(self.cube_shape[:-1], self.FILL_VALUE, dtype=np.float32)
        background[self.i_min:self.i_max+1, self.x_min:self.x_max+1] = self.matrix
        background[background == self.FILL_VALUE] = np.nan
        plot_image(background, 'Depth map of {} on {}'.format(self.name, self.cube_name),
                   cmap='viridis_r', **kwargs)






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





def get_horizon_amplitudes(labels, geom, horizon_idx=0, window=3, offset=0, scale=False, chunk_size=512):
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
    #pylint: disable=function-redefined
    low = window // 2
    high = max(window - low, 0)

    h5py_cube = geom.h5py_file['cube_h']
    i_offset, x_offset, depth = geom.ilines_offset, geom.xlines_offset, geom.depth
    i_len, x_len = geom.ilines_len, geom.xlines_len
    scale_val = (geom.value_max - geom.value_min)

    if callable(scale):
        pass
    elif scale is True:
        def scale(array):
            array -= geom.value_min
            array *= (1 / scale_val)
            return array
    elif scale is False:
        def scale(array):
            return array

    horizon_min, horizon_max = _find_min_max(labels, horizon_idx)
    chunk_size = min(chunk_size, horizon_max - horizon_min + window)

    background = np.zeros((i_len, x_len, window))
    depth_map = np.full((geom.ilines_len, geom.xlines_len), FILL_VALUE_MAP)

    for h_start in range(horizon_min - low, horizon_max + high, chunk_size):
        h_end = min(h_start + chunk_size, horizon_max + high, depth)
        data_chunk = h5py_cube[h_start:h_end, :, :]
        data_chunk = scale(data_chunk)

        background, depth_map = _get_horizon_amplitudes(background, depth_map, data_chunk, labels, horizon_idx,
                                                        i_offset, x_offset, depth, low, high, window,
                                                        h_start, h_end, chunk_size, offset)

    background = np.squeeze(background)
    return background, depth_map

@njit
def _find_min_max(labels, horizon_idx):
    """ Jit-accelerated function of finding minimum and maximum of horizon depth inside labels dictionary. """
    min_, max_ = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    for value in labels.values():
        h = value[horizon_idx]
        if h != FILL_VALUE:
            if h > max_:
                max_ = h
            if h < min_:
                min_ = h
    return min_, max_

@njit
def _get_horizon_amplitudes(background, depth_map, data, labels, horizon_idx, ilines_offset, xlines_offset,
                            depth, low, high, window, h_start, h_end, chunk_size, offset):
    """ Jit-accelerated function of cutting window of amplitudes along the horizon. """
    for key, value in labels.items():
        h = value[horizon_idx]
        if h != FILL_VALUE:
            h += offset
        h_low, h_high = h - low, h + high

        if h_high < depth:
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



def get_line_horizon_amplitudes(labels, geom, horizon_idx=0, orientation='i', line=None, window=3,
                                offset=0, scale=False):
    """ Amazing! """
    #pylint: disable=function-redefined, too-many-branches
    low = window // 2
    high = max(window - low, 0)

    i_offset, x_offset = geom.ilines_offset, geom.xlines_offset
    i_len, x_len = geom.ilines_len, geom.xlines_len
    scale_val = (geom.value_max - geom.value_min)

    if orientation.startswith('i'):
        h5py_cube = geom.h5py_file['cube']
        slide_transform = lambda array: array
        w_shape = (1, x_len, window)
        filtering_matrix = geom.zero_traces[line, :].reshape(w_shape[:2])

        depth_map = np.full((x_len,), FILL_VALUE_MAP)
        for xl in range(x_len):
            key = line + i_offset, xl + x_offset
            value = labels.get(key)
            if value is not None:
                h = value[horizon_idx]
                if h != FILL_VALUE:
                    h += int(np.rint(offset))
                    depth_map[xl] = h

    elif orientation.startswith('x'):
        h5py_cube = geom.h5py_file['cube_x']
        slide_transform = lambda array: array.T
        w_shape = (i_len, 1, window)
        filtering_matrix = geom.zero_traces[:, line].reshape(w_shape[:2])

        depth_map = np.full((i_len,), FILL_VALUE_MAP)
        for il in range(i_len):
            key = il + i_offset, line + x_offset
            value = labels.get(key)
            if value is not None:
                h = value[horizon_idx]
                if h != FILL_VALUE:
                    h += int(np.rint(offset))
                    depth_map[il] = h

    if callable(scale):
        pass
    elif scale is True:
        def scale(array):
            array -= geom.value_min
            array *= (1 / scale_val)
            return array
    elif scale is False:
        def scale(array):
            return array

    slide = h5py_cube[line, :, :]
    slide = slide_transform(slide)
    slide = scale(slide)

    background = np.zeros((len(depth_map), window))
    for i, h in enumerate(depth_map):
        if h != FILL_VALUE_MAP:
            background[i, :] = slide[i, h-low:h+high]
    background = background.reshape(w_shape)
    return slide, background, depth_map, filtering_matrix



def compute_local_corrs(data, zero_traces, locality=4, **kwargs):
    """ Compute average correlation between each column in data and nearest traces.

    Parameters
    ----------
    data : ndarray
        Amplitudes along the horizon of shape (n_ilines, n_xlines, window).
    zero_traces : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions to remove.
    locality : {4, 8}
        Defines number of nearest traces to average correlations from.

    Returns
    -------
    array-like
        Matrix of (n_ilines, n_xlines) shape with computed metric for each point.
    """
    _ = kwargs

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
                    loc = locs[i]
                    il_, xl_ = il + loc[0], xl + loc[1]

                    if (0 <= il_ < i_range) and (0 <= xl_ < x_range):
                        if bad_traces[il_, xl_] == 0:
                            trace_ = data[il_, xl_, :]
                            s += np.corrcoef(trace, trace_)[0, 1]
                            c += 1
                if c != 0:
                    corrs[il, xl] = s / c
    return corrs


def compute_support_corrs(data, zero_traces, supports=1, safe_strip=0, line_no=None, **kwargs):
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
        If str, then must define either `iline` or `xline` mode. In each respective one, iline/xline given by
        `line_no` argument is used to generate supports.
    safe_strip : int
        Used only for `int` mode of `supports` parameter and defines minimum distance from borders for sampled points.
    line_no : int
        Used only for `str` mode of `supports` parameter to define exact iline/xline to use.

    Returns
    -------
    array-like
        Matrix of either (n_ilines, n_xlines, n_supports) or (n_ilines, n_xlines) shape with
        computed metric for each point.
    """
    _ = kwargs

    bad_traces = np.copy(zero_traces)
    bad_traces[np.std(data, axis=-1) == 0.0] = 1

    if isinstance(supports, (int, tuple, list, np.ndarray)):
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

        return _compute_support_corrs_np(data, supports, bad_traces)

    if isinstance(supports, str):
        if supports.startswith('i'):
            support_il = line_no or data.shape[0] // 2
            return _compute_line_corrs_np(data, bad_traces, support_il=support_il)

        if supports.startswith('x'):
            support_xl = line_no or data.shape[1] // 2
            return _compute_line_corrs_np(data, bad_traces, support_xl=support_xl)
    raise ValueError('`Supports` must be either int, sequence, ndarray or string. ')

def _compute_support_corrs_np(data, supports, bad_traces):
    """ NumPy function to compute correlations with a number of support traces. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    data_n = data - np.mean(data, axis=-1, keepdims=True)
    data_stds = np.std(data, axis=-1)
    bad_traces[data_stds == 0] = 1

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]
    support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
    support_stds = np.std(support_traces, axis=-1)

    corrs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        cov = np.sum(support_n[i] * data_n, axis=-1) / depth
        temp = cov / (support_stds[i] * data_stds)
        temp[bad_traces == 1] = 0
        corrs[:, :, i] = temp
    return corrs

@njit
def _compute_support_corrs_numba(data, supports, bad_traces):
    """ Jit-accelerated function to compute correlations with a number of support traces.
    Will become faster after keyword support implementation in Numba.
    """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    corrs = np.zeros((i_range, x_range, n_supports))
    for il in range(i_range):
        for xl in range(x_range):
            if bad_traces[il, xl] == 0:
                trace = data[il, xl, :]

                for i in range(n_supports):
                    corrs[il, xl, i] = np.corrcoef(trace, support_traces[i, :])[0, 1]
    return corrs

def _compute_line_corrs_np(data, bad_traces, support_il=None, support_xl=None):
    depth = data.shape[-1]

    data_n = data - np.mean(data, axis=-1, keepdims=True)
    data_stds = np.std(data, axis=-1)
    bad_traces[data_stds == 0] = 1

    if support_il is not None and support_xl is not None:
        raise ValueError('Use `compute_support_corrs` for given trace. ')

    if support_il is not None:
        support_traces = data[[support_il], :, :]
        support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
        support_stds = np.std(support_traces, axis=-1)
        bad_traces[:, support_stds[0, :] == 0] = 1
    if support_xl is not None:
        support_traces = data[:, [support_xl], :]
        support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
        support_stds = np.std(support_traces, axis=-1)
        bad_traces[support_stds[:, 0] == 0, :] = 1

    cov = np.sum(support_n * data_n, axis=-1) / depth
    corrs = cov / (support_stds * data_stds)
    corrs[bad_traces == 1] = 0
    return corrs

@njit
def _compute_iline_corrs_numba(data, support_il, bad_traces):
    """ Jit-accelerated function to compute correlations along given iline. """
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
def _compute_xline_corrs_numba(data, support_xl, bad_traces):
    """ Jit-accelerated function to compute correlations along given xline. """
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


def compute_hilbert(data, depth_map, mode='median', kernel_size=3, eps=1e-5, **kwargs):
    """ Compute phase along the horizon. """
    mode = kwargs.get('hilbert_mode') or mode

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
