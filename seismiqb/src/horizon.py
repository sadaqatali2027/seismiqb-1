""" Horizon class and metrics. """
#pylint: disable=too-many-lines, import-error
import os
from copy import copy
from functools import wraps
from textwrap import dedent
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numba import njit, prange
import matplotlib.pyplot as plt


import cv2
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from scipy.signal import hilbert, medfilt
from skimage.measure import label, regionprops

from ..batchflow import HistoSampler
from ..batchflow.models.metrics import Metrics

from .geometry import SeismicGeometry
from .utils import compute_running_mean
from .plot_utils import plot_image



class BaseLabel(ABC):
    """ Base class for labels. """
    @abstractmethod
    def from_file(self, path):
        """ Labels must be loadable from file on disk. """

    @abstractmethod
    def from_points(self, array):
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
            self.synchronized = False

            if after:
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


    def __init__(self, storage, geometry, name=None, debug=False, debug_attrs=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
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
                self.format = 'full_matrix'

            elif storage.ndim == 2:
                # matrix of (iline, xline) shape with every value being height
                self.format = 'matrix'

        getattr(self, 'from_{}'.format(self.format))(storage, **kwargs)
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
    def from_points(self, points, transform=False, **kwargs):
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


    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file. """
        _ = kwargs

        self.path = path
        self.name = os.path.basename(path)
        points = self.file_to_points(path)
        self.from_points(points, transform)

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


    def from_matrix(self, matrix, i_min, x_min, transform=False, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        points = self.matrix_to_points(matrix)
        points[:, 0] += i_min
        points[:, 1] += x_min
        self.from_points(points, transform=transform)


    def from_full_matrix(self, matrix, transform=False, **kwargs):
        """ Init from matrix that covers the whole cube. """
        _ = kwargs

        points = self.matrix_to_points(matrix)
        self.from_points(points, transform=transform)

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.asarray(matrix != Horizon.FILL_VALUE).nonzero()
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


    def from_dict(self, dictionary, transform=True, **kwargs):
        """ Init from mapping from (iline, xline) to depths. """
        _ = kwargs, dictionary, transform

        points = self.dict_to_points(dictionary)
        self.from_points(points, transform=transform)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    @staticmethod
    def from_mask(mask, geometry, min_point, threshold=0.5, averaging='mean', minsize=0,
                  prefix='prediction', **kwargs):
        """ Convert mask to a list of horizons.
        Returned list is sorted on length of horizons.

        Parameters
        ----------
        grid_info : dict
            Information about mask creation parameters. For details, check :meth:`.SeismicCubeset.mask_to_horizons`.
            Required keys are `geom` and `range` to infer geometry and leftmost upper point.
        threshold : float
            Parameter of mask-thresholding.
        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).
        minsize : int
            Minimum length of a horizon to be saved.
        prefix : str
            Name of horizon to use.
        """
        _ = kwargs
        i_min, x_min, h_min = min_point

        mask = np.copy(mask)
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        mask = mask.astype(np.int32)

        # Note that this does not evaluate anything: all of attributes are properties and are cached
        regions = regionprops(label(mask))

        # Create an instance of Horizon for each separate region
        horizons = []
        for i, region in enumerate(regions):

            coords = region.coords
            coords = pd.DataFrame(coords, columns=['iline', 'xline', 'height'])
            horizon = getattr(coords.groupby(['iline', 'xline']), averaging)()

            # columns in mask local coordinate system
            ilines = horizon.index.get_level_values('iline').values
            xlines = horizon.index.get_level_values('xline').values
            heights = horizon.values

            # convert to cubic coordinates, make points array, init Horizon from it
            if len(heights) > minsize:
                ilines += i_min
                xlines += x_min
                heights += h_min

                points = np.hstack([ilines.reshape(-1, 1),
                                    xlines.reshape(-1, 1),
                                    heights.reshape(-1, 1)])
                horizons.append(Horizon(points, geometry, name=f'{prefix}_{i}'))

        horizons.sort(key=len)
        return horizons


    # Functions to use to change the horizon
    def synchronize(self, base='matrix'):
        """ Synchronize the whole horizon instance with base storage. """
        if not self.synchronized:
            if base == 'matrix':
                self.from_matrix(self.matrix, self.i_min, self.x_min)
            elif base == 'points':
                self.from_points(self.points)
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


    def smooth_out(self, kernel_size=3, sigma=0.8, iters=1, preserve_borders=True, **kwargs):
        """ Convolve the horizon with gaussian kernel with special treatment to absent points:
        if the point was present in the original horizon, then it is changed to a weighted sum of all
        present points nearby;
        if the point was absent in the original horizon and there is at least one non-fill point nearby,
        then it is changed to a weighted sum of all present points nearby.

        Parameters
        ----------
        kernel_size : int
            Size of gaussian filter.
        sigma : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        """
        def smoothing_function(src, **kwds):
            _ = kwds
            k = int(np.floor(kernel_size / 2))

            ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
            x_points, y_points = np.meshgrid(ax, ax)
            kernel = np.exp(-0.5 * (np.square(x_points) + np.square(y_points)) / np.square(sigma))
            gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))
            raveled_gaussian_kernel = gaussian_kernel.ravel()

            @njit
            def _smoothing_function(src, fill_value):
                #pylint: disable=not-an-iterable
                dst = np.copy(src)
                for iline in range(k, src.shape[0]-k):
                    for xline in prange(k, src.shape[1]-k):
                        element = src[iline-k:iline+k+1, xline-k:xline+k+1]

                        s, sum_weights = 0.0, 0.0
                        for item, weight in zip(element.ravel(), raveled_gaussian_kernel):
                            if item != fill_value:
                                s += item * weight
                                sum_weights += weight

                        if sum_weights != 0.0:
                            val = s / sum_weights
                            dst[iline, xline] = val
                dst = np.rint(dst).astype(np.int32)
                return dst

            smoothed = src
            for _ in range(iters):
                smoothed = _smoothing_function(smoothed, self.FILL_VALUE)

            if preserve_borders:
                # pylint: disable=invalid-unary-operand-type
                idx_i, idx_x = np.asarray(~self.filled_matrix).nonzero()
                smoothed[idx_i, idx_x] = self.FILL_VALUE
            return smoothed

        self.apply_to_matrix(smoothing_function, before=False, after=True, **kwargs)


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


    def add_to_mask(self, mask, mask_bbox, width, alpha=1):
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
            mask[idx_i, idx_x, heights] = alpha
            heights += 1
        return mask


    def get_cube_values(self, window=23, offset=0, scale=False, chunk_size=256):
        """ Get values from the cube along the horizon.

        Parameters
        ----------
        window : int
            Width of data to cut.
        offset : int
            Value to add to each entry in matrix.
        scale : bool, callable
            If True, then values are scaled to [0, 1] range.
            If callable, then it is applied to iline-oriented slices of data from the cube.
        chunk_size : int
            Size of data along height axis processed at a time.
        """
        low = window // 2
        high = max(window - low, 0)
        chunk_size = min(chunk_size, self.h_max - self.h_min + window)

        h5py_cube = self.geometry.h5py_file['cube_h']
        background = np.zeros((self.geometry.ilines_len, self.geometry.xlines_len, window))

        # Make callable scaler
        if callable(scale):
            pass
        elif scale is True:
            scale = self.geometry.scaler
        elif scale is False:
            scale = lambda array: array


        for h_start in range(self.h_min - low, self.h_max + high, chunk_size):
            h_end = min(h_start + chunk_size, self.h_max + high, self.geometry.depth)

            # Get chunk from the cube (depth-wise)
            data_chunk = h5py_cube[h_start:h_end, :, :]
            data_chunk = scale(data_chunk)

            # Check which points of the horizon are in the current chunk (and present)
            idx_i, idx_x = np.asarray((self.matrix != self.FILL_VALUE) &
                                      (self.matrix >= h_start) &
                                      (self.matrix < h_end)).nonzero()
            heights = self.matrix[idx_i, idx_x]

            # Convert spatial coordinates to cubic, convert height to current chunk local system
            idx_i += self.i_min
            idx_x += self.x_min
            heights -= (h_start + low - offset)

            # Subsequently add values from the cube to background, shift horizon 1 unit lower,
            # remove all heights that are bigger than can fit into background
            for j in range(window):
                background[idx_i, idx_x, np.full_like(heights, j)] = data_chunk[heights, idx_i, idx_x]
                heights += 1

                idx_i = idx_i[heights < chunk_size]
                idx_x = idx_x[heights < chunk_size]
                heights = heights[heights < chunk_size]
        return background

    def get_cube_values_line(self, orientation='ilines', line=1, window=23, offset=0, scale=False):
        """ Get values from the cube along the horizon on a particular line.

        Parameters
        ----------
        orientation : str
            Whether to cut along ilines ('i') or xlines ('x').
        line : int
            Number of line to cut along.
        window : int
            Width of data to cut.
        offset : int
            Value to add to each entry in matrix.
        scale : bool, callable
            If True, then values are scaled to [0, 1] range.
            If callable, then it is applied to iline-oriented slices of data from the cube.
        chunk_size : int
            Size of data along height axis processed at a time.
        """
        low = window // 2

        # Make callable scaler
        if callable(scale):
            pass
        elif scale is True:
            scale = self.geometry.scaler
        elif scale is False:
            scale = lambda array: array

        # Parameters for different orientation
        if orientation.startswith('i'):
            h5py_cube = self.geometry.h5py_file['cube']
            slide_transform = lambda array: array

            hor_line = np.squeeze(self.matrix[line, :])
            background = np.zeros((self.geometry.xlines_len, window))
            idx_offset = self.x_min
            bad_traces = np.squeeze(self.geometry.zero_traces[line, :])

        elif orientation.startswith('x'):
            h5py_cube = self.geometry.h5py_file['cube_x']
            slide_transform = lambda array: array.T

            hor_line = np.squeeze(self.matrix[:, line])
            background = np.zeros((self.geometry.ilines_len, window))
            idx_offset = self.i_min
            bad_traces = np.squeeze(self.geometry.zero_traces[:, line])

        # Check where horizon is
        idx = np.asarray((hor_line != self.FILL_VALUE)).nonzero()[0]
        heights = hor_line[idx]

        # Convert coordinates to cubic system
        idx += idx_offset
        heights -= (low - offset)

        slide = h5py_cube[line, :, :]
        slide = slide_transform(slide)
        slide = scale(slide)

        # Subsequently add values from the cube to background and shift horizon 1 unit lower
        for j in range(window):
            test = slide[idx, heights]
            background[idx, np.full_like(idx, j)] = test
            heights += 1

        idx = np.asarray((hor_line == self.FILL_VALUE)).nonzero()[0]
        idx += idx_offset
        bad_traces[idx] = 1

        bad_traces = bad_traces.reshape((1, -1) if orientation.startswith('i') else (-1, 1))
        background = background.reshape((1, -1, window) if orientation.startswith('i') else (-1, 1, window))
        return background, bad_traces


    # Horizon properties
    @property
    def amplitudes(self):
        """ Values from the cube along the horizon. """
        return self.get_cube_values(window=1)

    @property
    def binary_matrix(self):
        """ Matrix with ones at places where horizon is present and zeros everywhere else. """
        return (self.matrix > 0).astype(bool)

    @property
    def borders_matrix(self):
        """ Borders of horizons (borders of holes inside are not included). """
        filled_matrix = self.filled_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(filled_matrix, structure, border_value=0)
        return filled_matrix ^ eroded # binary difference operation

    @property
    def boundaries_matrix(self):
        """ Borders of horizons (borders of holes inside included). """
        binary_matrix = self.binary_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(binary_matrix, structure, border_value=0)
        return binary_matrix ^ eroded # binary difference operation

    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        return len(self) / (np.prod(self.matrix.shape) - np.sum(self.geometry.zero_traces))

    @property
    def filled_matrix(self):
        """ Binary matrix with filled holes. """
        structure = np.ones((3, 3))
        filled_matrix = binary_fill_holes(self.binary_matrix, structure)
        return filled_matrix

    @property
    def full_matrix(self):
        """ Matrix in cubic coordinate system. """
        return self.put_on_full()

    @property
    def grad_i(self):
        """ Change of heights along iline direction. """
        return self.grad_along_axis(0)

    @property
    def grad_x(self):
        """ Change of heights along xline direction. """
        return self.grad_along_axis(1)

    @property
    def hash(self):
        """ Hash on current data of the horizon. """
        return hash(self.matrix.data.tobytes())

    @property
    def number_of_holes(self):
        """ Number of holes inside horizon borders. """
        holes_array = self.filled_matrix != self.binary_matrix
        _, num = label(holes_array, connectivity=2, return_num=True, background=0)
        return num

    @property
    def perimeter(self):
        """ Number of points in the borders. """
        return np.sum((self.borders_matrix == 1).astype(np.int32))

    @property
    def solidity(self):
        """ Ratio of area covered by horizon to total area inside borders. """
        return len(self) / np.sum(self.filled_matrix)


    def grad_along_axis(self, axis=0):
        """ Change of heights along specified direction. """
        grad = np.diff(self.matrix, axis=axis, prepend=0)
        grad[np.abs(grad) > 10000] = self.FILL_VALUE
        grad[self.matrix == self.FILL_VALUE] = self.FILL_VALUE
        return grad


    # Evaluate horizon on its own / against other(s)
    def evaluate(self, supports=20, plot=True):
        """ Compute crucial metrics of a horizon. """
        msg = f"""
        Number of labeled points: {len(self)}
        Number of points inside borders: {np.sum(self.filled_matrix)}
        Perimeter (length of borders): {self.perimeter}
        Percentage of labeled non-bad traces: {self.coverage}
        Percentage of labeled traces inside borders: {self.solidity}
        Number of holes inside borders: {self.number_of_holes}
        """
        print(dedent(msg))

        HorizonMetrics(self).evaluate('support_corrs', supports=supports, agg='mean', plot=plot)


    def compare_to(self, other, offset=0, absolute=True, printer=print, hist=True, plot=True):
        """ Shortcut for :meth:`.HorizonMetrics.evaluate` to compare against the best match of list of horizons. """
        HorizonMetrics([self, other]).evaluate('compare', agg=None, absolute=absolute, offset=offset,
                                               printer=printer, hist=hist, plot=plot)


    # Merge functions
    def verify_merge(self, other, mean_threshold=3.0, q_threshold=2.5, q=0.9, adjacency=0):
        """ Collect stats of overlapping of two horizons.

        Returns a number that encodes position of two horizons, as well as dictionary with collected statistics.
        If code is 0, then horizons are too far away from each other (heights-wise), and therefore are not mergeable.
        If code is 1, then horizons are too far away from each other (spatially) even with adjacency, and therefore
        are not mergeable.
        If code is 2, then horizons are close enough spatially (with adjacency), but are not overlapping, and therefore
        an additional check (`adjacent_merge`) is needed.
        If code is 3, then horizons are definitely overlapping and are close enough to meet all the thresholds, and
        therefore are mergeable without any additional checks.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to compare.
        mean_threshold : number
            Height threshold for mean distances.
        q_threshold : number
            Height threshold for quantile distances.
        q : number
            Quantile to compute.
        adjacency : int
            Margin to consider horizons close (spatially).
        """
        overlap_info = {}

        # Overlap bbox
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1
        overlap_bbox = np.array([[overlap_i_min, overlap_i_max],
                                 [overlap_x_min, overlap_x_max]],
                                dtype=np.int32)

        overlap_info.update({'i_min': overlap_i_min,
                             'i_max': overlap_i_max,
                             'x_min': overlap_x_min,
                             'x_max': overlap_x_max,
                             'bbox': overlap_bbox})

        # Simplest possible check: horizon bboxes are too far from each other
        if (overlap_i_min - overlap_i_max > adjacency) or (overlap_x_min - overlap_x_max > adjacency):
            merge_code = 1
            spatial_position = 'distant'
        else:
            merge_code = 2
            spatial_position = 'adjacent'

        # Compare matrices on overlap without adjacency:
        if (overlap_i_min - overlap_i_max < 0) and (overlap_x_min - overlap_x_max < 0):
            self_overlap = self.matrix[overlap_i_min - self.i_min:overlap_i_max - self.i_min,
                                       overlap_x_min - self.x_min:overlap_x_max - self.x_min]

            other_overlap = other.matrix[overlap_i_min - other.i_min:overlap_i_max - other.i_min,
                                         overlap_x_min - other.x_min:overlap_x_max - other.x_min]

            diffs_on_overlap = np.where((self_overlap != self.FILL_VALUE) & (other_overlap != self.FILL_VALUE),
                                        self_overlap - other_overlap, np.nan)
            abs_diffs = np.abs(diffs_on_overlap)

            mean_on_overlap = np.nanmean(abs_diffs)
            q_on_overlap = np.nanquantile(abs_diffs, q)
            max_on_overlap = np.nanmax(abs_diffs)

            if np.isnan(mean_on_overlap):
                # bboxes are overlapping, but horizons don't
                merge_code = 2
                spatial_position = 'adjacent'
            elif mean_on_overlap < mean_threshold and q_on_overlap < q_threshold:
                merge_code = 3
                spatial_position = 'overlap'
            else:
                merge_code = 0
                spatial_position = 'separated'

            overlap_info.update({'mean': mean_on_overlap,
                                 'q': q_on_overlap,
                                 'max': max_on_overlap,
                                 'diffs': diffs_on_overlap[~np.isnan(diffs_on_overlap)],
                                 'nandiffs': diffs_on_overlap,})

        overlap_info['spatial_position'] = spatial_position
        return merge_code, overlap_info



    def adjacent_merge(self, other, mean_threshold=3.0, q_threshold=2.5, q=0.9, adjacency=3,
                       check_only=False, force_merge=False, inplace=False):
        """ Collect stats on possible adjacent merge (that is merge with some margin), and, if needed, merge horizons.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to compare.
        mean_threshold : number
            Height threshold for mean distances.
        q_threshold : number
            Height threshold for quantile distances.
        q : number
            Quantile to compute.
        adjacency : int
            Margin to consider horizons close (spatially).
        check_only : bool
            Whether to try to merge horizons or just collect the stats.
        force_merge : bool
            Whether to chcek stats before merging. Can be useful if used after :class:`.Horizon.verify_merge` method.
        inplace : bool
            Whether to create new instance or update `self`.
        """
        adjacency_info = {}

        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.full((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                             self.FILL_VALUE, dtype=np.int32)

        # Make all the indices for later usage: in every coordinate system for both horizons
        self_idx_i = self.points[:, 0] - shared_i_min # in shared system
        self_idx_x = self.points[:, 1] - shared_x_min
        self_idx_i_ = self.points[:, 0] - self.i_min  # in local system of self
        self_idx_x_ = self.points[:, 1] - self.x_min

        other_idx_i = other.points[:, 0] - shared_i_min # in shared system
        other_idx_x = other.points[:, 1] - shared_x_min
        other_idx_i_ = other.points[:, 0] - other.i_min # in local system of other
        other_idx_x_ = other.points[:, 1] - other.x_min


        # Put the second of the horizons on background
        background[other_idx_i, other_idx_x] = other.matrix[other_idx_i_, other_idx_x_]

        # Enlarge the image to count for adjacency
        if not force_merge:
            kernel = np.ones((3, 3), np.float32)
            dilated_background = cv2.dilate(background.astype(np.float32), kernel,
                                            iterations=adjacency).astype(np.int32)
        else:
            dilated_background = background

        # Make counts: number of horizons for each point (can be either 0, 1 or 2)
        counts = (dilated_background > 0).astype(np.int32)
        counts[self_idx_i, self_idx_x] += 1

        # Gete heights on overlap
        so_idx_i, so_idx_x = np.asarray(counts == 2).nonzero()
        self_idx_i_so = so_idx_i - self.i_min + shared_i_min
        self_idx_x_so = so_idx_x - self.x_min + shared_x_min
        self_heights = self.matrix[self_idx_i_so, self_idx_x_so]
        other_heights = background[so_idx_i, so_idx_x]

        # Compare heights to check whether horizons are close enough to each other
        mergeable = force_merge
        if len(so_idx_i) != 0 and not force_merge:
            other_heights_dilated = dilated_background[so_idx_i, so_idx_x]

            diffs_on_shared_overlap = self_heights - other_heights_dilated
            abs_diffs = np.abs(diffs_on_shared_overlap)

            mean_on_shared_overlap = np.mean(abs_diffs)
            q_on_shared_overlap = np.quantile(abs_diffs, q)
            max_on_shared_overlap = np.max(abs_diffs)

            if mean_on_shared_overlap < mean_threshold and q_on_shared_overlap < q_threshold:
                mergeable = True

            adjacency_info.update({'mean': mean_on_shared_overlap,
                                   'q': q_on_shared_overlap,
                                   'max': max_on_shared_overlap,
                                   'diffs': diffs_on_shared_overlap})

        # Actually merge two horizons
        merged = None
        if check_only is False and mergeable:
            # Put the first horizon on background
            background[self_idx_i, self_idx_x] = self.matrix[self_idx_i_, self_idx_x_]

            # Values on overlap
            overlap_heights = np.where(other_heights != self.FILL_VALUE,
                                       (self_heights + other_heights) / 2, self_heights)

            background[so_idx_i, so_idx_x] = overlap_heights

            if inplace:
                # Change `self` inplace
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min)
            else:
                # Return a new instance of horizon
                merged = Horizon(background, self.geometry, self.name,
                                 i_min=shared_i_min, x_min=shared_x_min)

        return mergeable, merged, adjacency_info


    # Convenient overloads
    def __getitem__(self, point):
        """ Get value in point in cubic coordinates. """
        point = point[0] - self.i_min, point[1] - self.x_min
        return self.matrix[point]

    def __len__(self):
        """ Number of labeled traces. """
        return len(np.asarray(self.matrix != self.FILL_VALUE).nonzero()[0])


    def dump_horizon(self, path, transform=None, add_height=True):
        """ Save horizon points on disc.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        self.synchronize()
        values = self.cubic_to_lines(copy(self.points))
        values = values if transform is None else transform(values)

        df = pd.DataFrame(values, columns=self.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)

        path = path if not add_height else '{}_#{}'.format(path, self.h_mean)
        df.to_csv(path, sep=' ', columns=self.COLUMNS, index=False, header=False)


    # Methods of (visual) representation of a horizon
    def __repr__(self):
        return f"""<horizon {self.name} for {self.cube_name} at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.cube_name} loaded from {self.format}
        Ilines from {self.i_min} to {self.i_max}
        Xlines from {self.x_min} to {self.x_max}
        Heights from {self.h_min} to {self.h_max}, mean is {self.h_mean:5.5}, std is {self.h_std:4.4}
        Currently synchronized: {self.synchronized}; In debug mode: {self.debug}; At {hex(id(self))}
        """
        return dedent(msg)


    def put_on_full(self, matrix=None, fill_value=None):
        """ Create a matrix in cubic coordinate system. """
        matrix = matrix if matrix is not None else self.matrix
        fill_value = fill_value if fill_value is not None else self.FILL_VALUE

        background = np.full(self.cube_shape[:-1], fill_value, dtype=np.float32)
        background[self.i_min:self.i_max+1, self.x_min:self.x_max+1] = matrix
        return background


    def show(self, src='matrix', fill_value=None, on_full=True, **kwargs):
        """ Nice visualization of a horizon-related matrix. """
        matrix = getattr(self, src) if isinstance(src, str) else src
        fill_value = fill_value if fill_value is not None else self.FILL_VALUE

        if on_full:
            matrix = self.put_on_full(matrix=matrix, fill_value=fill_value)
        else:
            matrix = copy(matrix).astype(np.float32)

        matrix[matrix == fill_value] = np.nan
        plot_image(matrix, 'Depth map {} of {} on {}'.format('on full'*on_full, self.name, self.cube_name),
                   cmap='viridis_r', **kwargs)


    def show_amplitudes_rgb(self, width=3, **kwargs):
        """ Show trace values on the horizon and surfaces directly under it.

        Parameters
        ----------
        width : int
            Space between surfaces to cut.
        """
        amplitudes = self.get_cube_values(window=1 + width*2, offset=width)

        amplitudes = amplitudes[:, :, (0, width, -1)]
        amplitudes -= amplitudes.min(axis=(0, 1)).reshape(1, 1, -1)
        amplitudes *= 1 / amplitudes.max(axis=(0, 1)).reshape(1, 1, -1)
        amplitudes[self.fullmatrix == self.FILL_VALUE, :] = 0
        amplitudes = amplitudes[:, :, ::-1]
        amplitudes *= np.asarray([1, 0.5, 0.25]).reshape(1, 1, -1)
        plot_image(amplitudes, 'RGB amplitudes of {} on cube {}'.format(self.name, self.cube_name),
                   rgb=True, **kwargs)




class HorizonMetrics(Metrics):
    """ Evaluate metric(s) on horizon(s).
    During initialization, data along the horizon is cut with the desired parameters.
    To get the value of a particular metric, use :meth:`.evaluate`::
        HorizonMetrics(horizon).evaluate('support_corrs', supports=20, agg='mean')

    To plot the results, set `plot` argument of :meth:`.evaluate` to True.

    Parameters
    horizons : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to evaluate.
        Can be either one horizon, then this horizon is evaluated on its own,
        or sequence of two horizons, then they are compared against each other,
        or nested sequence of horizon and list of horizons, then the first horizon is compared against the
        best match from the list.
    other parameters
        Passed direcly to :meth:`.Horizon.get_cube_values` or :meth:`.Horizon.get_cube_values_line`.
    """
    def __init__(self, horizons, orientation=None, window=23, offset=0, scale=False, chunk_size=256, line=1):
        super().__init__()
        horizons = list(horizons) if isinstance(horizons, tuple) else horizons
        horizons = horizons if isinstance(horizons, list) else [horizons]
        self.horizons = horizons

        # Save parameters for later evaluation
        self.orientation, self.line = orientation, line
        self.window, self.offset, self.scale, self.chunk_size = window, offset, scale, chunk_size

        # The first horizon is used to evaluate metrics
        self.horizon = horizons[0]
        if orientation is None: # metrics are computed on full cube (spatially)
            self._data = None # evaluated later
            self.bad_traces = np.copy(self.horizon.geometry.zero_traces)
            self.bad_traces[self.horizon.full_matrix == Horizon.FILL_VALUE] = 1
            self.spatial = True

        else: # metrics are computed on a specific slide
            self._data, self.bad_traces = self.horizon.get_cube_values_line(orientation=orientation, line=line,
                                                                            window=window, offset=offset, scale=scale)
            self.spatial = False


    @property
    def data(self):
        """ Create `data` attribute at the first time of evaluation. """
        if self._data is None:
            self._data = self.horizon.get_cube_values(window=self.window, offset=self.offset,
                                                      scale=self.scale, chunk_size=self.chunk_size)
        return self._data


    def evaluate(self, metrics, agg='mean', plot=False, show_plot=True, savepath=None, backend='matplotlib',
                 plot_kwargs=None, scalar=False, **kwargs):
        """ Calculate desired metrics.
        To plot the results, set `plot` argument to True.

        Parameters
        ----------
        metrics : str or sequence of str
            Names of metrics to evaluate.
        agg : int, str or callable
            Function to transform metric from ndarray of (n_ilines, n_xlines, N) shape to (n_ilines, n_xlines) shape.
            If callable, then directly applied to the output of metric computation function.
            If str, then must be a function from `numpy` module. Applied along the last axis only.
            If int, then index of slice along the last axis to return.
        kwargs : dict
            Metric-specific parameters.

        Returns
        -------
        If `metric` is str, then metric value
        If `metric` is dict, than dict where keys are metric names and values are metric values.
        """
        _metrics = [metrics] if isinstance(metrics, str) else metrics
        _agg = [agg]*len(_metrics) if not isinstance(agg, (tuple, list)) else agg

        res = {}
        for name, agg_func in zip(_metrics, _agg):
            # Get metric, then aggregate
            metric_fn = getattr(self, name)
            metric_val, plot_dict = metric_fn(**kwargs)
            metric_val = self._aggregate(metric_val, agg_func)

            # Get plot parameters
            # TODO: make plot functions use only needed parameters
            ignore_value = plot_dict.pop('ignore_value', None)
            spatial = plot_dict.pop('spatial', True)
            _ = backend, plot_kwargs, plot_dict.pop('zmin', -1), plot_dict.pop('zmax', 1)

            # np.nan allows to ignore values
            if ignore_value is not None:
                copy_metric = np.copy(metric_val)
                copy_metric[copy_metric == ignore_value] = np.nan
            else:
                copy_metric = metric_val

            # Actual plot
            if plot:
                if spatial:
                    plot_image(copy_metric, savefig=savepath, show_plot=show_plot, **plot_dict)
                else:
                    pass
            if scalar:
                print('Scalar value of metric is {}'.format(np.nanmean(copy_metric)))
            res[name] = metric_val

        res = res[metrics] if isinstance(metrics, str) else res
        return res


    def _aggregate(self, metric, agg=None):
        if agg is not None:
            if callable(agg):
                metric = agg(metric)
            elif isinstance(agg, str):
                metric = getattr(np, agg)(metric, axis=-1)
            elif isinstance(agg, (int, slice)):
                metric = metric[..., agg]
        return metric


    def local_corrs(self, locality=4, **kwargs):
        """ Compute average correlation between each column in data and nearest traces.

        Parameters
        ----------
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

        bad_traces = np.copy(self.bad_traces)
        bad_traces[np.std(self.data, axis=-1) == 0.0] = 1
        metric = _compute_local_corrs(self.data, bad_traces, locs)
        title = 'local correlation'

        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict


    def support_corrs(self, supports=1, safe_strip=0, line_no=None, **kwargs):
        """ Compute correlations with support traces.

        Parameters
        ----------
        supports : int, sequence, ndarray or str
            Defines mode of generating support traces.
            If int, then that number of random non-zero traces positions are generated.
            If sequence or ndarray, then must be of shape (N, 2) and is used as positions of support traces.
            If str, then must define either `iline` or `xline` mode. In each respective one, iline/xline given by
            `line_no` argument is used to generate supports.
        safe_strip : int
            Used only for `int` mode of `supports` parameter and defines minimum distance
            from borders for sampled points.
        line_no : int
            Used only for `str` mode of `supports` parameter to define exact iline/xline to use.

        Returns
        -------
        array-like
            Matrix of either (n_ilines, n_xlines, n_supports) or (n_ilines, n_xlines) shape with
            computed metric for each point.
        """
        _ = kwargs
        bad_traces = np.copy(self.bad_traces)
        bad_traces[np.std(self.data, axis=-1) == 0.0] = 1

        if isinstance(supports, (int, tuple, list, np.ndarray)):
            if isinstance(supports, int):
                title = 'correlation with {} random supports'.format(supports)
                if safe_strip:
                    bad_traces[:, :safe_strip], bad_traces[:, -safe_strip:] = 1, 1
                    bad_traces[:safe_strip, :], bad_traces[-safe_strip:, :] = 1, 1

                non_zero_traces = np.where(bad_traces == 0)
                indices = np.random.choice(len(non_zero_traces[0]), supports)
                supports = np.array([non_zero_traces[0][indices], non_zero_traces[1][indices]]).T

            elif isinstance(supports, (tuple, list, np.ndarray)):
                title = 'correlation with {} supports'.format(len(supports))
                if min(len(item) == 2 for item in supports) is False:
                    raise ValueError('Each of `supports` sequence must contain coordinate of trace (il, xl). ')
                supports = np.array(supports)

            metric = _compute_support_corrs_np(self.data, supports, bad_traces)

        elif isinstance(supports, str):
            title = 'correlation on {} {}'.format(line_no, supports)
            if supports.startswith('i'):
                support_il = line_no or self.data.shape[0] // 2
                metric = _compute_line_corrs_np(self.data, bad_traces, support_il=support_il)

            if supports.startswith('x'):
                support_xl = line_no or self.data.shape[1] // 2
                metric = _compute_line_corrs_np(self.data, bad_traces, support_xl=support_xl)

        else:
            raise ValueError('`Supports` must be either int, sequence, ndarray or string. ')

        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict


    def hilbert(self, mode='median', kernel_size=3, eps=1e-5, **kwargs):
        """ Compute phase along the horizon. """
        _ = kwargs
        fullmatrix = self.horizon.fullmatrix

        analytic = hilbert(self.data, axis=-1)
        phase = (np.angle(analytic))
        phase = phase % (2 * np.pi) - np.pi
        phase[fullmatrix == Horizon.FILL_VALUE, :] = 0

        horizon_phase = phase[:, :, phase.shape[-1] // 2]
        horizon_phase = correct_pi(horizon_phase, eps)

        if mode == 'mean':
            median_phase = compute_running_mean(horizon_phase, kernel_size)
        else:
            median_phase = medfilt(horizon_phase, kernel_size)
        median_phase[fullmatrix == Horizon.FILL_VALUE] = 0

        img = np.minimum(median_phase - horizon_phase, 2 * np.pi + horizon_phase - median_phase)
        img[fullmatrix == Horizon.FILL_VALUE] = 0
        img = np.where(img < -np.pi, img + 2 * np. pi, img)

        metric = np.zeros((*img.shape, 2+self.data.shape[2]))
        metric[:, :, 0] = img
        metric[:, :, 1] = median_phase
        metric[:, :, 2:] = phase

        title = 'phase by {}'.format(mode)
        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            # **kwargs
        }
        return metric, plot_dict


    def compare(self, offset=0, absolute=True, hist=True, printer=print, **kwargs):
        """ Compare horizons on against the best match from the list of horizons.

        Parameters
        ----------
        offset : number
            Value to shift horizon down. Can be used to take into account different counting bases.
        absolute : bool
            Whether to use absolute values for differences.
        hist : bool
            Whether to plot histogram of differences.
        printer : callable
            Function to print results, for example `print` or any other callable that can log data.
        """
        _ = kwargs
        if len(self.horizons) != 2:
            raise ValueError('Can compare two horizons exactly or one to the best match from list of horizons. ')
        if isinstance(self.horizons[1], Horizon):
            self.horizons[1] = [self.horizons[1]]

        lst = []
        for hor in self.horizons[1]:
            if hor.geometry.name == self.horizon.geometry.name:
                overlap_info = Horizon.verify_merge(self.horizon, hor, adjacency=3)[1]
                lst.append((hor, overlap_info))
        lst.sort(key=lambda x: x[1].get('mean', 999999))
        other, overlap_info = lst[0] # the best match

        self_fullmatrix = self.horizon.full_matrix
        other_fullmatrix = other.full_matrix
        metric = np.where((self_fullmatrix != other.FILL_VALUE) & (other_fullmatrix != other.FILL_VALUE),
                          offset + self_fullmatrix - other_fullmatrix, np.nan)
        if absolute:
            metric = np.abs(metric)

        window_rate = np.mean(np.abs(metric[~np.isnan(metric)]) < (5 / other.geometry.sample_rate))
        max_abs_error = np.nanmax(np.abs(metric))
        max_abs_error_count = np.sum(metric == max_abs_error) + np.sum(metric == -max_abs_error)
        at_1 = len(np.asarray((self_fullmatrix != other.FILL_VALUE) &
                              (other_fullmatrix == other.FILL_VALUE)).nonzero()[0])
        at_2 = len(np.asarray((self_fullmatrix == other.FILL_VALUE) &
                              (other_fullmatrix != other.FILL_VALUE)).nonzero()[0])

        if printer is not None:
            msg = f"""
            Comparing horizons:       {self.horizon.name}
                                    {other.name}
            {'—'*45}

            Rate in 5ms:                         {window_rate:8.4}
            Mean/std of errors:       {np.nanmean(metric):8.4} / {np.nanstd(metric):8.4}
            Max abs error/count:      {max_abs_error:8.4} / {max_abs_error_count:8}
            {'—'*45}

            Lengths of horizons:                 {len(self.horizon):8}
                                                {len(other):8}
            {'—'*45}
            Average heights of horizons:         {(offset + self.horizon.h_mean):8.4}
                                                {other.h_mean:8.4}
            {'—'*45}
            Coverage of horizons:                {self.horizon.coverage:8.4}
                                                {other.coverage:8.4}
            {'—'*45}
            Solidity of horizons:                {self.horizon.solidity:8.4}
                                                {other.solidity:8.4}
            {'—'*45}
            Number of holes in horizons:         {self.horizon.number_of_holes:8}
                                                {other.number_of_holes:8}
            {'—'*45}
            Additional traces labeled:           {at_1:8}
            (present in one, absent in other)    {at_2:8}
            {'—'*45}
            """
            printer(dedent(msg))

        if hist and not np.isnan(max_abs_error):
            _ = plt.hist(metric.ravel(), bins=100)

        title = 'Height differences between {} and {}'.format(self.horizon.name, other.name)
        plot_dict = {
            'spatial': True,
            'title': '{} on cube {}'.format(title, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': 0, 'zmax': np.max(metric),
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


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
def correct_pi(horizon_phase, eps):
    """ Jit-accelerated function to <>. """
    for i in range(horizon_phase.shape[0]):
        prev = horizon_phase[i, 0]
        for j in range(1, horizon_phase.shape[1] - 1):
            if np.abs(np.abs(prev) - np.pi) <= eps and np.abs(np.abs(horizon_phase[i, j + 1]) - np.pi) <= eps:
                horizon_phase[i, j] = prev
            prev = horizon_phase[i, j]
    return horizon_phase
