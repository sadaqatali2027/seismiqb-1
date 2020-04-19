""" Horizon class and metrics. """
#pylint: disable=too-many-lines, import-error
import os
from copy import copy
from itertools import product
from textwrap import dedent
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numba import njit, prange

import cv2
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from scipy.ndimage import find_objects
from skimage.measure import label

from ..batchflow import HistoSampler

from .utils import round_to_array, groupby_mean
from .plot_utils import plot_image, plot_images_overlap, show_sampler



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



class UnstructuredHorizon(BaseLabel):
    """  Contains unstructured horizon.

    Initialized from `storage` and `geometry`, where `storage` is a csv-like file.

    The main inner storage is `dataframe`, that is a mapping from indexing headers to horizon depth.
    Since the index of that dataframe is the same, as the index of dataframe of a geometry, these can be combined.

    UnstructuredHorizon provides following features:
        - Method `add_to_mask` puts 1's on the location of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - Few of visualization methods: view from above, slices along iline/xline axis, etc.

    There is a lazy method creation in place: the first person who needs them would need to code them.
    """
    CHARISMA_SPEC = ['INLINE', '_', 'INLINE_3D', 'XLINE', '__', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y', 'height']
    REDUCED_CHARISMA_SPEC = ['INLINE_3D', 'CROSSLINE_3D', 'height']

    FBP_SPEC = ['FieldRecord', 'TraceNumber', 'file_id', 'FIRST_BREAK_TIME']

    def __init__(self, storage, geometry, name=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.format = None

        # Storage
        self.dataframe = None
        self.attached = False

        # Heights information
        self.h_min, self.h_max = None, None
        self.h_mean, self.h_std = None, None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, pd.DataFrame):
            # points-like dataframe
            self.format = 'dataframe'

        elif isinstance(storage, np.ndarray) and storage.ndim == 2 and storage.shape[1] == 3:
            # array with row in (iline, xline, height) format
            self.format = 'points'

        getattr(self, 'from_{}'.format(self.format))(storage, **kwargs)


    def from_points(self, points, **kwargs):
        """ Not needed. """


    def from_dataframe(self, dataframe, attach=True, height_prefix='height', transform=False):
        """ Load horizon data from dataframe.

        Parameters
        ----------
        attack : bool
            Whether to store horizon data in common dataframe inside `geometry` attributes.
        transform : bool
            Whether to transform line coordinates to the cubic ones.
        height_prefix : str
            Column name with height.
        """
        if transform:
            dataframe[height_prefix] = (dataframe[height_prefix] - self.geometry.delay) / self.geometry.sample_rate
        dataframe.rename(columns={height_prefix: self.name}, inplace=True)
        dataframe.set_index(self.geometry.index_headers, inplace=True)
        self.dataframe = dataframe

        self.h_min, self.h_max = self.dataframe.min().values[0], self.dataframe.max().values[0]
        self.h_mean, self.h_std = self.dataframe.mean().values[0], self.dataframe.std().values[0]

        if attach:
            self.attach()


    def from_file(self, path, names=None, columns=None, height_prefix='height', reader_params=None, **kwargs):
        """ Init from path to csv-like file.

        Parameters
        ----------
        names : sequence of str
            Names of columns in file.
        columns : sequence of str
            Names of columns to actually load.
        height_prefix : str
            Column name with height.
        reader_params : None or dict
            Additional parameters for file reader.
        """
        #pylint: disable=anomalous-backslash-in-string
        _ = kwargs
        if names is None:
            with open(path) as file:
                line_len = len(file.readline().split(' '))
            if line_len == 3:
                names = UnstructuredHorizon.REDUCED_CHARISMA_SPEC
            elif line_len == 9:
                names = UnstructuredHorizon.CHARISMA_SPEC
        columns = columns or self.geometry.index_headers + [height_prefix]

        self.path = path
        self.name = os.path.basename(path)

        defaults = {'sep': '\s+'}
        reader_params = reader_params or {}
        reader_params = {**defaults, **reader_params}
        df = pd.read_csv(path, names=names, usecols=columns, **reader_params)

        # Convert coordinates of horizons to the one that present in cube geometry
        # df[columns] = np.rint(df[columns]).astype(np.int64)
        df[columns] = df[columns].astype(np.int64)
        for i, idx in enumerate(self.geometry.index_headers):
            df[idx] = round_to_array(df[idx].values, self.geometry.uniques[i])

        self.from_dataframe(df, transform=True, height_prefix=columns[-1])

    def attach(self):
        """ Store horizon data in common dataframe inside `geometry` attributes. """
        if not hasattr(self.geometry, 'horizons'):
            self.geometry.horizons = pd.DataFrame(index=self.geometry.dataframe.index)

        self.geometry.horizons = pd.merge(self.geometry.horizons, self.dataframe,
                                          left_index=True, right_index=True,
                                          how='left')
        self.attached = True


    def add_to_mask(self, mask, locations=None, width=3, alpha=1, iterator=None, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : sequence of arrays
            List of desired locations to load: along the first index, the second, and depth.
        width : int
            Width of an added horizon.
        iterator : None or sequence
            If provided, indices to use to load height from dataframe.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        shift_1, shift_2, h_min = [np.min(item) for item in locations]
        h_max = np.max(locations[-1])

        if iterator is None:
            # Usual case
            iterator = list(product(*[[self.geometry.uniques[idx][i] for i in locations[idx]] for idx in range(2)]))
            idx_iterator = np.array(list(product(*locations[:2])))
            idx_1 = idx_iterator[:, 0] - shift_1
            idx_2 = idx_iterator[:, 1] - shift_2

        else:
            #TODO: remove this and make separate method inside `SeismicGeometry` for loading data with same iterator
            #TODO: think about moving horizons to `geometry` attributes altogether..
            # Currently, used in `show_slide` only:
            axis = np.argmin(np.array([len(np.unique(np.array(iterator)[:, idx])) for idx in range(2)]))
            loc = iterator[axis][0]
            other_axis = 1 - axis

            others = self.geometry.dataframe[self.geometry.dataframe.index.get_level_values(axis) == loc]
            others = others.index.get_level_values(other_axis).values
            others_iterator = np.array([np.where(others == item[other_axis])[0][0] for item in iterator])

            idx_1 = np.zeros_like(others_iterator) if axis == 0 else others_iterator
            idx_2 = np.zeros_like(others_iterator) if axis == 1 else others_iterator


        heights = self.dataframe[self.name].get(iterator, np.nan).values.astype(np.int32)

        # Filter labels based on height
        heights_mask = np.asarray((np.isnan(heights) == False) & # pylint: disable=singleton-comparison
                                  (heights >= h_min + low) &
                                  (heights <= h_max - high)).nonzero()[0]

        idx_1 = idx_1[heights_mask]
        idx_2 = idx_2[heights_mask]
        heights = heights[heights_mask]
        heights -= (h_min + low)

        # Place values on current heights and shift them one unit below.
        for _ in range(width):
            mask[idx_1, idx_2, heights] = alpha
            heights += 1
        return mask

    # Methods to implement in the future
    def filter_points(self, **kwargs):
        """ Remove points that correspond to bad traces, e.g. zero traces. """

    def merge(self, **kwargs):
        """ Merge two instances into one. """

    def get_cube_values(self, **kwargs):
        """ Get cube values along the horizon.
        Can be easily done via subsequent `segyfile.depth_slice` and `reshape`.
        """

    def dump(self, **kwargs):
        """ Save the horizon to the disk. """

    # Visualization
    def show_slide(self, loc, width=3, axis=0, stable=True, order_axes=None, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([len(item) for item in locations])

        # Create the same indices, as for seismic slide loading
        #TODO: make slide indices shareable
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis, stable=stable)
        _, iterator = self.geometry.make_slide_indices(loc=loc, axis=axis, stable=stable, return_iterator=True)
        shape[1 - axis] = -1

        # Create mask with horizon
        mask = np.zeros_like(seismic_slide.reshape(shape))
        mask = self.add_to_mask(mask, locations, width=width, iterator=iterator if stable else None)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)

        # Display everything
        title = f'{self.geometry.index_headers[axis]} {loc} out of {self.geometry.lens[axis]}'
        meta_title = f'U-horizon {self.name} on {self.geometry.name}'
        plot_images_overlap([seismic_slide, mask], title=title, order_axes=order_axes, meta_title=meta_title, **kwargs)




class Horizon(BaseLabel):
    """ Contains spatially-structured horizon: each point describes a height on a particular (iline, xline).

    Initialized from `storage` and `geometry`.

    Storage can be one of:
        - csv-like file in CHARISMA or REDUCED_CHARISMA format.
        - ndarray of (N, 3) shape.
        - ndarray of (ilines_len, xlines_len) shape.
        - dictionary: a mapping from (iline, xline) -> height.
        - mask: ndarray of (ilines_len, xlines_len, depth) with 1's at places of horizon location.

    Main storages are `matrix` and `points` attributes: they are loaded lazily at the time of first access.
        - `matrix` is a depth map, ndarray of (ilines_len, xlines_len) shape with each point
          corresponding to horizon height at this point. Note that shape of the matrix is generally smaller
          than cube spatial range: that allows to save space.
          Attributes `i_min` and `x_min` describe position of the matrix in relation to the cube spatial range.
          Each point with absent horizon is filled with `FILL_VALUE`.
          Note that since the dtype of `matrix` is `np.int32`, we can't use `np.nan` as the fill value.
          In order to initialize from this storage, one must supply `matrix`, `i_min`, `x_min`.

        - `points` is a (N, 3) ndarray with every row being (iline, xline, height). Note that (iline, xline) are
          stored in cube coordinates that range from 0 to `ilines_len` and 0 to `xlines_len` respectively.
          Stored height is corrected on `time_delay` and `sample_rate` of the cube.
          In order to initialize from this storage, one must supply (N, 3) ndarray.

    Independently of type of initial storage, Horizon provides following:
        - Attributes `i_min`, `x_min`, `i_max`, `x_max`, `h_min`, `h_max`, `h_mean`, `h_std`, `bbox`,
          to completely describe location of the horizon in the 3D volume of the seismic cube.

        - Convenient methods of changing the horizon: `apply_to_matrix` and `apply_to_points`:
          these methods must be used instead of manually permuting `matrix` and `points` attributes.
          For example, filtration or smoothing of a horizon can be done with their help.

        - `Sampler` instance to generate points that are close to the horizon.
          Note that these points are scaled into [0, 1] range along each of the coordinate.

        - Method `add_to_mask` puts 1's on the location of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - `evaluate` allows to quickly assess the quality of a seismic reflection;
         for more metrics, check :class:`~.HorizonMetrics`.

        - A number of properties that describe geometrical, geological and mathematical characteristics of a horizon.
          For example, `borders_matrix` and `boundaries_matrix`: the latter containes outer and inner borders;
          `coverage` is the ratio between labeled traces and non-zero traces in the seismic cube;
          `solidity` is the ratio between labeled traces and traces inside the hull of the horizon;
          `perimeter` and `number_of_holes` speak for themselves.

        - Multiple instances of Horizon can be compared against one another and, if needed,
          merged into one (either in-place or not) via `check_proximity`, `overlap_merge`, `adjacent_merge` methods.

        - A wealth of visualization methods: view from above, slices along iline/xline axis, etc.
    """
    #pylint: disable=too-many-public-methods, import-outside-toplevel

    # CHARISMA: default seismic format of storing surfaces inside the 3D volume
    CHARISMA_SPEC = ['INLINE', '_', 'iline', 'XLINE', '__', 'xline', 'cdp_x', 'cdp_y', 'height']

    # REDUCED_CHARISMA: CHARISMA without redundant columns
    REDUCED_CHARISMA_SPEC = ['iline', 'xline', 'height']

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999

    def __init__(self, storage, geometry, name=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.format = None

        # Location of the horizon inside cube spatial range
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.bbox = None
        self._len = None

        # Underlying data storages
        self._matrix = None
        self._points = None
        self._depths = None

        # Heights information
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name
        self.cube_shape = geometry.cube_shape

        self.sampler = None

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


    @property
    def points(self):
        """ Storage of horizon data as (N, 3) array of (iline, xline, height) in cubic coordinates.
        If the horizon is created not from (N, 3) array, evaluated at the time of the first access.
        """
        if self._points is None and self.matrix is not None:
            points = self.matrix_to_points(self.matrix)
            points += np.array([self.i_min, self.x_min, 0])
            self._points = points
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.nonzero(matrix != Horizon.FILL_VALUE)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points


    @property
    def matrix(self):
        """ Storage of horizon data as depth map: matrix of (ilines_length, xlines_length) with each point
        corresponding to height. Matrix is shifted to a (i_min, x_min) point so it takes less space.
        If the horizon is created not from matrix, evaluated at the time of the first access.
        """
        if self._matrix is None and self.points is not None:
            self._matrix = self.points_to_matrix(self.points, self.i_min, self.x_min,
                                                 self.i_length, self.x_length)
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @staticmethod
    def points_to_matrix(points, i_min, x_min, i_length, x_length):
        """ Convert array of (N, 3) shape to a depth map (matrix) """
        matrix = np.full((i_length, x_length), Horizon.FILL_VALUE, np.int32)
        matrix[points[:, 0] - i_min, points[:, 1] - x_min] = points[:, 2]
        return matrix

    @property
    def depths(self):
        """ Array of depth only. Useful for faster stats computation when initialized from a matrix. """
        if self._depths is None:
            if self._points is not None:
                self._depths = self.points[:, -1]
            else:
                self._depths = self.matrix[self.matrix != self.FILL_VALUE]
        return self._depths


    @property
    def h_min(self):
        """ Minimum depth value. """
        if self._h_min is None:
            self._h_min = np.min(self.depths)
        return self._h_min

    @property
    def h_max(self):
        """ Maximum depth value. """
        if self._h_max is None:
            self._h_max = np.max(self.depths)
        return self._h_max

    @property
    def h_mean(self):
        """ Average depth value. """
        if self._h_mean is None:
            self._h_mean = np.mean(self.depths)
        return self._h_mean

    @property
    def h_std(self):
        """ Std of depths. """
        if self._h_std is None:
            self._h_std = np.std(self.depths)
        return self._h_std

    def __len__(self):
        """ Number of labeled traces. """
        if self._len is None:
            if self._points is not None:
                self._len = len(self.points)
            else:
                self._len = len(self.depths)
        return self._len

    def reset_storage(self, storage=None):
        """ Reset storage along with depth-wise stats."""
        self._depths = None
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None
        self._len = None

        if storage == 'matrix':
            self._matrix = None
        elif storage == 'points':
            self._points = None

    # Coordinate transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] -= self.geometry.ilines_offset
        array[:, 1] -= self.geometry.xlines_offset
        array[:, 2] -= self.geometry.delay
        array[:, 2] /= self.geometry.sample_rate
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array = array.astype(float)
        array[:, 0] += self.geometry.ilines_offset
        array[:, 1] += self.geometry.xlines_offset
        array[:, 2] *= self.geometry.sample_rate
        array[:, 2] += self.geometry.delay
        return array


    # Initialization from different containers
    def from_points(self, points, transform=False, verify=True, **kwargs):
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
        if verify:
            idx = np.where((points[:, 0] >= 0) &
                           (points[:, 1] >= 0) &
                           (points[:, 2] >= 0) &
                           (points[:, 0] < self.cube_shape[0]) &
                           (points[:, 1] < self.cube_shape[1]) &
                           (points[:, 2] < self.cube_shape[2]))[0]
            points = points[idx]
        self.points = np.rint(points).astype(np.int32)

        # Collect stats on separate axes. Note that depth stats are properties
        self.reset_storage('matrix')
        self.i_min, self.x_min, self._h_min = np.min(self.points, axis=0).astype(np.int32)
        self.i_max, self.x_max, self._h_max = np.max(self.points, axis=0).astype(np.int32)

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max]],
                             dtype=np.int32)


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


    def from_matrix(self, matrix, i_min, x_min, length=None, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        self.matrix = matrix
        self.i_min, self.x_min = i_min, x_min
        self.i_max, self.x_max = i_min + matrix.shape[0] - 1, x_min + matrix.shape[1] - 1

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max]],
                             dtype=np.int32)

        self.reset_storage('points')
        self._len = length


    def from_full_matrix(self, matrix, **kwargs):
        """ Init from matrix that covers the whole cube. """
        _ = kwargs
        self.from_matrix(matrix, 0, 0, **kwargs)


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
    def from_mask(mask, grid_info=None, geometry=None, shifts=None,
                  threshold=0.5, minsize=0, prefix='predict', **kwargs):
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
        if grid_info is not None:
            geometry = grid_info['geom']
            shifts = np.array([item[0] for item in grid_info['range']])

        if geometry is None or shifts is None:
            raise TypeError('Pass `grid_info` or `geometry` and `shifts` to `from_mask` method of Horizon creation.')

        # Labeled connected regions with an integer
        labeled = label(mask >= threshold)
        objects = find_objects(labeled)

        # Create an instance of Horizon for each separate region
        horizons = []
        for i, sl in enumerate(objects):
            max_possible_length = 1
            for j in range(3):
                max_possible_length *= sl[j].stop - sl[j].start

            if max_possible_length >= minsize:
                indices = np.nonzero(labeled[sl] == i + 1)

                if len(indices[0]) >= minsize:
                    coords = np.vstack([indices[i] + sl[i].start for i in range(3)]).T

                    points = groupby_mean(coords) + shifts
                    horizons.append(Horizon(points, geometry, name=f'{prefix}_{i}'))

        horizons.sort(key=len)
        return horizons


    # Functions to use to change the horizon
    def apply_to_matrix(self, function, **kwargs):
        """ Apply passed function to matrix storage.
        Automatically synchronizes the instance after.

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

        self.reset_storage('points') # applied to matrix, so we need to re-create points

    def apply_to_points(self, function, **kwargs):
        """ Apply passed function to points storage.
        Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to points storage directly.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        self.points = function(self.points, **kwargs)
        self.reset_storage('matrix') # applied to points, so we need to re-create matrix


    def filter_points(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from points storage."""
        filtering_matrix = filtering_matrix or self.geometry.zero_traces

        def filtering_function(points, **kwds):
            _ = kwds
            return _filtering_function(points, filtering_matrix)

        self.apply_to_points(filtering_function, **kwargs)

    def filter_matrix(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from matrix storage."""
        filtering_matrix = filtering_matrix or self.geometry.zero_traces
        idx_i, idx_x = np.asarray(filtering_matrix[self.i_min:self.i_max + 1,
                                                   self.x_min:self.x_max + 1] == 1).nonzero()

        def filtering_function(matrix, **kwds):
            _ = kwds
            matrix[idx_i, idx_x] = self.FILL_VALUE
            return matrix

        self.apply_to_matrix(filtering_function, **kwargs)


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

        self.apply_to_matrix(smoothing_function, **kwargs)


    # Horizon usage: point/mask generation
    def create_sampler(self, bins=None, quality_grid=None, **kwargs):
        """ Create sampler based on horizon location.

        Parameters
        ----------
        bins : sequence
            Size of ticks alongs each respective axis.
        """
        _ = kwargs
        default_bins = self.cube_shape // np.array([5, 20, 20])
        bins = bins if bins is not None else default_bins
        quality_grid = self.geometry.quality_grid if quality_grid is True else quality_grid or None

        if isinstance(quality_grid, np.ndarray):
            points = _filtering_function(np.copy(self.points), 1 - quality_grid)
        else:
            points = self.points

        self.sampler = HistoSampler(np.histogramdd(points/self.cube_shape, bins=bins))

    def show_sampler(self, n=100000, eps=3, show_unique=False, **kwargs):
        """ Generate a lot of points and look at their (iline, xline) positions.

        Parameters
        ----------
        n : int
            Number of points to generate.
        eps : int
            Window of painting.
        """
        show_sampler(self.sampler, None, self.geometry, n=n, eps=eps, show_unique=show_unique, **kwargs)


    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
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
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        mask_bbox = np.array([[locations[0][0], locations[0][-1]+1],
                              [locations[1][0], locations[1][-1]+1],
                              [locations[2][0], locations[2][-1]+1]],
                             dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        #TODO: add clear explanation about usage of advanced index in Horizon
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)
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

        cube_hdf5 = self.geometry.file_hdf5['cube_h']
        background = np.full((self.geometry.ilines_len, self.geometry.xlines_len, window), 0.0)

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
            data_chunk = cube_hdf5[h_start:h_end, :, :]
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

            # Remove traces that are not fully inside `window`
            mask = (heights + window <= (h_end - h_start))
            idx_i = idx_i[mask]
            idx_x = idx_x[mask]
            heights = heights[mask]

            # Subsequently add values from the cube to background, shift horizon 1 unit lower,
            # remove all heights that are bigger than can fit into background
            for j in range(window):
                background[idx_i, idx_x, np.full_like(heights, j)] = data_chunk[heights, idx_i, idx_x]
                heights += 1

                mask = heights < chunk_size
                idx_i = idx_i[mask]
                idx_x = idx_x[mask]
                heights = heights[mask]

        background[self.geometry.zero_traces == 1] = np.nan
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
            cube_hdf5 = self.geometry.file_hdf5['cube']
            slide_transform = lambda array: array

            hor_line = np.squeeze(self.matrix[line, :])
            background = np.zeros((self.geometry.xlines_len, window))
            idx_offset = self.x_min
            bad_traces = np.squeeze(self.geometry.zero_traces[line, :])

        elif orientation.startswith('x'):
            cube_hdf5 = self.geometry.file_hdf5['cube_x']
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

        slide = cube_hdf5[line, :, :]
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
        return len(self) / (np.prod(self.cube_shape[:2]) - np.sum(self.geometry.zero_traces))

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
    def evaluate(self, supports=20, plot=True, savepath=None, printer=print, **kwargs):
        """ Compute crucial metrics of a horizon. """
        msg = f"""
        Number of labeled points: {len(self)}
        Number of points inside borders: {np.sum(self.filled_matrix)}
        Perimeter (length of borders): {self.perimeter}
        Percentage of labeled non-bad traces: {self.coverage}
        Percentage of labeled traces inside borders: {self.solidity}
        Number of holes inside borders: {self.number_of_holes}
        """
        printer(dedent(msg))
        from .metrics import HorizonMetrics
        return HorizonMetrics(self).evaluate('support_corrs', supports=supports, agg='nanmean',
                                             plot=plot, savepath=savepath, **kwargs)


    def check_proximity(self, other, offset=0):
        """ Shortcut for :meth:`.HorizonMetrics.evaluate` to compare against the best match of list of horizons. """
        _, overlap_info = self.verify_merge(other)
        diffs = overlap_info['diffs'] + offset

        overlap_info = {
            **overlap_info,
            'mean': np.mean(diffs),
            'abs_mean': np.mean(np.abs(diffs)),
            'max': np.max(diffs),
            'abs_max': np.max(np.abs(diffs)),
            'std': np.std(diffs),
            'abs_std': np.std(np.abs(diffs)),
            'window_rate': np.mean(np.abs(diffs) < (5 / self.geometry.sample_rate)),
            'offset_diffs': diffs,
        }
        return overlap_info


    # Merge functions
    def verify_merge(self, other, mean_threshold=3.0, adjacency=0):
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
        adjacency : int
            Margin to consider horizons to be close (spatially).
        """
        overlap_info = {}

        # Overlap bbox
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        i_range = overlap_i_min - overlap_i_max
        x_range = overlap_x_min - overlap_x_max

        # Simplest possible check: horizon bboxes are too far from each other
        if i_range >= adjacency or x_range >= adjacency:
            merge_code = 1
            spatial_position = 'distant'
        else:
            merge_code = 2
            spatial_position = 'adjacent'

        # Compare matrices on overlap without adjacency:
        if merge_code != 1 and i_range < 0 and x_range < 0:
            self_overlap = self.matrix[overlap_i_min - self.i_min:overlap_i_max - self.i_min,
                                       overlap_x_min - self.x_min:overlap_x_max - self.x_min]

            other_overlap = other.matrix[overlap_i_min - other.i_min:overlap_i_max - other.i_min,
                                         overlap_x_min - other.x_min:overlap_x_max - other.x_min]

            self_mask = self_overlap != self.FILL_VALUE
            other_mask = other_overlap != self.FILL_VALUE
            mask = self_mask & other_mask
            diffs_on_overlap = self_overlap[mask] - other_overlap[mask]

            if len(diffs_on_overlap) == 0:
                # bboxes are overlapping, but horizons are not
                merge_code = 2
                spatial_position = 'adjacent'
            else:
                abs_diffs = np.abs(diffs_on_overlap)
                mean_on_overlap = np.mean(abs_diffs)
                if mean_on_overlap < mean_threshold:
                    merge_code = 3
                    spatial_position = 'overlap'
                else:
                    merge_code = 0
                    spatial_position = 'separated'

                overlap_info.update({'mean': mean_on_overlap,
                                     'diffs': diffs_on_overlap})

        overlap_info['spatial_position'] = spatial_position
        return merge_code, overlap_info


    def overlap_merge(self, other, inplace=False):
        """ Merge two horizons into one.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.
        """
        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Add both horizons to the background
        background[shared_self_i_min:shared_self_i_min+self.i_length,
                   shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Correct overlapping points
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        overlap_i_min -= shared_i_min
        overlap_i_max -= shared_i_min
        overlap_x_min -= shared_x_min
        overlap_x_max -= shared_x_min

        overlap = background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max]
        mask = overlap >= 0
        overlap[mask] //= 2
        overlap[~mask] -= self.FILL_VALUE
        background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max] = overlap

        background[background == 0] = self.FILL_VALUE
        length = len(self) + len(other) - mask.sum()
        # Create new instance or change `self`
        if inplace:
            # Change `self` inplace
            self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
            merged = True
        else:
            # Return a new instance of horizon
            merged = Horizon(background, self.geometry, self.name,
                             i_min=shared_i_min, x_min=shared_x_min, length=length)
        return merged


    def adjacent_merge(self, other, mean_threshold=3.0, adjacency=3, inplace=False):
        """ Check if adjacent merge (that is merge with some margin) is possible, and, if needed, merge horizons.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to merge.
        mean_threshold : number
            Height threshold for mean distances.
        adjacency : int
            Margin to consider horizons close (spatially).
        inplace : bool
            Whether to create new instance or update `self`.
        """
        # Simplest possible check: horizons are too far away from one another (depth-wise)
        overlap_h_min, overlap_h_max = max(self.h_min, other.h_min), min(self.h_max, other.h_max)
        if overlap_h_max - overlap_h_min < 0:
            return False

        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Put the second of the horizons on background
        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Enlarge the image to account for adjacency
        kernel = np.ones((3, 3), np.float32)
        dilated_background = cv2.dilate(background.astype(np.float32), kernel,
                                        iterations=adjacency).astype(np.int32)

        # Make counts: number of horizons in each point; create indices of overlap
        counts = (dilated_background != 0).astype(np.int32)
        counts[shared_self_i_min:shared_self_i_min+self.i_length,
               shared_self_x_min:shared_self_x_min+self.x_length] += 1
        counts_idx = counts == 2

        # Determine whether horizon can be merged (adjacent and height-close) or not
        mergeable = False
        if counts_idx.any():
            # Put the first horizon on dilated background, compute mean
            background[shared_self_i_min:shared_self_i_min+self.i_length,
                       shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

            # Compute diffs on overlap
            diffs = background[counts_idx] - dilated_background[counts_idx]
            diffs = np.abs(diffs)
            diffs = diffs[diffs < (-self.FILL_VALUE // 2)]

            if len(diffs) != 0 and np.mean(diffs) < mean_threshold:
                mergeable = True

        if mergeable:
            background[(background < 0) & (background != self.FILL_VALUE)] -= self.FILL_VALUE
            background[background == 0] = self.FILL_VALUE

            length = len(self) + len(other) # since there is no direct overlap

            # Create new instance or change `self`
            if inplace:
                # Change `self` inplace
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
                merged = True
            else:
                # Return a new instance of horizon
                merged = Horizon(background, self.geometry, self.name,
                                 i_min=shared_i_min, x_min=shared_x_min, length=length)
            return merged
        return False


    @staticmethod
    def merge_list(horizons, mean_threshold=2.0, adjacency=3, minsize=50):
        """ !!. """
        horizons = [horizon for horizon in horizons if len(horizon) >= minsize]

        # iterate over list of horizons to merge what can be merged
        i = 0
        flag = True
        while flag:
            # the procedure continues while at least a pair of horizons is mergeable
            flag = False
            while True:
                if i >= len(horizons):
                    break

                j = i + 1
                while True:
                    # attempt to merge each horizon to i-th horizon with fixed i
                    if j >= len(horizons):
                        break

                    merge_code, _ = Horizon.verify_merge(horizons[i], horizons[j],
                                                         mean_threshold=mean_threshold,
                                                         adjacency=adjacency)
                    if merge_code == 3:
                        merged = Horizon.overlap_merge(horizons[i], horizons[j], inplace=True)
                    elif merge_code == 2:
                        merged = Horizon.adjacent_merge(horizons[i], horizons[j], inplace=True,
                                                        mean_threshold=mean_threshold,
                                                        adjacency=adjacency)
                    else:
                        merged = False

                    if merged:
                        _ = horizons.pop(j)
                        flag = True
                    else:
                        j += 1
                i += 1
        return horizons



    def adjacent_merge_old(self, other, mean_threshold=3.0, adjacency=3,
                           check_only=False, force_merge=False, inplace=False):
        """ Collect stats on possible adjacent merge (that is merge with some margin), and, if needed, merge horizons.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to compare.
        mean_threshold : number
            Height threshold for mean distances.
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

        # Get heights on overlap
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
            max_on_shared_overlap = np.max(abs_diffs)

            if mean_on_shared_overlap < mean_threshold:
                mergeable = True

            adjacency_info.update({'mean': mean_on_shared_overlap,
                                   'max': max_on_shared_overlap,
                                   'diffs': diffs_on_shared_overlap})

        # Actually merge two horizons
        merged = None
        if check_only is False and mergeable:
            # Put the first horizon on background
            background[self_idx_i, self_idx_x] = self.matrix[self_idx_i_, self_idx_x_]

            # Values on overlap
            overlap_heights = np.where(other_heights != self.FILL_VALUE,
                                       np.rint((self_heights + other_heights) / 2), self_heights)

            background[so_idx_i, so_idx_x] = overlap_heights

            if inplace:
                # Change `self` inplace
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min)
            else:
                # Return a new instance of horizon
                merged = Horizon(background, self.geometry, self.name,
                                 i_min=shared_i_min, x_min=shared_x_min)

        return mergeable, merged, adjacency_info


    def dump(self, path, transform=None, add_height=True):
        """ Save horizon points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        values = self.cubic_to_lines(copy(self.points))
        values = values if transform is None else transform(values)

        df = pd.DataFrame(values, columns=self.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)

        path = path if not add_height else f'{path}_#{self.h_mean}'
        df.to_csv(path, sep=' ', columns=self.COLUMNS, index=False, header=False)


    # Methods of (visual) representation of a horizon
    def __repr__(self):
        return f"""<horizon {self.name} for {self.cube_name} at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.cube_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Heights range:     {self.h_min} to {self.h_max}
        Heights mean:      {self.h_mean:.6}
        Heights std:       {self.h_std:.6}

        Length:            {len(self)}
        Perimeter:         {self.perimeter}
        Coverage:          {self.coverage:3.5}
        Solidity:          {self.solidity:3.5}
        Num of holes:      {self.number_of_holes}
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
        plot_image(matrix, 'Depth map {} of `{}` on `{}`'.format('on full'*on_full, self.name, self.cube_name),
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
        amplitudes -= np.nanmin(amplitudes, axis=(0, 1)).reshape(1, 1, -1)
        amplitudes *= 1 / np.nanmax(amplitudes, axis=(0, 1)).reshape(1, 1, -1)
        amplitudes[self.full_matrix == self.FILL_VALUE, :] = np.nan
        amplitudes = amplitudes[:, :, ::-1]
        amplitudes *= np.asarray([1, 0.5, 0.25]).reshape(1, 1, -1)
        plot_image(amplitudes, 'RGB amplitudes of {} on cube {}'.format(self.name, self.cube_name),
                   rgb=True, **kwargs)


    def show_slide(self, loc, width=3, axis='i', order_axes=None, heights=None, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([len(item) for item in locations])

        # Load seismic and mask
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis)
        mask = np.zeros(shape)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)

        if heights:
            seismic_slide = seismic_slide[:, heights]
            mask = mask[:, heights]

        # Display everything
        header = self.geometry.index_headers[axis]
        title = f'{header} {loc} out of {self.geometry.lens[axis]}'
        meta_title = f'Horizon `{self.name}` on `{self.geometry.name}`'
        plot_images_overlap([seismic_slide, mask], title=title, order_axes=order_axes, meta_title=meta_title, **kwargs)



class StructuredHorizon(Horizon):
    """ Convenient alias for `Horizon` class. """


@njit
def _filtering_function(points, filtering_matrix):
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(points), dtype=np.int32)

    for i in range(len(points)):
        il, xl = points[i, 0], points[i, 1]
        if filtering_matrix[il, xl] == 1:
            mask[i] = 0
    return points[mask == 1, :]
