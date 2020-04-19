""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import sys
import logging
from textwrap import dedent
from random import random
from itertools import product
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import h5py
import segyio
import h5pickle

from .utils import lru_cache, find_min_max #, SafeIO
from .plot_utils import plot_images_overlap



class SpatialDescriptor:
    """ Allows to set names for parts of information about index.
    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    allows to get instance.lens[idx], where `idx` is position of `INLINE_3D` inside instance.index.

    Roughly equivalent to::
    @property
    def ilines_len(self):
        idx = self.index_headers.index('INLINE_3D')
        return self.lens[idx]
    """
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, header=None, attribute=None, name=None):
        self.header = header
        self.attribute = attribute

        if name is not None:
            self.name = name

    def __get__(self, obj, obj_class=None):
        # If attribute is already stored in object, just return it
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]

        # Find index of header, use it to access attr
        try:
            idx = obj.index_headers.index(self.header)
            return getattr(obj, self.attribute)[idx]
        except ValueError:
            raise ValueError(f'Current index does not contain {self.header}.')


def add_descriptors(cls):
    """ Add multiple descriptors to the decorated class.
    Name of each descriptor is `alias + postfix`.

    Roughly equivalent to::
    ilines = SpatialDescriptor('INLINE_3D', 'uniques', 'ilines')
    xlines = SpatialDescriptor('CROSSLINE_3D', 'uniques', 'xlines')

    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    xlines_len = SpatialDescriptor('CROSSLINE_3D', 'lens', 'xlines_len')
    etc
    """
    attrs = ['uniques', 'offsets', 'lens']  # which attrs hold information
    postfixes = ['', '_offset', '_len']     # postfix of current attr

    headers = ['INLINE_3D', 'CROSSLINE_3D'] # headers to use
    aliases = ['ilines', 'xlines']          # alias for header

    for attr, postfix in zip(attrs, postfixes):
        for alias, header in zip(aliases, headers):
            name = alias + postfix
            descriptor = SpatialDescriptor(header=header, attribute=attr, name=name)
            setattr(cls, name, descriptor)
    return cls



@add_descriptors
class SeismicGeometry:
    """ This class selects which type of geometry to initialize:the SEG-Y or the HDF5 one,
    depending on the passed path.

    Independent of exact format, `SeismicGeometry` provides following:
        - Attributes to describe shape and structure of the cube, as well as exact values of file-wide headers,
          for example, `time_delay` and `sample_rate`.

        - Ability to infer information about the cube amplitudes:
          `trace_container` attribute contains examples of amplitudes inside the cube and allows to compute statistics.

        - If needed, spatial stats can also be inferred: attributes `min_matrix`, `mean_matrix`, etc
          allow to create a complete spatial map (that is view from above) of the desired statistic for the whole cube.
          `hist_matrix` contains a histogram of values for each trace in the cube, and can be used as
          a proxy for amplitudes in each trace for evaluating aggregated statistics.

        - `load_slide` (2D entity) or `load_crop` (3D entity) methods to load data from the cube.
          Load slides takes a number of slide and axis to cut along; makes use of `lru_cache` to work
          faster for subsequent loads. Cache is bound for each instance.
          Load crops works off of complete location specification (3D slice).

        - `quality_map` attribute is a spatial matrix that assess cube hardness;
          `quality_grid` attribute contains a grid of locations to train model on, based on `quality_map`.

        - `show_slide` method allows to do exactly what the name says, and has the same API, as `load_slide`.
          `repr` allows to get a summary of the cube statistics.

    Refer to the documentation of respective classes to learn about more their structure, attributes and methods.
    """
    #TODO: add separate class for cube-like labels
    SEGY_ALIASES = ['sgy', 'segy', 'seg']
    HDF5_ALIASES = ['hdf5', 'h5py']

    # Attributes to store during SEG-Y -> HDF5 conversion
    PRESERVED = [
        'depth', 'delay', 'sample_rate',
        'byte_no', 'offsets', 'ranges', 'lens', # `uniques` can't be saved due to different lenghts of arrays
        'value_min', 'value_max', 'q01', 'q99', 'q001', 'q999', 'bins', 'trace_container',
        'ilines', 'xlines', 'ilines_offset', 'xlines_offset', 'ilines_len', 'xlines_len',
        'zero_traces', 'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix', 'hist_matrix',
        '_quality_map',
    ]

    # Headers to load from SEG-Y cube
    HEADERS_PRE_FULL = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE', 'CDP', 'CDP_TRACE', 'offset', ]
    HEADERS_POST_FULL = ['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']
    HEADERS_POST = ['INLINE_3D', 'CROSSLINE_3D']

    # Headers to use as id of a trace
    INDEX_PRE = ['FieldRecord', 'TraceNumber']
    INDEX_POST = ['INLINE_3D', 'CROSSLINE_3D']
    INDEX_CDP = ['CDP_Y', 'CDP_X']

    def __new__(cls, path, *args, **kwargs):
        """ Select the type of geometry based on file extension. """
        _ = args, kwargs
        fmt = os.path.splitext(path)[1][1:]

        if fmt in cls.SEGY_ALIASES:
            new_cls = SeismicGeometrySEGY
        elif fmt in cls.HDF5_ALIASES:
            new_cls = SeismicGeometryHDF5
        else:
            raise TypeError('Unknown format of the cube.')

        instance = super().__new__(new_cls)
        return instance

    def __init__(self, path, *args, process=True, **kwargs):
        _ = args
        self.path = path

        # Names of different lengths and format: helpful for outside usage
        self.name = os.path.basename(self.path)
        self.short_name = self.name.split('.')[0]
        self.long_name = ':'.join(self.path.split('/')[-2:])
        self.format = os.path.splitext(self.path)[1][1:]

        self._quality_map = None
        self._quality_grid = None

        self.has_stats = False
        if process:
            self.process(**kwargs)


    def scaler(self, array, mode='minmax'):
        """ Normalize array of amplitudes cut from the cube.

        Parameters
        ----------
        array : ndarray
            Crop of amplitudes.
        mode : str
            If `minmax`, then data is scaled to [0, 1] via minmax scaling.
            If `q` or `normalize`, then data is divided by the maximum of absolute values of the
            0.01 and 0.99 quantiles.
            If `q_clip`, then data is clipped to 0.01 and 0.99 quantiles and then divided by the
            maximum of absolute values of the two.
        """
        if mode in ['q', 'normalize']:
            return array / max(abs(self.q01), abs(self.q99))
        if mode in ['q_clip']:
            return np.clip(array, self.q01, self.q99) / max(abs(self.q01), abs(self.q99))
        if mode == 'minmax':
            scale = (self.value_max - self.value_min)
            return (array - self.value_min) / scale
        raise ValueError('Wrong mode', mode)


    def parse_axis(self, axis):
        """ Convert string representation of an axis into integer, if needed. """
        if isinstance(axis, str):
            if axis in self.index_headers:
                axis = self.index_headers.index(axis)
            elif axis in ['i', 'il', 'iline']:
                axis = 0
            elif axis in ['x', 'xl', 'xline']:
                axis = 1
            elif axis in ['h', 'height', 'depth']:
                axis = 2
        return axis


    def make_slide_locations(self, loc, axis=0):
        """ Create locations (sequence of locations for each axis) for desired slide along desired axis. """
        axis = self.parse_axis(axis)

        locations = [np.arange(item) for item in self.lens]
        locations += [np.arange(self.depth)]
        locations[axis] = [loc]
        return locations


    # Spatial matrices
    @lru_cache(100)
    def get_quantile_matrix(self, q):
        """ Restore the quantile matrix for desired `q` from `hist_matrix`.

        Parameters
        ----------
        q : number
            Quantile to compute. Must be in (0, 1) range.
        """
        #pylint: disable=line-too-long
        threshold = self.depth * q
        cumsums = np.cumsum(self.hist_matrix, axis=-1)

        positions = np.argmax(cumsums >= threshold, axis=-1)
        idx_1, idx_2 = np.nonzero(positions)
        indices = positions[idx_1, idx_2]

        broadcasted_bins = np.broadcast_to(self.bins, (*positions.shape, len(self.bins)))

        q_matrix = np.zeros_like(positions, dtype=np.float)
        q_matrix[idx_1, idx_2] += broadcasted_bins[idx_1, idx_2, indices]
        q_matrix[idx_1, idx_2] += (broadcasted_bins[idx_1, idx_2, indices+1] - broadcasted_bins[idx_1, idx_2, indices]) * \
                                   (threshold - cumsums[idx_1, idx_2, indices-1]) / self.hist_matrix[idx_1, idx_2, indices]
        q_matrix[q_matrix == 0.0] = np.nan
        setattr(self, f'q{int(q*100)}_matrix', q_matrix)
        return q_matrix


    @property
    def quality_map(self):
        """ Spatial matrix to show harder places in the cube. """
        if self._quality_map is None:
            self.make_quality_map([0.1], ['support_js', 'support_hellinger'])
        return self._quality_map

    def make_quality_map(self, quantiles, metric_names, **kwargs):
        """ Create `quality_map` matrix that shows harder places in the cube.

        Parameters
        ----------
        quantiles : sequence of floats
            Quantiles for computing hardness thresholds. Must be in (0, 1) ranges.
        metric_names : sequence or str
            Metrics to compute to assess hardness of cube.
        """
        from .metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_map = GeometryMetrics(self).evaluate('quality_map', quantiles=quantiles, agg=None,
                                                     metric_names=metric_names, **kwargs)
        self._quality_map = quality_map
        return quality_map


    @property
    def quality_grid(self):
        """ Spatial grid based on `quality_map`. """
        if self._quality_grid is None:
            self.make_quality_grid((20, 150))
        return self._quality_grid

    def make_quality_grid(self, frequencies, iline=True, xline=True, margin=0, **kwargs):
        """ Create `quality_grid` based on `quality_map`.

        Parameters
        ----------
        frequencies : sequence of numbers
            Grid frequencies for individual levels of hardness in `quality_map`.
        margin : int
            Margin of boundaries to not include in the grid.
        iline, xline : bool
            Whether to make lines in grid to account for `ilines`/`xlines`.
        """
        from .metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_grid = GeometryMetrics(self).make_grid(self.quality_map, frequencies,
                                                       iline=iline, xline=xline, margin=margin, **kwargs)
        self._quality_grid = quality_grid
        return quality_grid


    # Instance introspection and visualization methods
    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        attrs = [
            'dataframe', 'trace_container', 'zero_traces',
            *[attr for attr in self.__dict__
              if 'matrix' in attr or '_quality' in attr],
        ]
        return sum(sys.getsizeof(getattr(self, attr)) for attr in attrs if hasattr(self, attr))

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)

    def __repr__(self):
        return 'Inferred geometry for {}: ({}x{}x{})'.format(os.path.basename(self.path), *self.cube_shape)

    def __str__(self):
        msg = f"""
        Geometry for cube              {self.path}
        Current index:                 {self.index_headers}
        Shape:                         {self.cube_shape}
        Time delay and sample rate:    {self.delay}, {self.sample_rate}
        Depth of one trace is:         {self.depth}
        Size of the instance:          {self.ngbytes:4.3} GB
        """

        if self.has_stats:
            msg += f"""
        Num of unique amplitudes:      {len(np.unique(self.trace_container))}
        Mean/std of amplitudes:        {np.mean(self.trace_container):6.6}/{np.std(self.trace_container):6.6}
        Min/max amplitudes:            {self.value_min:6.6}/{self.value_max:6.6}
        q01/q99 amplitudes:            {self.q01:6.6}/{self.q99:6.6}
            """
        return dedent(msg)

    def log(self, printer=None):
        """ Log some info into desired stream. """
        if not callable(printer):
            path_log = '/'.join(self.path.split('/')[:-1]) + '/CUBE_INFO.log'
            handler = logging.FileHandler(path_log, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))

            logger = logging.getLogger('geometry_logger')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            printer = logger.info
        printer(str(self))

    def show_slide(self, loc=None, start=None, end=None, step=1, axis=0, stable=True, order_axes=None, **kwargs):
        """ Load slide in `segy` or `hdf5` fashion and display it. """
        axis = self.parse_axis(axis)
        slide = self.load_slide(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)

        title = f'{self.index_headers[axis]} {loc} out of {self.lens[axis]}'
        meta_title = ''
        plot_images_overlap([slide], title=title, order_axes=order_axes, meta_title=meta_title, **kwargs)

    def show_amplitude_hist(self, scaler=None, bins=50):
        """ Show distribution of amplitudes in `trace_container`. Optionally applies chosen `scaler`. """
        import matplotlib.pyplot as plt #pylint: disable=import-outside-toplevel
        data = np.copy(self.trace_container)
        if scaler:
            data = self.scaler(data, mode=scaler)

        title = f'Amplitude distribution for {self.short_name}\nMean/std: {np.mean(data):3.3}/{np.std(data):3.3}'
        plt.figure()
        _ = plt.hist(data.ravel(), bins=bins)
        plt.title(title)
        plt.show()


class SeismicGeometrySEGY(SeismicGeometry):
    """ Class to infer information about SEG-Y cubes and provide convenient methods of working with them.
    A wrapper around `segyio` to provide higher-level API.

    In order to initialize instance, one must supply `path`, `headers` and `index`:
        - `path` is a location of SEG-Y file
        - `headers` is a sequence of trace headers to infer from the file
        - `index_headers` is a subset of `headers` that is used as trace (unique) identifier:
          for example, `INLINE_3D` and `CROSSLINE_3D` has a one-to-one correspondance with trace numbers.
          Another example is `FieldRecord` and `TraceNumber`.

    Each instance is basically built around `dataframe` attribute, which describes mapping from
    indexing headers to trace numbers. It is used to, for example, get all trace indices from a desired `FieldRecord`.
    `set_index` method can be called to change indexing headers of the dataframe.

    One can add stats to the instance by calling `collect_stats` method, that makes a full pass through
    the cube in order to analyze distribution of amplitudes. It also collects a number of trace examples
    into `trace_container` attribute, that can be used for later evaluation of various statistics.
    """
    #pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
    def __init__(self, path, headers=None, index_headers=None, **kwargs):
        self.structured = False
        self.dataframe = None
        self.segyfile = None

        self.headers = headers or self.HEADERS_POST
        self.index_headers = index_headers or self.INDEX_POST

        super().__init__(path, **kwargs)


    # Methods of inferring dataframe and amplitude stats
    def process(self, collect_stats=False, **kwargs):
        """ Create dataframe based on `segy` file headers. """
        # Note that all the `segyio` structure inference is disabled
        # self.segyfile = SafeIO(self.path, opener=segyio.open, mode='r', strict=False, ignore_geometry=True)
        self.segyfile = segyio.open(self.path, mode='r', strict=False, ignore_geometry=True)
        self.segyfile.mmap()

        self.depth = len(self.segyfile.trace[0])
        self.delay = self.segyfile.header[0].get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = segyio.dt(self.segyfile) / 1000

        # Load all the headers
        dataframe = {}
        for column in self.headers:
            dataframe[column] = self.segyfile.attributes(getattr(segyio.TraceField, column))[slice(None)]

        dataframe = pd.DataFrame(dataframe)
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'index': 'trace_index'}, inplace=True)
        self.dataframe = dataframe.set_index(self.index_headers)

        self.add_attributes()
        if collect_stats:
            self.collect_stats(**kwargs)

    def set_index(self, index_headers, sortby=None):
        """ Change current index to a subset of loaded headers. """
        self.dataframe.reset_index(inplace=True)
        if sortby:
            self.dataframe.sort_values(index_headers, inplace=True, kind='mergesort')# the only stable sorting algorithm
        self.dataframe.set_index(index_headers, inplace=True)
        self.index_headers = index_headers
        self.add_attributes()

    def add_attributes(self):
        """ Infer info about curent index from `dataframe` attribute. """
        self.index_len = len(self.index_headers)
        self._zero_trace = np.zeros(self.depth)

        # Unique values in each of the indexing column
        self.unsorted_uniques = [np.unique(self.dataframe.index.get_level_values(i).values)
                                 for i in range(self.index_len)]
        self.uniques = [np.sort(item) for item in self.unsorted_uniques]
        self.uniques_inversed = [{v: j for j, v in enumerate(self.uniques[i])}
                                 for i in range(self.index_len)]

        self.byte_no = [getattr(segyio.TraceField, h) for h in self.index_headers]
        self.offsets = [np.min(item) for item in self.uniques]
        self.lens = [len(item) for item in self.uniques]
        self.ranges = [(np.max(item) - np.min(item) + 1) for item in self.uniques]

        self.cube_shape = np.asarray([*self.lens, self.depth])

    def collect_stats(self, spatial=True, bins=25, num_keep=15000, **kwargs):
        """ Pass through file data to collect stats:
            - min/max values.
            - q01/q99 quantiles of amplitudes in the cube.
            - certain amount of traces are stored to `trace_container` attribute.

        If `spatial` is True, makes an additional pass through the cube to obtain following:
            - min/max/mean/std for every trace - `min_matrix`, `max_matrix` and so on.
            - histogram of values for each trace: - `hist_matrix`.
            - bins for histogram creation: - `bins`.

        Parameters
        ----------
        spatial : bool
            Whether to collect additional stats.
        bins : int or str
            Number of bins or name of automatic algorithm of defining number of bins.
        num_keep : int
            Number of traces to store.
        """
        #pylint: disable=not-an-iterable
        _ = kwargs

        num_traces = len(self.segyfile.header)

        # Get min/max values, store some of the traces
        trace_container = []
        value_min, value_max = np.inf, -np.inf

        for i in tqdm(range(num_traces), desc='Finding min/max', ncols=1000):
            trace = self.segyfile.trace[i]

            trace_min, trace_max = find_min_max(trace)
            if trace_min < value_min:
                value_min = trace_min
            if trace_max > value_max:
                value_max = trace_max

            if random() < (num_keep / num_traces) and trace_min != trace_max:
                trace_container.extend(trace.tolist())
                #TODO: add dtype for storing

        # Collect more spatial stats: min, max, mean, std, histograms matrices
        if spatial:
            # Make bins
            bins = np.histogram_bin_edges(None, bins, range=(value_min, value_max)).astype(np.float)
            self.bins = bins

            # Create containers
            min_matrix, max_matrix = np.full(self.lens, np.nan), np.full(self.lens, np.nan)
            hist_matrix = np.full((*self.lens, len(bins)-1), np.nan)

            # Iterate over traces
            description = f'Collecting stats for {self.name}'
            for i in tqdm(range(num_traces), desc=description, ncols=1000):
                trace = self.segyfile.trace[i]
                header = self.segyfile.header[i]

                # i -> id in a dataframe
                keys = [header.get(field) for field in self.byte_no]
                store_key = [self.uniques_inversed[j][item] for j, item in enumerate(keys)]
                store_key = tuple(store_key)

                # for each trace, we store an entire histogram of amplitudes
                val_min, val_max = find_min_max(trace)
                min_matrix[store_key] = val_min
                max_matrix[store_key] = val_max

                if val_min != val_max:
                    histogram = np.histogram(trace, bins=bins)[0]
                    hist_matrix[store_key] = histogram

            # Restore stats from histogram
            midpoints = (bins[1:] + bins[:-1]) / 2
            probs = hist_matrix / np.sum(hist_matrix, axis=-1, keepdims=True)

            mean_matrix = np.sum(probs * midpoints, axis=-1)
            std_matrix = np.sqrt(np.sum((np.broadcast_to(midpoints, (*mean_matrix.shape, len(midpoints))) - \
                                            mean_matrix.reshape(*mean_matrix.shape, 1))**2 * probs,
                                        axis=-1))

            # Store everything into instance
            self.min_matrix, self.max_matrix = min_matrix, max_matrix
            self.mean_matrix, self.std_matrix = mean_matrix, std_matrix
            self.hist_matrix = hist_matrix
            self.zero_traces = (min_matrix == max_matrix).astype(np.int)
            self.zero_traces[np.isnan(min_matrix)] = 1

        self.value_min, self.value_max = value_min, value_max
        self.trace_container = np.array(trace_container)
        self.q001, self.q01, self.q99, self.q999 = np.quantile(trace_container, [0.001, 0.01, 0.99, 0.999])
        self.has_stats = True


    # Methods to load actual data from SEG-Y
    def load_trace(self, index):
        """ Load individual trace from segyfile.
        If passed `np.nan`, returns trace of zeros.
        """
        if not np.isnan(index):
            return self.segyfile.trace.raw[int(index)]
        return self._zero_trace

    def load_traces(self, trace_indices):
        """ Stack multiple traces together. """
        return np.stack([self.load_trace(idx) for idx in trace_indices])

    @lru_cache(128, attributes='index_headers')
    def load_slide(self, loc=None, axis=0, start=None, end=None, step=1, stable=True):
        """ Create indices and load actual traces for one slide.

        If the current index is 1D, then slide is defined by `start`, `end`, `step`.
        If the current index is 2D, then slide is defined by `loc` and `axis`.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        start, end, step : ints
            Parameters of slice loading for 1D index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        indices = self.make_slide_indices(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)
        slide = self.load_traces(indices)
        return slide


    def make_slide_indices(self, loc=None, axis=0, start=None, end=None, step=1, stable=True, return_iterator=False):
        """ Choose appropriate version of index creation for various lengths of current index.

        Parameters
        ----------
        start, end, step : ints
            Parameters of slice loading for 1d index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        return_iterator : bool
            Whether to also return the same iterator that is used to index current `dataframe`.
            Can be useful for subsequent loads from the same place in various instances.
        """
        if self.index_len == 1:
            _ = loc, axis
            result = self.make_slide_indices_1d(start=start, end=end, step=step, stable=stable,
                                                return_iterator=return_iterator)
        elif self.index_len == 2:
            _ = start, end, step
            result = self.make_slide_indices_2d(loc=loc, axis=axis, stable=stable,
                                                return_iterator=return_iterator)
        elif self.index_len == 3:
            raise NotImplementedError('Yet to be done!')
        else:
            raise ValueError('Index lenght must be less than 4. ')
        return result

    def make_slide_indices_1d(self, start=None, end=None, step=1, stable=True, return_iterator=False):
        """ 1D version of index creation. """
        start = start or self.offsets[0]
        end = end or self.uniques[0][-1]

        if stable:
            iterator = self.dataframe.index[(self.dataframe.index >= start) & (self.dataframe.index <= end)]
            iterator = iterator.values[::step]
        else:
            iterator = np.arange(start, end+1, step)

        indices = self.dataframe['trace_index'].get(iterator, np.nan).values

        if return_iterator:
            return indices, iterator
        return indices

    def make_slide_indices_2d(self, loc, axis=0, stable=True, return_iterator=False):
        """ 2D version of index creation. """
        other_axis = 1 - axis
        location = self.uniques[axis][loc]

        if stable:
            others = self.dataframe[self.dataframe.index.get_level_values(axis) == location]
            others = others.index.get_level_values(other_axis).values
        else:
            others = self.uniques[other_axis]

        iterator = list(zip([location] * len(others), others) if axis == 0 else zip(others, [location] * len(others)))
        indices = self.dataframe['trace_index'].get(iterator, np.nan).values

        #TODO: keep only uniques, when needed, with `nan` filtering
        if stable:
            indices = np.unique(indices)

        if return_iterator:
            return indices, iterator
        return indices


    def _load_crop(self, locations):
        """ Load 3D crop from the cube.

        Parameters
        ----------
        locations : sequence of arrays
            List of desired locations to load: along the first index, the second, and depth.

        Example
        -------
        If the current index is `INLINE_3D` and `CROSSLINE_3D`, then to load
        5:110 ilines, 100:1105 crosslines, 0:700 depths, locations must be::
            [
                np.arange(5, 110),
                np.arange(100, 1105),
                np.arange(0, 700)
            ]
        """
        shape = np.array([len(item) for item in locations])
        indices = self.make_crop_indices(locations)
        crop = self.load_traces(indices)[..., locations[-1]].reshape(shape)
        return crop

    def make_crop_indices(self, locations):
        """ Create indices for 3D crop loading. """
        iterator = list(product(*[[self.uniques[idx][i] for i in locations[idx]] for idx in range(2)]))
        indices = self.dataframe['trace_index'].get(list(iterator), np.nan).values
        return np.unique(indices)

    def load_crop(self, locations, threshold=10, mode=None, **kwargs):
        """ Smart choice between using :meth:`._load_crop` and stacking multiple slides created by :meth:`.load_slide`.
        """
        _ = kwargs
        shape = np.array([len(item) for item in locations])
        mode = mode or ('slide' if min(shape) < threshold else 'crop')

        if mode == 'slide':
            axis = np.argmin(shape)
            #TODO: add depth-slicing; move this logic to separate function
            if axis in [0, 1]:
                return np.stack([self.load_slide(loc, axis=axis)[..., locations[-1]]
                                 for loc in locations[axis]],
                                axis=axis)
        return self._load_crop(locations)

    # Convert SEG-Y to HDF5
    def make_hdf5(self, path_hdf5=None, postfix=''):
        """ Converts `.segy` cube to `.hdf5` format.

        Parameters
        ----------
        path_hdf5 : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        """
        if self.index_headers != self.INDEX_POST:
            # Currently supports only INLINE/CROSSLINE cubes
            raise TypeError(f'Current index must be {self.INDEX_POST}')

        path_hdf5 = path_hdf5 or (os.path.splitext(self.path)[0] + postfix + '.hdf5')

        # Remove file, if exists: h5py can't do that
        if os.path.exists(path_hdf5):
            os.remove(path_hdf5)

        # Create file and datasets inside
        with h5py.File(path_hdf5, "a") as file_hdf5:
            cube_hdf5 = file_hdf5.create_dataset('cube', self.cube_shape)
            cube_hdf5_x = file_hdf5.create_dataset('cube_x', self.cube_shape[[1, 2, 0]])
            cube_hdf5_h = file_hdf5.create_dataset('cube_h', self.cube_shape[[2, 0, 1]])

            # Default projection: (ilines, xlines, depth)
            # Depth-projection: (depth, ilines, xlines)
            pbar = tqdm(total=self.ilines_len + self.xlines_len, ncols=1000)

            pbar.set_description(f'Converting {self.long_name}; ilines projection')
            for i in range(self.ilines_len):
                slide = self.load_slide(i, stable=False)
                cube_hdf5[i, :, :] = slide.reshape(1, self.xlines_len, self.depth)
                cube_hdf5_h[:, i, :] = slide.T
                pbar.update()

            # xline-oriented projection: (xlines, depth, ilines)
            pbar.set_description(f'Converting {self.long_name} to hdf5; xlines projection')
            for x in range(self.xlines_len):
                slide = self.load_slide(x, axis=1, stable=False).T
                cube_hdf5_x[x, :, :,] = slide
                pbar.update()
            pbar.close()

            # Save all the necessary attributes to the `info` group
            for attr in self.PRESERVED:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    file_hdf5['/info/' + attr] = getattr(self, attr)



class SeismicGeometryHDF5(SeismicGeometry):
    """ Class to infer information about HDF5 cubes and provide convenient methods of working with them.

    In order to initialize instance, one must supply `path` to the HDF5 cube.

    All the attributes are loaded directly from HDF5 file itself, so most of the attributes from SEG-Y file
    are preserved, with the exception of `dataframe` and `uniques`.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, path, **kwargs):
        self.structured = True
        self.file_hdf5 = None

        super().__init__(path, **kwargs)

    def process(self, **kwargs):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        _ = kwargs
        # self.file_hdf5 = SafeIO(self.path, opener=h5pickle.File, mode='r')
        self.file_hdf5 = h5pickle.File(self.path, mode='r')
        self.add_attributes()

    def add_attributes(self):
        """ Store values from `hdf5` file to attributes. """
        self.index_headers = self.INDEX_POST

        for item in self.PRESERVED:
            try:
                value = self.file_hdf5['/info/' + item][()]
                setattr(self, item, value)
            except KeyError:
                pass
        # BC
        self.ilines_offset = min(self.ilines)
        self.xlines_offset = min(self.xlines)
        self.ilines_len = len(self.ilines)
        self.xlines_len = len(self.xlines)
        self.cube_shape = np.asarray([self.ilines_len, self.xlines_len, self.depth])
        self.has_stats = True

    # Methods to load actual data from HDF5
    def load_crop(self, locations, axis=None, **kwargs):
        """ Load 3D crop from the cube.
        Automatically chooses the fastest axis to use: as `hdf5` files store multiple copies of data with
        various orientations, some axis are faster than others depending on exact crop location and size.

        Parameters
        locations : sequence of arrays
            List of desired locations to load: along the first index, the second, and depth.
        axis : str or int
            Identificator of the axis to use to load data.
            Can be `iline`, `xline`, `height`, `depth`, `i`, `x`, `h`, 0, 1, 2.
        """
        _ = kwargs

        if axis is None:
            shape = np.array([len(item) for item in locations])
            axis = np.argmin(shape)
        else:
            mapping = {0: 0, 1: 1, 2: 2,
                       'i': 0, 'x': 1, 'h': 2,
                       'iline': 0, 'xline': 1, 'height': 2, 'depth': 2}
            axis = mapping[axis]

        if axis == 1 and 'cube_x' in self.file_hdf5:
            crop = self._load_x(*locations)
        elif axis == 2 and 'cube_h' in self.file_hdf5:
            crop = self._load_h(*locations)
        else: # backward compatibility
            crop = self._load_i(*locations)
        return crop

    def _load_i(self, ilines, xlines, heights):
        cube_hdf5 = self.file_hdf5['cube']
        dtype = cube_hdf5.dtype
        return np.stack([self._cached_load(cube_hdf5, iline)[xlines, :][:, heights]
                         for iline in ilines]).astype(dtype)

    def _load_x(self, ilines, xlines, heights):
        cube_hdf5 = self.file_hdf5['cube_x']
        dtype = cube_hdf5.dtype
        return np.stack([self._cached_load(cube_hdf5, xline)[heights, :][:, ilines].transpose([1, 0])
                         for xline in xlines], axis=1).astype(dtype)

    def _load_h(self, ilines, xlines, heights):
        cube_hdf5 = self.file_hdf5['cube_h']
        dtype = cube_hdf5.dtype
        return np.stack([self._cached_load(cube_hdf5, height)[ilines, :][:, xlines]
                         for height in heights], axis=2).astype(dtype)

    @lru_cache(128)
    def _cached_load(self, cube, loc):
        """ Load one slide of data from a certain cube projection.
        Caches the result in a thread-safe manner.
        """
        return cube[loc, :, :]

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis. """
        _ = kwargs
        location = self.make_slide_locations(loc=loc, axis=axis)
        return np.squeeze(self.load_crop(location))
