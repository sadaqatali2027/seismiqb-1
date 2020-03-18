""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import sys
import logging
from textwrap import dedent
from random import random
from itertools import product
from tqdm import tqdm, tqdm_notebook

import numpy as np
import pandas as pd
import h5py
import h5pickle
import segyio

from .utils import update_minmax, lru_cache, find_min_max
from .plot_utils import plot_images_overlap





class SpatialDescriptor:
    """ !!. """
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, header=None, attribute=None, name=None, ):
        self.header = header
        self.attribute = attribute

        if name is not None:
            self.name = name

    def __get__(self, obj, obj_class=None):
        #
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]

        #
        try:
            idx = obj.index.index(self.header)
            return getattr(obj, self.attribute)[idx]
        except ValueError:
            raise ValueError(f'Current index does not contain {self.header}.')


def add_descriptors(cls):
    """ !!. """
    attrs = ['vals', 'offsets', 'lens', 'uniques']
    postfixes = ['', '_offset', '_len', '_unique']

    aliases = ['ilines', 'xlines']
    headers = ['INLINE_3D', 'CROSSLINE_3D']

    for attr, postfix in zip(attrs, postfixes):
        for alias, header in zip(aliases, headers):
            name = alias+postfix
            descriptor = SpatialDescriptor(header=header, attribute=attr, name=name)
            setattr(cls, name, descriptor)
    return cls

@add_descriptors
class SeismicGeometry:
    """ !!. """
    #pylint: disable=attribute-defined-outside-init
    SEGY_ALIASES = ['sgy', 'segy', 'seg']
    HDF5_ALIASES = ['hdf5', 'h5py']

    PRESERVED = [
        'depth', 'delay', 'sample_rate',
        'fields', 'offsets', 'uniques', 'lens', # vals can't be saved due to different lenghts of arrays
        'value_min', 'value_max', 'q01', 'q99', 'trace_container',
        'ilines', 'xlines', 'ilines_offset', 'xlines_offset',
        'ilines_len', 'xlines_len', 'ilines_unique', 'xlines_unique',
        'zero_traces',
    ]

    HEADERS_PRE_FULL = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE', 'CDP', 'CDP_TRACE', 'offset', ]
    HEADERS_POST_FULL = ['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']

    HEADERS_POST = ['INLINE_3D', 'CROSSLINE_3D']

    INDEX_PRE = ['FieldRecord', 'TraceNumber']
    INDEX_POST = ['INLINE_3D', 'CROSSLINE_3D']
    INDEX_CDP = ['CDP_Y', 'CDP_X']

    def __init__(self, path, process=True, headers=None, index=None, **kwargs):
        self.path = path
        self.name = os.path.basename(self.path)
        self.long_name = '/'.join(self.path.split('/')[-2:])
        self.format = os.path.splitext(self.path)[1][1:]

        if process:
            self.process(headers, index, **kwargs)

    # Methods that wrap around SEGY-Y/H5PY
    def process(self, headers=None, index=None, **kwargs):
        """ !!. """
        if self.format in self.SEGY_ALIASES:
            self.structured = False
            self.dataframe = None
            self.depth = None

            self.headers = headers or self.HEADERS_POST
            self.index = index or self.INDEX_POST
            self.process_sgy(**kwargs)

        elif self.format in self.HDF5_ALIASES:
            self.structured = True
            self.process_h5py(**kwargs)

    def load_crop(self, location, axis=None, mode=None, threshold=10):
        """ !!. """
        if self.structured:
            _ = mode, threshold
            return self.load_h5py(location, axis=axis)

        _ = axis
        return self.load_sgy(location, mode=mode, threshold=threshold)

    def load_slide(self, loc, axis=0, stable=False):
        """ !!. """
        if self.structured:
            _ = stable
            return self.load_slide_h5py(loc, axis=axis)

        return self.load_slide_sgy(loc, axis=axis, stable=stable)

    def show_slide(self, loc, axis=0, stable=False, order_axes=None, **kwargs):
        """ !!. """
        slide = self.load_slide(loc=loc, axis=axis, stable=stable)

        title = f'{self.index[axis]} {loc} out of {self.lens[axis]}'
        meta_title = ''
        plot_images_overlap([slide], title=title, order_axes=order_axes, meta_title=meta_title, **kwargs)

    # SEG-Y methods: infer dataframe, attributes, load data from file
    def process_sgy(self, collect_stats=False, **kwargs):
        """ !!. """
        # !!.
        self.segyfile = segyio.open(self.path, mode='r', strict=False, ignore_geometry=True)
        self.segyfile.mmap()

        #
        self.depth = len(self.segyfile.trace[0])
        self.delay = self.segyfile.header[0].get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = segyio.dt(self.segyfile) / 1000

        #
        dataframe = {}
        for column in self.headers:
            dataframe[column] = self.segyfile.attributes(getattr(segyio.TraceField, column))[slice(None)]

        dataframe = pd.DataFrame(dataframe)
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'index': 'trace_index'}, inplace=True)
        self.dataframe = dataframe.set_index(self.index)

        #
        self.add_attributes_sgy()
        if collect_stats:
            self.collect_stats_sgy(**kwargs)

    def set_index(self, index, sortby=None):
        """ !!. """
        self.dataframe.reset_index(inplace=True)
        if sortby:
            self.dataframe.sort_values(index, inplace=True, kind='mergesort') # the only stable sorting algorithm
        self.dataframe.set_index(index, inplace=True)
        self.index = index
        self.add_attributes_sgy()

    def add_attributes_sgy(self):
        """ !!. """
        self._zero_trace = np.zeros(self.depth)
        # Attributes
        self.vals = [np.sort(np.unique(self.dataframe.index.get_level_values(i).values)) for i in range(2)]
        self.unsorted_vals = [np.unique(self.dataframe.index.get_level_values(i).values) for i in range(2)]

        self.fields = [getattr(segyio.TraceField, self.index[i]) for i in range(2)]
        self.offsets = [np.min(item) for item in self.vals]
        self.uniques = [len(item) for item in self.vals]
        self.lens = [(np.max(item) - np.min(item) + 1) for item in self.vals]

        self.cube_shape = np.asarray([*self.lens, self.depth])

    def collect_stats_sgy(self, spatial=True, bins='uniform', num_keep=15000, **kwargs):
        """ !!. """
        _ = kwargs

        num_traces = len(self.segyfile.header)

        # Get min/max values, store some of the traces
        trace_container = []
        value_min, value_max = np.inf, -np.inf

        for i in tqdm_notebook(range(num_traces), desc='Finding min/max', ncols=1000):
            trace = self.segyfile.trace[i]

            val_min, val_max = find_min_max(trace)
            if val_min < value_min:
                value_min = val_min
            if val_max > value_max:
                value_max = val_max

            if random() < (num_keep / num_traces) and val_min != val_max:
                trace_container.extend(trace.tolist())


        # Collect more spatial stats: min, max, mean, std, histograms matrices
        if spatial:
            # Make bins
            bins = np.histogram_bin_edges(None, 25, range=(value_min, value_max)).astype(np.float)
            self.bins = bins

            # Create containers
            min_matrix, max_matrix = np.full(self.lens, np.nan), np.full(self.lens, np.nan)
            hist_matrix = np.full((*self.lens, len(bins)-1), np.nan)

            # Iterate over traces
            description = f'Collecting stats for {self.name}'
            for i in tqdm_notebook(range(num_traces), desc=description, ncols=1000):

                header = self.segyfile.header[i]
                idx_1 = header.get(self.fields[0])
                idx_2 = header.get(self.fields[1])

                trace = self.segyfile.trace[i]
                val_min, val_max = find_min_max(trace)

                min_matrix[idx_1 - self.offsets[0], idx_2 - self.offsets[1]] = val_min
                max_matrix[idx_1 - self.offsets[0], idx_2 - self.offsets[1]] = val_max

                if val_min != val_max:
                    histogram = np.histogram(trace, bins=bins)[0]
                    hist_matrix[idx_1 - self.offsets[0], idx_2 - self.offsets[1], :] = histogram

            # Restore stats from histogram
            midpoints = (bins[1:] + bins[:-1]) / 2
            probs = hist_matrix / np.sum(hist_matrix, axis=-1, keepdims=True)

            mean_matrix = np.sum(probs * midpoints, axis=-1)
            std_matrix = np.sqrt(np.sum((np.broadcast_to(midpoints, (*mean_matrix.shape, len(midpoints))) - \
                                            mean_matrix.reshape(*mean_matrix.shape, 1))**2 * probs,
                                        axis=-1))

            # Store everything into instance
            self.min_matrix, self.max_matrix = min_matrix, max_matrix
            self.zero_traces = (min_matrix == max_matrix).astype(np.int)
            self.hist_matrix = hist_matrix
            self.mean_matrix, self.std_matrix = mean_matrix, std_matrix

        self.value_min, self.value_max = value_min, value_max
        self.trace_container = np.array(trace_container)
        self.q01, self.q99 = np.quantile(trace_container, [0.01, 0.99])

    # Methods to load actual data from SEG-Y
    @lru_cache(16384, classwide=False, attributes='index')
    def load_trace_sgy(self, index):
        """ !!. """
        if not np.isnan(index):
            return self.segyfile.trace.raw[int(index)]
        return np.zeros(self.depth)
        # return self._zero_trace

    def load_traces_sgy(self, trace_indices, heights=None):
        """ !!. """
        heights = slice(None) if heights is None else heights
        return np.stack([self.load_trace_sgy(idx) for idx in trace_indices])[..., heights]

    @lru_cache(128, classwide=False, attributes='index')
    def load_slide_sgy(self, loc, heights=None, axis=0, stable=False):
        """ !!. """
        indices = self.make_slide_indices(loc, axis=axis, stable=stable)
        slide = self.load_traces_sgy(indices, heights)
        return slide

    def make_slide_indices(self, loc, axis=0, stable=False, return_iterator=False):
        """ !!. """
        other_axis = 1 - axis
        location = self.vals[axis][loc]

        if stable:
            others = self.dataframe[self.dataframe.index.get_level_values(axis) == location]
            others = others.index.get_level_values(other_axis).values
        else:
            others = self.vals[other_axis]

        iterator = list(zip([location] * len(others), others) if axis == 0 else zip(others, [location] * len(others)))
        indices = self.dataframe['trace_index'].get(iterator, np.nan).values

        #TODO: keep only uniques, when needed, with `nan` filtering
        if stable:
            # indices = indices
            indices = np.unique(indices)

        if return_iterator:
            return indices, iterator
        return indices

    def load_crop_sgy(self, locations):
        """ !!. """
        shape = np.array([len(item) for item in locations])
        indices = self.make_crop_indices(locations)
        crop = self.load_traces_sgy(indices, locations[-1]).reshape(shape)
        return crop

    def make_crop_indices(self, locations):
        """ !!. """
        iterator = list(product(*[[self.vals[idx][i] for i in locations[idx]] for idx in range(2)]))
        indices = self.dataframe['trace_index'].get(list(iterator), np.nan).values
        return np.unique(indices)

    def load_sgy(self, locations, threshold=10, mode=None):
        """ Smart choice between using :meth:`.load_crop_sgy` and :meth:`.load_slide_sgy`. """
        shape = np.array([len(item) for item in locations])
        mode = mode or ('slide' if min(shape) < threshold else 'crop')

        if mode == 'slide':
            axis = np.argmin(shape)
            if axis in [0, 1]:
                return np.stack([self.load_slide_sgy(loc, axis=axis)[..., locations[-1]]
                                 for loc in locations[axis]],
                                axis=axis)
        return self.load_crop_sgy(locations)

    # H5PY methods: convert from SEG-Y, process cube and attributes, load data
    def make_h5py(self, path_h5py=None, postfix='', dtype=np.float32, create_projections=True):
        """ Converts `.sgy` cube to `.hdf5` format.

        Parameters
        ----------
        path_h5py : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        dtype : str
            data-type to use for storing the cube. Has to be supported by numpy.
        """
        if self.index != self.INDEX_POST:
            raise TypeError(f'Current index must be {self.INDEX_POST}')
        if self.format not in self.SEGY_ALIASES:
            raise TypeError(f'Format should be in {self.SEGY_ALIASES}')

        path_h5py = path_h5py or (os.path.splitext(self.path)[0] + postfix + '.hdf5')

        # Remove file, if exists: h5py can't do that
        if os.path.exists(path_h5py):
            os.remove(path_h5py)

        # Create file and datasets inside
        h5py_file = h5py.File(path_h5py, "a")
        cube_h5py = h5py_file.create_dataset('cube', self.cube_shape)
        if create_projections:
            cube_h5py_x = h5py_file.create_dataset('cube_x', self.cube_shape[[1, 2, 0]])
            cube_h5py_h = h5py_file.create_dataset('cube_h', self.cube_shape[[2, 0, 1]])

        # Default projection: (ilines, xlines, depth)
        pbar = tqdm_notebook(total=self.ilines_len + create_projections * (self.ilines_len + self.xlines_len),
                             ncols=1000)

        pbar.set_description(f'Converting {self.long_name}; ilines projection')
        for i in range(self.ilines_len):
            #
            slide = self.load_slide_sgy(i).reshape(1, self.xlines_len, self.depth).astype(dtype)
            cube_h5py[i, :, :] = slide
            pbar.update()

        if create_projections:
            # xline-oriented projection: (xlines, depth, ilines)
            pbar.set_description(f'Converting {self.long_name} to h5py; xlines projection')
            for x in range(self.xlines_len):
                slide = self.load_slide_sgy(x, axis=1).T.astype(dtype)
                cube_h5py_x[x, :, :,] = slide
                pbar.update()

            # depth-oriented projection: (depth, ilines, xlines)
            pbar.set_description(f'Converting {self.long_name} to h5py; depth projection')
            for i in range(self.ilines_len):
                slide = self.load_slide_sgy(i).T.astype(dtype)
                cube_h5py_h[:, i, :] = slide
                pbar.update()
        pbar.close()

        # Save all the necessary attributes to the `info` group
        for attr in self.PRESERVED:
            if hasattr(self, attr):
                h5py_file['/info/' + attr] = getattr(self, attr)

        h5py_file.close()

        self.h5py_file = h5py.File(path_h5py, "r")
        self.add_attributes_h5py()
        self.structured = True


    def process_h5py(self, **kwargs):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        _ = kwargs
        self.h5py_file = h5pickle.File(self.path, "r")
        self.add_attributes_h5py()

    def add_attributes_h5py(self):
        """ !!. """
        self.index = self.INDEX_POST

        for item in self.PRESERVED:
            try:
                value = self.h5py_file['/info/' + item][()]
                setattr(self, item, value)
            except KeyError:
                pass

        # BC
        self.ilines_offset = min(self.ilines)
        self.xlines_offset = min(self.xlines)
        self.ilines_len = len(self.ilines)
        self.xlines_len = len(self.xlines)
        self.cube_shape = np.asarray([self.ilines_len, self.xlines_len, self.depth])

    # Methods to load actual data from H5PY
    def load_h5py(self, locations, axis=None):
        """ !!. """
        if axis is None:
            shape = np.array([len(item) for item in locations])
            axis = np.argmin(shape)
        else:
            mapping = {0: 0, 1: 1, 2: 2,
                       'i': 0, 'x': 1, 'h': 2,
                       'iline': 0, 'xline': 1, 'height': 2, 'depth': 2}
            axis = mapping[axis]

        if axis == 1 and 'cube_x' in self.h5py_file:
            crop = self._load_h5py_x(*locations)
        elif axis == 2 and 'cube_h' in self.h5py_file:
            crop = self._load_h5py_h(*locations)
        else: # backward compatibility
            crop = self._load_h5py_i(*locations)
        return crop

    def _load_h5py_i(self, ilines, xlines, heights):
        h5py_cube = self.h5py_file['cube']
        dtype = h5py_cube.dtype
        return np.stack([self._load_slide_h5py(h5py_cube, iline)[xlines, :][:, heights]
                         for iline in ilines]).astype(dtype)

    def _load_h5py_x(self, ilines, xlines, heights):
        h5py_cube = self.h5py_file['cube_x']
        dtype = h5py_cube.dtype
        return np.stack([self._load_slide_h5py(h5py_cube, xline)[heights, :][:, ilines].transpose([1, 0])
                         for xline in xlines]).astype(dtype)

    def _load_h5py_h(self, ilines, xlines, heights):
        h5py_cube = self.h5py_file['cube_h']
        dtype = h5py_cube.dtype
        return np.stack([self._load_slide_h5py(h5py_cube, height)[ilines, :][:, xlines]
                         for height in heights]).astype(dtype)

    @lru_cache(128, classwide=False)
    def _load_slide_h5py(self, cube, loc):
        """ !!. """
        return cube[loc, :, :]

    def load_slide_h5py(self, loc, axis='iline'):
        """ !!. """
        location = self.make_slide_locations(loc=loc, axis=axis)
        return np.squeeze(self.load_h5py(location))

    # Common methods/properties for SEG-Y/hdf5
    def scaler(self, array, mode='minmax'):
        """ !!. """
        if mode == 'minmax':
            scale = (self.value_max - self.value_min)
            return (array - self.value_min) / scale
        raise ValueError('Wrong mode')

    def descaler(self, array, mode='minmax'):
        """ !!. """
        if mode == 'minmax':
            scale = (self.value_max - self.value_min)
            return array * scale + self.value_min
        raise ValueError('Wrong mode')


    def make_slide_locations(self, loc, axis=0, return_axis=False):
        """ !!. """
        locations = [np.arange(item) for item in self.uniques]
        locations += [np.arange(self.depth)]

        if isinstance(axis, str):
            if axis in self.index:
                axis = self.index.index(axis)
            elif axis in ['i', 'il', 'iline']:
                axis = 0
            elif axis in ['x', 'xl', 'xline']:
                axis = 1
            elif axis in ['h', 'height', 'depth']:
                axis = 2
        locations[axis] = [loc]

        if return_axis:
            return locations, axis
        return locations


    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        attrs = [
            'dataframe',
            'trace_container', 'min_matrix', 'max_matrix',
            'mean_matrix', 'std_matrix', 'zero_traces', 'hist_matrix'
        ]
        return sum(sys.getsizeof(getattr(self, attr)) for attr in attrs if hasattr(self, attr))

    @property
    def nmbytes(self):
        """ Size of instance in megabytes. """
        return self.nbytes / (1024**2)

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)


    @lru_cache(100, classwide=False)
    def get_quantile_matrix(self, q):
        """ !!. """
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

    # Visualization methods
    def __repr__(self):
        return 'Inferred geometry for {}: ({}x{}x{})'.format(os.path.basename(self.path), *self.cube_shape)

    def __str__(self):
        msg = f"""
        Geometry for cube {self.path}
        Time delay and sample rate: {self.delay}, {self.sample_rate}
        Depth of one trace is: {self.depth}
        Current index: {self.index}
        Shape: {self.cube_shape}
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




class OldSeismicGeometry:
    """ Class to hold information about seismic cubes.
    There are two supported formats: `.sgy` and `.hdf5`.
    Method `make_h5py` converts from former to latter.

    For either supported format method `load` allows to gather following information:
        Time delay and sample rate of traces.
        Both spatial-wise and depth-wise lengths of cube.
        Sorted lists of ilines and xlines that present in the cube, their respective lengths and offsets.
        Sorted lists of global X and Y coordinates that present in the cube.
        Mapping from global coordinates to xlines/ilines.
        Mapping from cube values to [0, 1] and vice versa.

    If cube is in `.sgy` format, then `il_xl_trace` dictionary is inferred.
    If cube is in `.hdf5` format, then `h5py_file` contains file handler.
    """
    #pylint: disable=too-many-instance-attributes
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path)
        self.h5py_file = None

        self.il_xl_trace = {}
        self.delay, self.sample_rate = None, None
        self.depth = None
        self.cube_shape = None

        self.ilines, self.xlines = set(), set()
        self.ilines_offset, self.xlines_offset = None, None
        self.ilines_len, self.xlines_len = None, None

        self.value_min, self.value_max = np.inf, -np.inf
        self.scaler, self.descaler = None, None
        self.zero_traces = None


    def load(self):
        """ Load of information about the file. """
        if not isinstance(self.path, str):
            raise ValueError('Path to a cube should be supplied!')

        ext = os.path.splitext(self.path)[1][1:]
        if ext in ['sgy', 'segy']:
            self._load_sgy()
        elif ext in ['hdf5']:
            self._load_h5py()
        else:
            raise TypeError('Format should be `sgy` or `hdf5`')

        # More useful variables
        self.ilines_offset = min(self.ilines)
        self.xlines_offset = min(self.xlines)
        self.ilines_len = len(self.ilines)
        self.xlines_len = len(self.xlines)
        self.cube_shape = np.asarray([self.ilines_len, self.xlines_len, self.depth])

        # Create transform to correct height with time-delay and sample rate
        #pylint: disable=attribute-defined-outside-init
        self.ilines_transform = lambda array: array - self.ilines_offset
        self.xlines_transform = lambda array: array - self.xlines_offset
        self.height_transform = lambda array: (array - self.delay) / self.sample_rate

        self.ilines_reverse = lambda array: array + self.ilines_offset
        self.xlines_reverse = lambda array: array + self.xlines_offset
        self.height_reverse = lambda array: array * self.sample_rate + self.delay

        # Callable to transform cube values to [0, 1] (and vice versa)
        if self.value_min is not None:
            scale = (self.value_max - self.value_min)
            self.scaler = lambda array: (array - self.value_min) / scale
            self.descaler = lambda array: array*scale + self.value_min


    def _load_sgy(self):
        """ Actual parsing of .sgy-file.
        Does one full path through the file for collecting all the
        necessary information.
        """
        with segyio.open(self.path, 'r', strict=False) as segyfile:
            segyfile.mmap() # makes operations faster

            self.depth = len(segyfile.trace[0])

            first_header = segyfile.header[0]
            first_iline = first_header.get(segyio.TraceField.INLINE_3D)
            first_xline = first_header.get(segyio.TraceField.CROSSLINE_3D)

            last_header = segyfile.header[-1]
            last_iline = last_header.get(segyio.TraceField.INLINE_3D)
            last_xline = last_header.get(segyio.TraceField.CROSSLINE_3D)

            ilines_offset, xlines_offset = first_iline, first_xline
            ilines_len = last_iline - first_iline + 1
            xlines_len = last_xline - first_xline + 1

            matrix = np.zeros((ilines_len, xlines_len))

            description = 'Working with {}'.format(os.path.basename(self.path))
            for i in tqdm(range(len(segyfile.header)), desc=description):
                header = segyfile.header[i]
                iline = header.get(segyio.TraceField.INLINE_3D)
                xline = header.get(segyio.TraceField.CROSSLINE_3D)

                # Map:  (iline, xline) -> index of trace
                self.il_xl_trace[(iline, xline)] = i

                # Set: all possible values for ilines/xlines
                self.ilines.add(iline)
                self.xlines.add(xline)

                # Gather stats: min/max value, location of zero-traces
                trace = segyfile.trace[i]
                self.value_min, self.value_max, matrix = update_minmax(trace, self.value_min, self.value_max,
                                                                       matrix, iline, xline,
                                                                       ilines_offset, xlines_offset)

        self.ilines = sorted(list(self.ilines))
        self.xlines = sorted(list(self.xlines))
        self.delay = header.get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = header.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL) / 1000
        self.zero_traces = matrix


    def _load_h5py(self):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        self.h5py_file = h5pickle.File(self.path, "r")
        attributes = ['depth', 'delay', 'sample_rate', 'value_min', 'value_max',
                      'ilines', 'xlines', 'zero_traces']

        for item in attributes:
            value = self.h5py_file['/info/' + item][()]
            setattr(self, item, value)

    def make_h5py(self, path_h5py=None, postfix='', dtype=np.float32):
        """ Converts `.sgy` cube to `.hdf5` format.

        Parameters
        ----------
        path_h5py : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        dtype : str
            data-type to use for storing the cube. Has to be supported by numpy.
        """
        if os.path.splitext(self.path)[1][1:] not in ['sgy', 'segy']:
            raise TypeError('Format should be `sgy`')
        path_h5py = path_h5py or (os.path.splitext(self.path)[0] + postfix + '.hdf5')

        # Recreate file. h5py can't do that
        if os.path.exists(path_h5py):
            os.remove(path_h5py)

        h5py_file = h5py.File(path_h5py, "a")
        cube_h5py = h5py_file.create_dataset('cube', self.cube_shape)
        cube_h5py_x = h5py_file.create_dataset('cube_x', self.cube_shape[[1, 2, 0]])
        cube_h5py_h = h5py_file.create_dataset('cube_h', self.cube_shape[[2, 0, 1]])

        # Copy traces from .sgy to .h5py
        with segyio.open(self.path, 'r', strict=False) as segyfile:
            segyfile.mmap()

            description = 'Converting {} to h5py'.format('/'.join(self.path.split('/')[-2:]))
            for il_ in tqdm(range(self.ilines_len), desc=description):
                slide = np.zeros((1, self.xlines_len, self.depth))
                iline = self.ilines[il_]

                for xl_ in range(self.xlines_len):
                    xline = self.xlines[xl_]
                    tr_ = self.il_xl_trace.get((iline, xline))
                    if tr_ is not None:
                        trace = segyfile.trace[tr_]
                        slide[0, xl_, :] = trace

                slide = slide.astype(dtype)
                cube_h5py[il_, :, :] = slide

            for xl_ in tqdm(range(self.xlines_len), desc='x_view'):
                slide = np.zeros((self.depth, self.ilines_len))
                xline = self.xlines[xl_]

                for il_ in range(self.ilines_len):
                    iline = self.ilines[il_]
                    tr_ = self.il_xl_trace.get((iline, xline))
                    if tr_ is not None:
                        trace = segyfile.trace[tr_]
                        slide[:, il_] = trace

                slide = slide.astype(dtype)
                cube_h5py_x[xl_, :, :,] = slide

            for il_ in tqdm(range(self.ilines_len), desc='h_view'):
                slide = np.zeros((self.depth, self.xlines_len))
                iline = self.ilines[il_]

                for xl_ in range(self.xlines_len):
                    xline = self.xlines[xl_]
                    tr_ = self.il_xl_trace.get((iline, xline))
                    if tr_ is not None:
                        trace = segyfile.trace[tr_]
                        slide[:, xl_] = trace

                slide = slide.astype(dtype)
                cube_h5py_h[:, il_, :] = slide

        # Save all the necessary attributes to the `info` group
        attributes = ['depth', 'delay', 'sample_rate', 'value_min', 'value_max',
                      'ilines', 'xlines', 'zero_traces']

        for item in attributes:
            h5py_file['/info/' + item] = getattr(self, item)

        h5py_file.close()
        self.h5py_file = h5py.File(path_h5py, "r")


    def __repr__(self):
        return 'Inferred geometry for {}: ({}x{}x{})'.format(os.path.basename(self.path), *self.cube_shape)

    def __str__(self):
        return ('''Geometry for cube: {}
                   Time delay and sample rate: {}, {}
                   Number of ilines: {}
                   Number of xlines: {}
                   Depth of one trace is: {}
                   ilines range from {} to {}
                   xlines range from {} to {}'''
                .format(self.path, self.delay, self.sample_rate, *self.cube_shape,
                        min(self.ilines), max(self.ilines),
                        min(self.xlines), max(self.xlines))
                )

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
