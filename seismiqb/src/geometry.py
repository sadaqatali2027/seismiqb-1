""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import logging

import h5py
import numpy as np
import segyio
from tqdm import tqdm



class SeismicGeometry():
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
    def __init__(self, path):
        self.path = path
        self.h5py_file = None

        self.il_xl_trace = {}
        self.delay, self.sample_rate = None, None
        self.depth = None
        self.cube_shape = None

        self.ilines, self.xlines = set(), set()
        self.ilines_offset, self.xlines_offset = None, None
        self.ilines_len, self.xlines_len = None, None

        self.cdp_x, self.cdp_y = set(), set()
        self.abs_to_lines = None

        self.value_min, self.value_max = np.inf, -np.inf
        self.scaler, self.descaler = None, None


    def load(self):
        """ Load a lot of information about the file. """
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
        self.cube_shape = [self.ilines_len, self.xlines_len, self.depth]

        # Create transform from global coordinates to ilines/xlines/depth
        transform_y = self._get_linear(self.cdp_y, self.ilines)
        transform_x = self._get_linear(self.cdp_x, self.xlines)
        transform_h = lambda h: ((h - self.delay) / self.sample_rate).astype(np.int64)
        self.abs_to_lines = (lambda array: np.stack([transform_y(array[:, 0]),
                                                     transform_x(array[:, 1]),
                                                     transform_h(array[:, 2])],
                                                    axis=-1))

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

            description = 'Working with {}'.format('/'.join(self.path.split('/')[-2:]))
            for i in tqdm(range(len(segyfile.header)), desc=description):
                header_ = segyfile.header[i]
                iline_ = header_.get(segyio.TraceField.INLINE_3D)
                xline_ = header_.get(segyio.TraceField.CROSSLINE_3D)
                cdp_x_ = header_.get(segyio.TraceField.CDP_X)
                cdp_y_ = header_.get(segyio.TraceField.CDP_Y)

                # Map:  (iline, xline) -> index of trace
                self.il_xl_trace[(iline_, xline_)] = i

                # Set: all possible values for ilines/xlines
                self.ilines.add(iline_)
                self.xlines.add(xline_)
                self.cdp_x.add(cdp_x_)
                self.cdp_y.add(cdp_y_)

        self.ilines = sorted(list(self.ilines))
        self.xlines = sorted(list(self.xlines))
        self.cdp_x = sorted(list(self.cdp_x))
        self.cdp_y = sorted(list(self.cdp_y))
        self.delay = header_.get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = header_.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL) // 1000


    def _load_h5py(self):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        self.h5py_file = h5py.File(self.path, "r")

        self.depth = self.h5py_file['/info/depth'][()]
        self.delay = self.h5py_file['/info/delay'][()]
        self.sample_rate = self.h5py_file['/info/sample_rate'][()]

        self.value_min = self.h5py_file['/info/value_min'][()]
        self.value_max = self.h5py_file['/info/value_max'][()]

        self.ilines = self.h5py_file['/info/ilines'][:]
        self.xlines = self.h5py_file['/info/xlines'][:]
        self.cdp_x = self.h5py_file['/info/cdp_x'][:]
        self.cdp_y = self.h5py_file['/info/cdp_y'][:]


    def _get_linear(self, set_x, set_y):
        """ Get linear-transformation that maps range of set_x into range of set_y. """
        a = (max(set_y) - min(set_y)) / (max(set_x) - min(set_x))
        b = max(set_y) - a * max(set_x)
        return lambda x: a * x + b


    def make_scalers(self, mode):
        """ Get scaling constants. """
        count = len(self.il_xl_trace)
        if mode == 'full':
            traces = np.arange(count)
        if mode == 'random':
            traces = np.random.choice(count, count//10)

        with segyio.open(self.path, 'r', strict=False) as segyfile:
            segyfile.mmap()

            description = 'Making scalers for {}'.format('/'.join(self.path.split('/')[-2:]))
            for i in tqdm(traces, desc=description):
                trace_ = segyfile.trace[i]

                if np.min(trace_) < self.value_min:
                    self.value_min = np.min(trace_)
                if np.max(trace_) > self.value_max:
                    self.value_max = np.max(trace_)


    def make_h5py(self, path_h5py=None):
        """ Converts `.sgy` cube to `.hdf5` format.
        By default, new cube is stored right next to original.
        """
        if os.path.splitext(self.path)[1][1:] not in ['sgy', 'segy']:
            raise TypeError('Format should be `sgy`')
        path_h5py = path_h5py or (os.path.splitext(self.path)[0] + '.hdf5')

        # Recreate file. h5py can't do that
        if os.path.exists(path_h5py):
            os.remove(path_h5py)

        h5py_file = h5py.File(path_h5py, "a")
        cube_h5py = h5py_file.create_dataset('cube', self.cube_shape)

        # Copy traces from .sgy to .h5py
        with segyio.open(self.path, 'r') as segyfile:
            segyfile.mmap()

            description = 'Converting {} to h5py'.format('/'.join(self.path.split('/')[-2:]))
            for il_ in tqdm(range(self.ilines_len), desc=description):
                slide = np.zeros((1, self.xlines_len, self.depth))

                for xl_ in range(self.xlines_len):
                    iline = self.ilines[il_]
                    xline = self.xlines[xl_]
                    tr_ = self.il_xl_trace[(iline, xline)]
                    slide[0, xl_, :] = segyfile.trace[tr_]
                cube_h5py[il_, :, :] = slide

        # Move all the necessary attributes to the `info` group
        h5py_file['/info/depth'] = self.depth
        h5py_file['/info/delay'] = self.delay
        h5py_file['/info/sample_rate'] = self.sample_rate

        h5py_file['/info/value_min'] = self.value_min
        h5py_file['/info/value_max'] = self.value_max

        h5py_file['/info/ilines'] = self.ilines
        h5py_file['/info/xlines'] = self.xlines
        h5py_file['/info/cdp_x'] = self.cdp_x
        h5py_file['/info/cdp_y'] = self.cdp_y

        self.h5py_file = h5py.File(path_h5py, "r")



    def log(self, path_log=None):
        """ Log some info. """
        path_log = path_log or ('/'.join(self.path.split('/')[:-1]) + '/CUBE_INFO.log')
        handler = logging.FileHandler(path_log, mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger('geometry_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info('Info for cube: {}'.format(self.path))

        logger.info('Depth of one trace is: {}'.format(self.depth))
        logger.info('Time delay: {}'.format(self.delay))
        logger.info('Sample rate: {}'.format(self.sample_rate))

        logger.info('Number of ILINES: {}'.format(self.ilines_len))
        logger.info('Number of XLINES: {}'.format(self.xlines_len))

        logger.info('ILINES range from {} to {}'.format(min(self.ilines), max(self.ilines)))
        logger.info('ILINES range from {} to {}'.format(min(self.xlines), max(self.xlines)))

        logger.info('CDP_X range from {} to {}'.format(min(self.cdp_x),
                                                       max(self.cdp_x)))
        logger.info('CDP_X range from {} to {}'.format(min(self.cdp_y),
                                                       max(self.cdp_y)))
