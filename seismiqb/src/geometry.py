""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import logging

import h5py
import numpy as np
import segyio
from tqdm import tqdm

from .utils import update_minmax



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
        self.height_correction = None
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
        self.cube_shape = [self.ilines_len, self.xlines_len, self.depth]

        # Create transform to correct height with time-delay and sample rate
        def transform(array):
            return np.stack([array[:, 0],
                             array[:, 1],
                             (array[:, 2] - self.delay) / self.sample_rate,
                             array[:, 3]],
                            axis=-1)
        self.height_correction = transform

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
        self.h5py_file = h5py.File(self.path, "r")
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

        # Copy traces from .sgy to .h5py
        with segyio.open(self.path, 'r', strict=False) as segyfile:
            segyfile.mmap()

            description = 'Converting {} to h5py'.format('/'.join(self.path.split('/')[-2:]))
            for il_ in tqdm(range(self.ilines_len), desc=description):
                slide = np.zeros((1, self.xlines_len, self.depth))

                for xl_ in range(self.xlines_len):
                    iline = self.ilines[il_]
                    xline = self.xlines[xl_]
                    tr_ = self.il_xl_trace.get((iline, xline))
                    if tr_ is not None:
                        slide[0, xl_, :] = segyfile.trace[tr_]
                cube_h5py[il_, :, :] = slide.astype(dtype)

        # Save all the necessary attributes to the `info` group
        attributes = ['depth', 'delay', 'sample_rate', 'value_min', 'value_max',
                      'ilines', 'xlines', 'zero_traces']

        for item in attributes:
            h5py_file['/info/' + item] = getattr(self, item)

        h5py_file.close()
        self.h5py_file = h5py.File(path_h5py, "r")


    def log(self, printer=None):
        """ Log some info. """
        if not callable(printer):
            path_log = '/'.join(self.path.split('/')[:-1]) + '/CUBE_INFO.log'
            handler = logging.FileHandler(path_log, mode='w')
            handler.setFormatter(logging.Formatter('%(message)s'))

            logger = logging.getLogger('geometry_logger')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            printer = logger.info

        printer('Info for cube: {}'.format(self.path))

        printer('Depth of one trace is: {}'.format(self.depth))
        printer('Time delay: {}'.format(self.delay))
        printer('Sample rate: {}'.format(self.sample_rate))

        printer('Number of ILINES: {}'.format(self.ilines_len))
        printer('Number of XLINES: {}'.format(self.xlines_len))

        printer('ILINES range from {} to {}'.format(min(self.ilines), max(self.ilines)))
        printer('ILINES range from {} to {}'.format(min(self.xlines), max(self.xlines)))
