""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import logging

import numpy as np
import segyio
from tqdm import tqdm



class SeismicGeometry():
    """ Class to hold information about .sgy-file. """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, path):
        self.path = path
        self.il_xl_trace = {}
        self.delay, self.sample_rate = None, None
        self.value_min, self.value_max = np.inf, -np.inf
        self.scaler, self.descaler = None, None
        self.depth = None

        self.x_to_xline, self.y_to_iline = {}, {}
        self.ilines, self.xlines = set(), set()
        self.ilines_offset, self.xlines_offset = None, None
        self.ilines_len, self.xlines_len = None, None
        self.cube_shape = None

        self.cdp_x, self.cdp_y = set(), set()
        self.abs_to_lines = None


    def load(self):
        """ Actual parsing of .sgy-file.
        Does one full path through the file for collecting all the
        necessary information, including:
            `il_xl_trace` dictionary for map from (iline, xline) point
                to trace number
            `ilines`, `xlines` lists with possible values of respective coordinate
            `depth` contains length of each trace
        """
        if not isinstance(self.path, str):
            raise ValueError('Path to a segy-cube should be supplied!')

        # init all the containers
        with segyio.open(self.path, 'r', strict=False) as segyfile:
            segyfile.mmap() # makes operation faster

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

                # Maps:  cdp_x -> xline, cdp_y -> iline
                self.y_to_iline[cdp_y_] = iline_
                self.x_to_xline[cdp_x_] = xline_

        # More useful variables
        self.ilines = sorted(list(self.ilines))
        self.xlines = sorted(list(self.xlines))
        self.ilines_offset = min(self.ilines)
        self.xlines_offset = min(self.xlines)
        self.ilines_len = len(self.ilines)
        self.xlines_len = len(self.xlines)
        self.cube_shape = [self.ilines_len, self.xlines_len, self.depth]

        # Create transform from global coordinates to ilines/xlines/depth
        self.delay = header_.get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = header_.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL) // 1000

        transform_y = self._get_linear(self.cdp_y, self.ilines)
        transform_x = self._get_linear(self.cdp_x, self.xlines)
        transform_h = lambda h: ((h - self.delay) / self.sample_rate).astype(np.int64)
        self.abs_to_lines = (lambda array: np.stack([transform_y(array[:, 0]),
                                                     transform_x(array[:, 1]),
                                                     transform_h(array[:, 2])],
                                                    axis=-1))


    def _get_linear(self, set_x, set_y):
        """ Get linear-transformation that maps range of set_x into range of set_y.
        """
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

        # Callable to transform cube values to [0, 1] (and vice versa)
        scale = (self.value_max - self.value_min)
        self.scaler = lambda array: (array - self.value_min) / scale
        self.descaler = lambda array: array*scale + self.value_min


    def log(self, path_log=None):
        """ Log some info. """
        # pylint: disable=logging-format-interpolation
        path_log = path_log or ('/'.join(self.path.split('/')[:-1]) + '/CUBE_INFO.log')
        handler = logging.FileHandler(path_log, mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger('geometry_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info('Info for cube: {}'.format(self.path))
        with segyio.open(self.path, 'r', strict=False) as segyfile:
            header_file = segyfile.bin
            header_trace = segyfile.header[0]
            logger.info("\nFILE HEADER:")
            _ = [logger.info('{}: {}'.format(k, v))
                 for k, v in header_file.items()]

            logger.info("\nTRACE HEADER:")
            _ = [logger.info('{}: {}'.format(k, v))
                 for k, v in header_trace.items()]

        logger.info('\nSHAPES INFO:')
        logger.info('Depth of one trace is: {}'.format(self.depth))

        logger.info('Number of ILINES: {}'.format(self.ilines_len))
        logger.info('Number of XLINES: {}'.format(self.xlines_len))

        logger.info('ILINES range from {} to {}'.format(min(self.ilines), max(self.ilines)))
        logger.info('ILINES range from {} to {}'.format(min(self.xlines), max(self.xlines)))

        logger.info('CDP_X range from {} to {}'.format(min(self.cdp_x),
                                                       max(self.cdp_x)))
        logger.info('CDP_X range from {} to {}'.format(min(self.cdp_y),
                                                       max(self.cdp_y)))
