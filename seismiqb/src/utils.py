""" Utility functions. """
import dill
import logging
import numpy as np
import segyio
from tqdm import tqdm

from ..batchflow import Sampler, HistoSampler, NumpySampler, ConstantSampler


class Geometry():
    """ Class to hold information about .sgy-file. """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, path, **kwargs):
        if not isinstance(path, str):
            raise ValueError('')

        self.il_xl_trace = {}
        self.x_to_xline, self.y_to_iline = {}, {}
        self.ilines, self.xlines = set(), set()
        self.possible_cdp_x, self.possible_cdp_y = set(), set()
        self.value_min, self.value_max = np.inf, -np.inf
        self.get_geometry(path)

        if isinstance(kwargs.get('log'), str):
            self._log(path, path_log=kwargs.get('log'))


    def get_geometry(self, path):
        """ Actual parsing of .sgy-file.
        Does one full path through the file for collecting all the
        necessary information, including:
            `il_xl_trace` dictionary for map from (iline, xline) point
                to trace number
            `ilines`, `xlines` lists with possible values of respective coordinate
            `depth` contains length of each trace
        """
        # init all the containers
        with segyio.open(path, 'r', strict=False) as segyfile:
            segyfile.mmap() # makes operation faster

            self.depth = len(segyfile.trace[0])

            for i in tqdm(range(len(segyfile.header))):
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
                self.possible_cdp_x.add(cdp_x_)
                self.possible_cdp_y.add(cdp_y_)

                # Map:  cdp_x -> xline
                # Map:  cdp_y -> iline
                self.y_to_iline[cdp_y_] = iline_
                self.x_to_xline[cdp_x_] = xline_

                trace_ = segyfile.trace[i]
                if np.min(trace_) < self.value_min:
                    self.value_min = np.min(trace_)

                if np.max(trace_) > self.value_max:
                    self.value_max = np.max(trace_)

            # More useful variables
            self.ilines = sorted(list(self.ilines))
            self.xlines = sorted(list(self.xlines))
            self.ilines_offset = min(self.ilines)
            self.xlines_offset = min(self.xlines)
            self.ilines_len = len(self.ilines)
            self.xlines_len = len(self.xlines)
            self.cube_shape = [self.ilines_len, self.xlines_len, self.depth]


    def _log(self, path, path_log):
        """ Log some info. """
        logging.basicConfig(level=logging.INFO,
                            format=' %(message)s',
                            filename=path_log, filemode='w')
        logger = logging.getLogger('geometry_logger')

        with segyio.open(path, 'r', strict=False) as segyfile:
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

        logger.info('Number of ILINES: '.format(self.ilines_len))
        logger.info('Number of XLINES: '.format(self.xlines_len))

        logger.info('ILINES range from {} to {}'.format(min(self.ilines), max(self.ilines)))
        logger.info('ILINES range from {} to {}'.format(min(self.xlines), max(self.xlines)))

        logger.info('CDP_X range from {} to {}'.format(min(self.possible_cdp_x),
                                                       max(self.possible_cdp_x)))
        logger.info('CDP_X range from {} to {}'.format(min(self.possible_cdp_y),
                                                       max(self.possible_cdp_y)))



def repair(path_cube, geometry, path_save,
           i_low=0, i_high=-2, x_low=0, x_high=-2):
    """ Cuts unnecessary inlines/xlines from cube. """
    with segyio.open(path_cube, 'r', strict=False) as src:
        src.mmap()
        spec = segyio.spec()
        spec.sorting = int(src.sorting)
        spec.format = int(src.format)
        spec.samples = range(geometry.depth)
        spec.ilines = geometry.ilines[i_low:i_high]
        spec.xlines = geometry.xlines[x_low:x_high]

        with segyio.create(path_save, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]

            c = 0
            for il_ in tqdm(spec.ilines):
                for xl_ in spec.xlines:
                    tr_ = geometry.il_xl_trace[(il_, xl_)]
                    dst.header[c] = src.header[tr_]
                    dst.header[c][segyio.TraceField.FieldRecord] = il_
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_FILE] = il_

                    dst.header[c][segyio.TraceField.TraceNumber] = xl_ - geometry.xlines_offset
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_LINE] = xl_ - geometry.xlines_offset
                    dst.trace[c] = src.trace[tr_]
                    c += 1
            dst.bin = src.bin
            dst.bin = {segyio.BinField.Traces: c}

    # Check that repaired cube can be opened in 'strict' mode
    with segyio.open(path_save, 'r', strict=True) as segyfile:
        pass



def parse_labels(path_labels_txt, cube_geometry, sample_rate=4, delay=280, save_to=None):
    """ Transform labels from .txt files to one dictionary.
    Parameters
    ----------

    path_labels_txt : list of str
        Paths to .txt files with labels.
    cube_geometry : Geometry
        Instance of Geometry for corresponding cube with data.
    sample_rate : float, optional
        Frequency of trace sampling in time.
    delay : float, optional
        Time-delay of trace sampling.

    Returns
    -------
    il_xl_h : dict
        Mapping from (iline, xline) to height.

    Notes
    -----
    It can be helpful to look at .sgy-file headers. To do so, pass `verbose`
    parameter to Geometry constructor.
    """
    il_xl_h = {}
    for file_path in path_labels_txt:
        with open(file_path, 'r') as file:
            print('Parsing labels from' + file_path)
            for line in tqdm(file):
                line = line.split()
                line_x, line_y, line_h = np.array(line).astype(float).astype(int)

                iline_ = cube_geometry.y_to_iline.get(line_y) or cube_geometry.y_to_iline.get(line_y + 1)
                xline_ = cube_geometry.x_to_xline.get(line_x) or cube_geometry.x_to_xline.get(line_x + 1)
                k = int((line_h+delay)/sample_rate)

                if il_xl_h.get((iline_, xline_)) is not None:
                    il_xl_h[(iline_, xline_)].append(k)
                else:
                    il_xl_h[(iline_, xline_)] = [k]

    if isinstance(save_to, str):
        with open(save_to, 'wb') as file:
            dill.dump(il_xl_h, file)

    return il_xl_h


def make_geometries(dataset=None, load_from=None, save_to=None):
    """ Create Geometry for every cube in dataset and store it
    in `geometries` attribute of passed dataset.
    """
    if isinstance(load_from, str):
        with open(load_from, 'rb') as file:
            geometries = dill.load(file)
        print('Geometries are loaded from ' + load_from)
    else:
        geometries = {}
        for path_data in dataset.indices:
            print('Creating Geometry for file: ' + path_data)
            geometries.update({path_data: Geometry(path_data)})

    if len(geometries) < len(dataset):
        geometries = {path_data:geometries for path_data in dataset.indices}

    setattr(dataset, 'geometries', geometries)

    if isinstance(save_to, str):
        with open(save_to, 'wb') as file:
            dill.dump(geometries, file)
        print('Geometries are saved to ' + save_to)
    return geometries


def make_labels(dataset=None, path_to_txt=None, load_from=None, save_to=None):
    """ Create labels for every cube and store it in `labels`
    attribute of passed dataset.
    """
    if isinstance(load_from, str):
        with open(load_from, 'rb') as file:
            labels = dill.load(file)
        print('Labels are loaded from ' + load_from)
    else:
        labels = {}
        for path_data in dataset.indices:
            labels_ = parse_labels(path_to_txt[path_data],
                                   getattr(dataset, 'geometries')[path_data])
            labels.update({path_data: labels_})

    if len(labels) != len(dataset):
        labels = {path_data:labels for path_data in dataset.indices}

    setattr(dataset, 'labels', labels)

    if isinstance(save_to, str):
        with open(save_to, 'wb') as file:
            dill.dump(labels, file)
        print('Labels are saved to ' + save_to)
    return labels


def make_samplers(dataset=None, mode='hist', p=None,
                  load_from=None, save_to=None, **kwargs):
    """ Create samplers for every cube and store it in `samplers`
    attribute of passed dataset. Also creates one combined sampler
    and stores it in `sampler` attribute of passed dataset.

    Parameters
    ----------

    mode : str or Sampler
        Type of sampler to be created.
        If 'hist', then sampler is estimated from given labels.
        If 'numpy', then sampler is created with `kwargs` parameters.
        If instance of Sampler is provided, it must generate points from unit cube.

    p : list
        Weights for each mixture in final sampler.

    Note
    ----
    Passed `dataset` must have `geometries` and `labels` attributes if
    you want to create HistoSampler.
    """
    lowcut, highcut = [0, 0, 0], [1, 1, 1]

    if isinstance(load_from, str):
        with open(load_from, 'rb') as file:
            samplers = dill.load(file)
        print('Samplers are loaded from ' + load_from)
    else:
        samplers = {}
        if not isinstance(mode, dict):
            mode = {path_data:mode for path_data in dataset.indices}

        for path_data in dataset.indices:
            if isinstance(mode[path_data], Sampler):
                sampler = mode[path_data]
            elif mode[path_data] == 'numpy':
                sampler = NumpySampler(**kwargs)
            elif mode[path_data] == 'hist':
                _geom = getattr(dataset, 'geometries')[path_data]
                _labels = getattr(dataset, 'labels')[path_data]
                print('Creating histosampler for ' + path_data)
                bins = kwargs.get('bins') or 100
                sampler = make_histosampler(_labels, _geom, bins=bins)
            else:
                print('Making placeholder sampler for' + path_data)
                sampler = NumpySampler('u', low=0, high=1, dim=3)

            sampler = sampler.truncate(low=lowcut, high=highcut)
            samplers.update({path_data: sampler})
    setattr(dataset, 'samplers', samplers)

    # One sampler to rule them all
    p = p or [1/len(dataset) for path_data in dataset.indices]

    sampler = 0 & NumpySampler('n', dim=4)
    for i, path_data in enumerate(dataset.indices):
        sampler_ = (ConstantSampler(path_data)
                    & samplers[path_data].apply(lambda d: d.astype(np.object)))
        sampler = sampler | (p[i] & sampler_)
    sampler = sampler.truncate(low=lowcut, high=highcut, expr=lambda p: p[:, 1:])
    setattr(dataset, 'sampler', sampler)

    if isinstance(save_to, str):
        with open(save_to, 'wb') as file:
            dill.dump(samplers, file)
        print('Samplers are saved to ' + save_to)
    return samplers

def make_histosampler(input_dict, geometry, bins=100):
    """ Makes sampler from labels. """
    array_repr = []
    for item in tqdm(input_dict.items()):
        for h in item[1]:
            temp_ = [(item[0][0] - geometry.ilines_offset)/geometry.ilines_len,
                     (item[0][1] - geometry.xlines_offset)/geometry.xlines_len,
                     h/geometry.depth]
            array_repr.append(temp_)
    array_repr = np.array(array_repr)
    sampler = HistoSampler(np.histogramdd(array_repr, bins=bins))
    return sampler
