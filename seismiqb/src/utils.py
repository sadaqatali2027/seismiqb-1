""" Utility functions. """
import dill

import numpy as np
import segyio
from numba.typed import Dict
from numba import types
from numba import njit
from tqdm import tqdm
import pandas as pd

from ..batchflow import Sampler, HistoSampler, NumpySampler, ConstantSampler

FILL_VALUE = -999.0

class Geometry():
    """ Class to hold information about .sgy-file. """
    # pylint: disable=R0902, W0106
    def __init__(self, path_data, **kwargs):
        if not isinstance(path_data, str):
            raise ValueError('')

        self.il_xl_trace = {}
        self.x_to_xline, self.y_to_iline = {}, {}
        self.ilines, self.xlines = set(), set()
        self.possible_cdp_x, self.possible_cdp_y = set(), set()
        self.value_min, self.value_max = np.inf, -np.inf
        self.get_geometry(path_data)

        if kwargs.get('verbose'):
            self._quack(path_data)


    def get_geometry(self, path_data):
        """ Actual parsing of .sgy-file.
        Does one full path through the file for collecting all the
        necessary information, including:
            `il_xl_trace` dictionary for map from (iline, xline) point
                to trace number
            `ilines`, `xlines` lists with possible values of respective coordinate
            `depth` contains length of each trace
        """
        # init all the containers
        with segyio.open(path_data, 'r', strict=False) as segyfile:
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
                self.ilines.add(iline_), self.xlines.add(xline_)
                self.possible_cdp_x.add(cdp_x_), self.possible_cdp_y.add(cdp_y_)

                # Map:  cdp_x -> xline
                # Map:  cdp_y -> iline
                self.y_to_iline[cdp_y_] = iline_
                self.x_to_xline[cdp_x_] = xline_

                # trace_ = segyfile.trace[i]
                # if np.min(trace_) < self.value_min:
                #     self.value_min = np.min(trace_)

                # if np.max(trace_) > self.value_max:
                #     self.value_max = np.max(trace_)

            # More useful variables
            self.ilines = sorted(list(self.ilines))
            self.xlines = sorted(list(self.xlines))
            self.ilines_offset = min(self.ilines)
            self.xlines_offset = min(self.xlines)
            self.ilines_len = len(self.ilines)
            self.xlines_len = len(self.xlines)
            self.cube_shape = [self.ilines_len, self.xlines_len, self.depth]


    def _quack(self, path_data):
        """ Log some info. """
        with segyio.open(path_data, 'r', strict=False) as segyfile:
            header_file = segyfile.bin
            header_trace = segyfile.header[0]
            print("File header: ", header_file)
            print("\n\nTrace header: ", header_trace)

        print('\nDepth of one trace is:', self.depth)

        print('Number of ILINES:', self.ilines_len)
        print('Number of XLINES:', self.xlines_len)

        print('ILINES range from ' + str(min(self.ilines)) + ' to ' + str(max(self.ilines)))
        print('XLINES range from ' + str(min(self.xlines)) + ' to ' + str(max(self.xlines)))

        print('CDP_X range from ' + str(min(self.possible_cdp_x))
              + ' to ' + str(max(self.possible_cdp_x)))
        print('CDP_Y range from ' + str(min(self.possible_cdp_y))
              + ' to ' + str(max(self.possible_cdp_y)))


    def get(self, attr, default=None):
        """ Get. """
        if hasattr(self, attr):
            return getattr(self, attr)
        return default



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

    with segyio.open(path_save, 'r', strict=True) as segyfile:
        segyfile.mmap()
        print('Cube can be opened in `strict` mode now! ')


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

def read_point_cloud(paths, default=None, order=('iline', 'xline', 'height'), transforms=None, **kwargs):
    """ Read point cloud of horizont-labels from files.
    """
    paths = (paths, ) if isinstance(paths, str) else paths

    # default params of pandas-parser
    if default is None:
        default = dict(sep='\s+', names=['xline', 'iline', 'height'])

    # read point clouds
    point_clouds = []
    for path in paths:
        copy = default.copy()
        copy.update(kwargs.get(path, dict()))
        cloud = pd.read_csv(path, **copy)
        point_clouds.append(cloud.loc[:, order].values)

    points = np.concatenate(point_clouds)

    # apply transforms
    if transforms is not None:
        for i in range(points.shape[-1]):
            points[:, i] = transforms[i](points[:, i])

    return points

def make_labels_dict(point_cloud):
    """ Make labels-dict using cloud of points.
    """
    # round and cast
    ilines_xlines = np.round(point_cloud[:, :2]).astype(np.int64)

    # make dict-types
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.float64[:]

    # find max array-length
    counts = Dict.empty(key_type, types.int64)

    @njit
    def fill_counts_get_max(counts, ilines_xlines):
        max_count = 0
        for i in range(len(ilines_xlines)):
            il, xl = ilines_xlines[i, :2]
            if counts.get((il, xl)) is None:
                counts[(il, xl)] = 0

            counts[(il, xl)] += 1
            if counts[(il, xl)] > max_count:
                max_count = counts[(il, xl)]

        return max_count

    max_count = fill_counts_get_max(counts, ilines_xlines)

    # put key-value pairs into numba-dict
    labels = Dict.empty(key_type, value_type)

    @njit
    def fill_labels(labels, counts, ilines_xlines, max_count):
        # zero-out the counts
        for k in counts.keys():
            counts[k] = 0

        # fill labels-dict
        for i in range(len(ilines_xlines)):
            il, xl = ilines_xlines[i, :2]
            if labels.get((il, xl)) is None:
                labels[(il, xl)] = np.full((max_count, ), FILL_VALUE, np.float64)

            labels[(il, xl)][counts[(il, xl)]] = point_cloud[i, 2]
            counts[(il, xl)] += 1

    fill_labels(labels, counts, ilines_xlines, max_count)
    return labels

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
