""" Utility functions. """
import os
from copy import copy
from collections import OrderedDict, MutableMapping
from threading import RLock
from functools import wraps
from glob import glob
from hashlib import blake2b

import dill
import blosc
from tqdm import tqdm
import numpy as np
import pandas as pd
import segyio

from numba import njit



class SafeIO:
    """ Opens the file handler with desired `open` function, closes it at destruction.
    Can log open and close actions to the `log_file`.
    getattr, getitem and `in` operator are directed to the `handler`.
    """
    def __init__(self, path, opener=open, log_file=None, **kwargs):
        self.path = path
        self.log_file = log_file # or '/notebooks/log_safeio.txt'
        self.handler = opener(path, **kwargs)

        if self.log_file:
            self._info(self.log_file, f'Opened {self.path}')

    def _info(self, log_file, msg):
        with open(log_file, 'a') as f:
            f.write('\n' + msg)

    def __getattr__(self, key):
        return getattr(self.handler, key)

    def __getitem__(self, key):
        return self.handler[key]

    def __contains__(self, key):
        return key in self.handler

    def __del__(self):
        if self.log_file:
            self._info(self.log_file, f'Tried to close {self.path}')

        self.handler.close()

        if self.log_file:
            self._info(self.log_file, f'Closed {self.path}')


class IndexedDict(OrderedDict):
    """ Allows to use both indices and keys to subscript. """
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)



def stable_hash(key):
    """ Hash that stays the same between different runs of Python interpreter. """
    if not isinstance(key, (str, bytes)):
        key = ''.join(sorted(str(key)))
    if not isinstance(key, bytes):
        key = key.encode('ascii')
    return str(blake2b(key).hexdigest())

class Singleton:
    """ There must be only one!"""
    instance = None
    def __init__(self):
        if not Singleton.instance:
            Singleton.instance = self

class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to class methods.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    storage : None, OrderedDict or PickleDict
        Storage to use.
        If None, then no caching is applied.
    classwide : bool
        If True, then first argument of a method (self) is changed to class name for the purposes on hashing.
    anchor : bool
        If True, then code of the whole directory this file is located is used to create a persistent hash
        for the purposes of storing.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    pickle_module: str
        Module to use to save/load files on disk. Used only if `storage` is :class:`.PickleDict`.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Notes
    -----
    All arguments to the decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=None, classwide=False, attributes=None):
        self.maxsize = maxsize
        self.classwide = classwide

        # Make `attributes` always a list
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton()
        self.lock = RLock()
        self.reset()


    def reset(self):
        """ Clear cache and stats. """
        self.cache = OrderedDict()
        self.is_full = False
        self.stats = {'hit': 0, 'miss': 0}

    def make_key(self, args, kwargs):
        """ Create a key from a combination of instance reference or class reference,
        method args, and instance attributes.
        """
        key = list(args)
        # key[0] is `instance` if applied to a method
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(key[0], attr))
                key.append(attr_hash)

        if self.classwide:
            key[0] = key[0].__class__
        return tuple(key)


    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self.make_key(args, kwargs)

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache.get(key, self.default)
                if result is not self.default:
                    del self.cache[key]
                    self.cache[key] = result
                    self.stats['hit'] += 1
                    return result

            # The result was not found in cache: evaluate function
            result = func(*args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats['miss'] += 1
                if key in self.cache:
                    pass
                elif self.is_full:
                    self.cache.popitem(last=False)
                    self.cache[key] = result
                else:
                    self.cache[key] = result
                    self.is_full = (len(self.cache) >= self.maxsize)
            return result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        return wrapper



#TODO: rethink
def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

    Parameters
    ----------
    path : str
        Location of original .sgy cube.
    geometry : SeismicGeometry
        Infered information about original cube.
    path_save : str
        Place to save subcube.
    i_range : array-like
        Ilines to include in subcube.
    x_range : array-like
        Xlines to include in subcube.

    Notes
    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.
    """
    i_low, i_high = i_range[0], i_range[-1]
    x_low, x_high = x_range[0], x_range[-1]

    with segyio.open(path, 'r', strict=False) as src:
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
    with segyio.open(path_save, 'r', strict=True) as _:
        pass

#TODO: rename, add some defaults
def convert_point_cloud(path, path_save, names=None, order=None, transform=None):
    """ Change set of columns in file with point cloud labels.
    Usually is used to remove redundant columns.

    Parameters
    ----------
    path : str
        Path to file to convert.
    path_save : str
        Path for the new file to be saved to.
    names : str or sequence of str
        Names of columns in the original file. Default is Petrel's export format, which is
        ('_', '_', 'iline', '_', '_', 'xline', 'cdp_x', 'cdp_y', 'height'), where `_` symbol stands for
        redundant keywords like `INLINE`.
    order : str or sequence of str
        Names and order of columns to keep. Default is ('iline', 'xline', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    data = df.loc[:, order]
    if transform:
        data = data.apply(transform)
    data.to_csv(path_save, sep=' ', index=False, header=False)



@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape, order):
    """ Jit-accelerated function to glue together crops according to grid.
    At positions, where different crops overlap, only the maximum value is saved.
    This function is usually called inside SeismicCropBatch's method `assemble_crops`.
    """
    #pylint: disable=assignment-from-no-return
    total = len(array_grid)
    background = np.full(predict_shape, np.min(array_crops))

    for i in range(total):
        il, xl, h = array_grid[i, :]
        il_end = min(background.shape[0], il+crop_shape[0])
        xl_end = min(background.shape[1], xl+crop_shape[1])
        h_end = min(background.shape[2], h+crop_shape[2])

        crop = np.transpose(array_crops[i], order)
        crop = crop[:(il_end-il), :(xl_end-xl), :(h_end-h)]
        previous = background[il:il_end, xl:xl_end, h:h_end]
        background[il:il_end, xl:xl_end, h:h_end] = np.maximum(crop, previous)
    return background


@njit
def groupby_mean(array):
    """ Faster version of mean-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1], 1

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1]
            c += 1
        else:
            output[position, :2] = prev
            output[position, -1] = s / c
            position += 1

            prev = curr
            s, c = array[i, -1], 1

    output[position, :2] = prev
    output[position, -1] = s / c
    position += 1
    return output[:position]




@njit
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.
    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        if p <= ticks[0]:
            values[i] = ticks[0]
        elif p >= ticks[-1]:
            values[i] = ticks[-1]
        else:
            ix = np.searchsorted(ticks, p)

            if abs(ticks[ix] - p) <= abs(ticks[ix-1] - p):
                values[i] = ticks[ix]
            else:
                values[i] = ticks[ix-1]
    return values


@njit
def find_min_max(array):
    """ Get both min and max values in just one pass through array."""
    n = array.size
    max_val = min_val = array[0]
    for i in range(1, n):
        min_val = min(array[i], min_val)
        max_val = max(array[i], max_val)
    return min_val, max_val



def compute_running_mean(x, kernel_size):
    """ Fast analogue of scipy.signal.convolve2d with gaussian filter. """
    k = kernel_size // 2
    padded_x = np.pad(x, (k, k), mode='symmetric')
    cumsum = np.cumsum(padded_x, axis=1)
    cumsum = np.cumsum(cumsum, axis=0)
    return _compute_running_mean_jit(x, kernel_size, cumsum)

@njit
def _compute_running_mean_jit(x, kernel_size, cumsum):
    """ Jit accelerated running mean. """
    #pylint: disable=invalid-name
    k = kernel_size // 2
    result = np.zeros_like(x).astype(np.float32)

    canvas = np.zeros((cumsum.shape[0] + 2, cumsum.shape[1] + 2))
    canvas[1:-1, 1:-1] = cumsum
    cumsum = canvas

    for i in range(k, x.shape[0] + k):
        for j in range(k, x.shape[1] + k):
            d = cumsum[i + k + 1, j + k + 1]
            a = cumsum[i - k, j  - k]
            b = cumsum[i - k, j + 1 + k]
            c = cumsum[i + 1 + k, j - k]
            result[i - k, j - k] = float(d - b - c + a) /  float(kernel_size ** 2)
    return result




class PickleDict(MutableMapping):
    """ Persistent dictionary.
    Keys are file names, and values are stored/loaded via pickle module of choice.
    """
    def __init__(self, dirname, maxsize, pickle_module='dill'):
        self.dirname = dirname
        self.maxsize = maxsize
        self.pickle_module = pickle_module


    @staticmethod
    def load(path):
        """ Load data from path. """
        pickle_module = os.path.splitext(path)[1][1:]

        if pickle_module == 'dill':
            with open(path, 'rb') as dill_file:
                restored = dill.load(dill_file)
        elif pickle_module == 'blosc':
            with open(path, 'rb') as blosc_file:
                restored = dill.loads(blosc.decompress(blosc_file.read()))
        return restored

    @staticmethod
    def dump(path, value):
        """ Save data to path. """
        pickle_module = os.path.splitext(path)[1][1:]

        if pickle_module == 'dill':
            with open(path, 'wb') as file:
                dill.dump(value, file)
        elif pickle_module == 'blosc':
            with open(path, 'w+b') as file:
                file.write(blosc.compress(dill.dumps(value)))


    def __getitem__(self, key):
        key = stable_hash(key)
        path = os.path.join(self.dirname, str(key)[:10]) + '.{}'.format(self.pickle_module)

        try:
            return self.load(path)
        except FileNotFoundError:
            raise KeyError(key) from None

    def __setitem__(self, key, value):
        if len(self) > self.maxsize:
            self.popitem(last=False)

        key = stable_hash(key)
        path = os.path.join(self.dirname, str(key)[:10]) + '.{}'.format(self.pickle_module)

        if os.path.exists(path):
            with open(path, 'a'):
                os.utime(path, None)
        else:
            self.dump(path, value)

    def __delitem__(self, key):
        pass

    def popitem(self, last=False):
        """ Delete either the oldest or the newest file in the directory. """
        lst = []
        for path in os.listdir(self.dirname):
            filepath = os.path.join(self.dirname, path)
            if os.path.isfile(filepath):
                lst.append((filepath, os.path.getmtime(filepath)))
        lst.sort(key=lambda item: item[1])

        if last is False:
            os.remove(lst[0][0])
        else:
            os.remove(lst[-1][0])


    def __len__(self):
        return len(os.listdir(self.dirname))

    def __iter__(self):
        return iter(os.listdir(self.dirname))

    def __repr__(self):
        return 'PickleDict on {}'.format(self.dirname)


class file_cache:
    """ Thread-safe least recent used cache. Must be applied to class methods.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    storage : None, OrderedDict or PickleDict
        Storage to use.
        If None, then no caching is applied.
    classwide : bool
        If True, then first argument of a method (self) is changed to class name for the purposes on hashing.
    anchor : bool
        If True, then code of the whole directory this file is located is used to create a persistent hash
        for the purposes of storing.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    pickle_module: str
        Module to use to save/load files on disk. Used only if `storage` is :class:`.PickleDict`.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Notes
    -----
    All arguments to the decorated method must be hashable.
    """
    #pylint: disable=invalid-name
    def __init__(self, maxsize=None, storage=OrderedDict(), classwide=True, anchor=None, attributes=None,
                 pickle_module='dill'):
        self.maxsize = maxsize
        self.storage = storage
        self.classwide = classwide
        self.pickle_module = pickle_module

        # Make attributes always a list
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        # Create one stable hash from every file in current (src) directory
        if anchor is True:
            src_dir = os.path.dirname(os.path.realpath(__file__))
            code_lines = b''
            for path in glob(src_dir + '/*'):
                if os.path.isfile(path):
                    with open(path, 'rb') as code_file:
                        code_lines += code_file.read()
            self.anchor = stable_hash(code_lines)
        else:
            self.anchor = False

        self.is_full = False
        self.default = Singleton()
        self.lock = RLock()
        self.reset()


    def reset(self):
        """ Clear cache and stats. """
        if self.storage is None:
            self.cache = None
        elif isinstance(self.storage, str):
            self.cache = PickleDict(self.storage, maxsize=self.maxsize, pickle_module=self.pickle_module)
        else:
            #TODO: add good explanation of this
            self.cache = copy(self.storage)

        self.is_full = False
        self.stats = {'hit': 0, 'miss': 0}

    def make_key(self, args, kwargs):
        """ Create a key from a combination of instance reference, class reference, method args,
        instance attributes or even current code state.
        """
        key = list(args)
        if kwargs:
            for k, v in kwargs.items():
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(key[0], attr))
                key.append(attr_hash)

        if self.classwide:
            key[0] = key[0].__class__
            if self.anchor:
                key[0] = self.anchor
        else:
            if self.anchor:
                key.append(self.anchor)
        return tuple(key)


    def __call__(self, func):
        if self.cache is None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self.make_key(args, kwargs)
            with self.lock:
                result = self.cache.get(key, self.default)
                if result is not self.default:
                    del self.cache[key]
                    self.cache[key] = result
                    self.stats['hit'] += 1
                    return result
            result = func(*args, **kwargs)
            with self.lock:
                self.stats['miss'] += 1
                if key in self.cache:
                    pass
                elif self.is_full:
                    self.cache.popitem(last=False)
                    self.cache[key] = result
                else:
                    self.cache[key] = result
                    self.is_full = (len(self.cache) >= self.maxsize)
            return result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.reset = self.reset
        wrapper.stats = lambda: self.stats
        return wrapper

@njit
def find_max_overlap(point, horizon_matrix, coverage, xlines_len, ilines_len,
                        stride, shape, fill_value, min_empty=5, border_gap=0):
    candidates, shapes = [], []
    orders, intersections = [], []

    hor_height = int(horizon_matrix[point[0], point[1]])

    ils = [point[0] - stride, point[0] - shape[1] + stride]
    for il in ils:
        if il > border_gap and il + shape[1] < ilines_len - border_gap:
            empty_space = np.nonzero(coverage[il: il + shape[1],
                            point[1]:point[1] + shape[0]] == fill_value)
            if len(empty_space[0]) > min_empty:
                candidates.append([il, point[1], hor_height - shape[2] // 2])
                shapes.append([shape[1], shape[0], shape[2]])
                orders.append([0, 2, 1])
                intersections.append(shape[1] - len(empty_space[0]))
                coverage[il: il + shape[1], point[1]:point[1] + shape[0]] = 0

    xls = [point[1] - stride, point[1] - shape[1] + stride]
    for xl in xls:
        if xl > border_gap and xl + shape[1] < xlines_len - border_gap:
            empty_space = np.nonzero(coverage[point[0]:point[0] + shape[0],
                                                    xl: xl + shape[1]] == fill_value)
            if len(empty_space[0]) > min_empty:
                candidates.append([point[0], xl, hor_height - shape[2] // 2])
                shapes.append(shape)
                orders.append([2, 0, 1])
                intersections.append(shape[1] - len(empty_space[0]))
                coverage[point[0]:point[0] + shape[0], xl: xl + shape[1]] = 0

    if len(candidates) == 0:
        return None

    candidates_array = np.array(candidates)
    shapes_array = np.array(shapes)
    orders_array = np.array(orders)
    top2 = np.argsort(np.array(intersections))[:2]
    return (candidates_array[top2], \
                shapes_array[top2], \
                orders_array[top2]
                )
