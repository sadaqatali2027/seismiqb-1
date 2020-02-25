""" Utility functions. """
import os
from collections import OrderedDict, MutableMapping
from threading import RLock
from functools import wraps
from glob import glob
from hashlib import blake2b

import dill
from tqdm import tqdm
import numpy as np
import segyio

from numba import njit

from ._const import FILL_VALUE



class classproperty:
    """ Adds property to the class, not to its instances. """
    #pylint: disable=invalid-name
    def __init__(self, prop):
        self.prop = prop
    def __get__(self, obj, owner):
        return self.prop(owner)


class PickleDict(MutableMapping):
    """ Persistent dictionary.
    Keys are file names, and values are stored/loaded via pickle.
    """
    def __init__(self, dirname, maxsize):
        self.dirname = dirname
        self.maxsize = maxsize


    @staticmethod
    def load(path):
        """ Load data from path. """
        with open(path, 'rb') as dill_file:
            restored = dill.load(dill_file)
        return restored

    @staticmethod
    def dump(path, value):
        """ Save data to path. """
        with open(path, 'wb') as file:
            dill.dump(value, file)


    def __setitem__(self, key, value):
        if len(self) > self.maxsize:
            self.popitem(last=False)

        key = blake2b(str(key).encode('ascii')).hexdigest()
        path = os.path.join(self.dirname, str(key)[:10])

        if os.path.exists(path):
            with open(path, 'a'):
                os.utime(path, None)
        else:
            self.dump(path, value)

    def __getitem__(self, key):
        key = blake2b(str(key).encode('ascii')).hexdigest()
        path = os.path.join(self.dirname, str(key)[:10])

        try:
            return self.load(path)
        except FileNotFoundError:
            raise KeyError(key) from None

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
    def __init__(self, maxsize=None, storage=OrderedDict(), classwide=True, anchor=None):
        self.maxsize = maxsize
        self.storage = storage
        self.is_full = False
        self.classwide = classwide

        if anchor is True:
            src_dir = os.path.dirname(os.path.realpath(__file__))
            code_lines = b''
            for path in glob(src_dir + '/*'):
                if os.path.isfile(path):
                    with open(path, 'rb') as code_file:
                        code_lines += code_file.read()
            self.anchor = blake2b(code_lines).hexdigest()
        else:
            self.anchor = False

        self.default = Singleton()
        self.lock = RLock()
        self.reset()


    def reset(self):
        """ Clear cache and stats. """
        if self.storage is None:
            self.cache = None
        elif isinstance(self.storage, str):
            self.cache = PickleDict(self.storage, maxsize=self.maxsize)
        else:
            self.cache = self.storage

        self.is_full = False
        self.stats = {'hit': 0, 'miss': 0}

    def make_key(self, args, kwargs):
        """ Make a key. """
        key = list(args)
        if kwargs:
            for k, v in kwargs.items():
                key.append((k, v))

        if self.classwide:
            key[0] = key[0].__class__
            if self.anchor is not None:
                key[0] = self.anchor
        else:
            key[0] = hash(key[0])
            if self.anchor is not None:
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
        wrapper.cache = self.cache
        wrapper.reset = self.reset
        wrapper.stats = self.stats
        return wrapper




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



@njit
def create_mask(ilines_, xlines_, hs_,
                il_xl_h, ilines_offset, xlines_offset, geom_depth,
                mode, width, horizons_idx, n_horizons=-1):
    """ Jit-accelerated function for fast mask creation for seismic horizons.
    This function is usually called inside SeismicCropBatch's method `create_masks`.
    """
    #pylint: disable=line-too-long, too-many-nested-blocks, too-many-branches
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))
    all_horizons = True
    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = iline_ + ilines_offset, xline_ + xlines_offset
            if il_xl_h.get((il_, xl_)) is None:
                continue
            m_temp = np.zeros(geom_depth)
            if mode == 'horizon':
                heights = il_xl_h[(il_, xl_)]
                if all_horizons:
                    filtered_idx = np.array([idx for idx, height_ in enumerate(heights)
                                             if height_ != FILL_VALUE])
                    filtered_idx = np.array([idx for idx in filtered_idx
                                             if heights[idx] > hs_[0] and heights[idx] < hs_[-1]])
                    if len(filtered_idx) == 0:
                        continue
                    if n_horizons != -1 and len(filtered_idx) >= n_horizons:
                        filtered_idx = np.random.choice(filtered_idx, replace=False, size=n_horizons)
                        all_horizons = False
                    if horizons_idx[0] != -1:
                        filtered_idx = np.array([idx for idx, height_ in enumerate(heights)
                                                 if height_ != FILL_VALUE and idx in horizons_idx])
                for idx in filtered_idx:
                    _height = heights[idx]
                    if _height != FILL_VALUE:
                        if width == 0:
                            m_temp[_height] = 1
                        else:
                            m_temp[max(0, _height - width):min(_height + width, geom_depth)] = 1
            elif mode == 'stratum':
                current_col = 1
                start = 0
                sorted_heights = sorted(il_xl_h[(il_, xl_)])
                for height_ in sorted_heights:
                    if height_ == FILL_VALUE:
                        height_ = start
                    if start > hs_[-1]:
                        break
                    m_temp[start:height_ + 1] = current_col
                    start = height_ + 1
                    current_col += 1
                    m_temp[sorted_heights[-1] + 1:min(hs_[-1] + 1, geom_depth)] = current_col
            else:
                raise ValueError('Mode should be either `horizon` or `stratum`')
            mask[i, j, :] = m_temp[hs_]
    return mask


@njit
def create_mask_f(ilines_, xlines_, hs_, il_xl_h, ilines_offset, xlines_offset, geom_depth):
    """ Jit-accelerated function for fast mask creation for seismic facies.
    This function is usually called inside SeismicCropBatch's method `create_masks`.
    """
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))

    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = iline_ + ilines_offset, xline_ + xlines_offset
            if il_xl_h.get((il_, xl_)) is None:
                continue

            m_temp = np.zeros(geom_depth)
            value = il_xl_h.get((il_, xl_))
            s_points, e_points, classes = value

            for s_p, e_p, c in zip(s_points, e_points, classes):
                m_temp[max(0, s_p):min(e_p+1, geom_depth)] = c+1
            mask[i, j, :] = m_temp[hs_]
    return mask



@njit
def count_nonfill(array):
    """ Jit-accelerated function to count non-fill elements. """
    count = 0
    for i in array:
        if i != FILL_VALUE:
            count += 1
    return count



@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape, order):
    """ Jit-accelerated function to glue together crops according to grid.
    At positions, where different crops overlap, only the maximum value is saved.
    This function is usually called inside SeismicCropBatch's method `assemble_crops`.
    """
    #pylint: disable=assignment-from-no-return
    total = len(array_grid)
    background = np.zeros(predict_shape)

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



@njit(parallel=True)
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
        ticks_ = ticks[i]
        if p <= ticks_[0]:
            values[i] = ticks_[0]
        elif p >= ticks_[-1]:
            values[i] = ticks_[-1]
        else:
            ix = np.searchsorted(ticks_, p)

            if abs(ticks_[ix] - p) <= abs(ticks_[ix-1] - p):
                values[i] = ticks_[ix]
            else:
                values[i] = ticks_[ix-1]
    return values



@njit
def update_minmax(array, val_min, val_max, matrix, il, xl, ilines_offset, xlines_offset):
    """ Get both min and max values in just one pass through array.
    Simultaneously updates (inplace) matrix if the trace is filled with zeros.
    """
    maximum = array[0]
    minimum = array[0]
    for i in array[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i

    if (minimum == 0) and (maximum == 0):
        matrix[il - ilines_offset, xl - xlines_offset] = 1

    if minimum < val_min:
        val_min = minimum
    if maximum > val_max:
        val_max = maximum

    return val_min, val_max, matrix



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
