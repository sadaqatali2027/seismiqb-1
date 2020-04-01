""" Contains metrics for various labels (horizons, facies, etc) and cubes. """
#pylint: disable=too-many-lines, not-an-iterable
from copy import copy
from textwrap import dedent
from tqdm.auto import tqdm

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

import cv2
from scipy.signal import hilbert, medfilt

from ..batchflow.models.metrics import Metrics

from .horizon import Horizon
from .utils import compute_running_mean
from .plot_utils import plot_image



class BaseSeismicMetric(Metrics):
    """ !!. """
    LOCAL_DEFAULTS = {
        'kernel_size': 3,
        'reduce_func': 'nanmean',
        'agg': None,
    }

    SUPPORT_DEFAULTS = {
        'supports': 20,
        'agg': 'mean',
    }

    SMOOTHING_DEFAULTS = {
        'kernel_size': 21,
        'sigma': 10.0,
    }

    EPS = 0.00001

    def evaluate(self, metrics, agg='mean', plot=False, show_plot=True, savepath=None, backend='matplotlib',
                 plot_kwargs=None, scalar=False, **kwargs):
        """ Calculate desired metrics.
        To plot the results, set `plot` argument to True.

        Parameters
        ----------
        metrics : str or sequence of str
            Names of metrics to evaluate.
        agg : int, str or callable
            Function to transform metric from ndarray of (n_ilines, n_xlines, N) shape to (n_ilines, n_xlines) shape.
            If callable, then directly applied to the output of metric computation function.
            If str, then must be a function from `numpy` module. Applied along the last axis only.
            If int, then index of slice along the last axis to return.
        kwargs : dict
            Metric-specific parameters.

        Returns
        -------
        If `metric` is str, then metric value
        If `metric` is dict, than dict where keys are metric names and values are metric values.
        """
        _metrics = [metrics] if isinstance(metrics, str) else metrics
        _agg = [agg]*len(_metrics) if not isinstance(agg, (tuple, list)) else agg

        res = {}
        for name, agg_func in zip(_metrics, _agg):
            # Get metric, then aggregate
            metric_fn = getattr(self, name)
            metric_val, plot_dict = metric_fn(**kwargs)
            metric_val = self._aggregate(metric_val, agg_func)

            # Get plot parameters
            # TODO: make plot functions use only needed parameters
            plot_dict = {**plot_dict, **(plot_kwargs or {})}
            ignore_value = plot_dict.pop('ignore_value', None)
            spatial = plot_dict.pop('spatial', True)
            _ = backend, plot_dict.pop('zmin', -1), plot_dict.pop('zmax', 1)

            # np.nan allows to ignore values
            if ignore_value is not None:
                copy_metric = np.copy(metric_val)
                copy_metric[copy_metric == ignore_value] = np.nan
            else:
                copy_metric = metric_val

            # Actual plot
            if plot:
                if spatial:
                    plot_image(copy_metric, savefig=savepath, show_plot=show_plot, **plot_dict)
                else:
                    pass
            if scalar:
                print('Scalar value of metric is {}'.format(np.nanmean(copy_metric)))
            res[name] = metric_val

        res = res[metrics] if isinstance(metrics, str) else res
        return res

    def _aggregate(self, metric, agg=None):
        if agg is not None:
            if callable(agg):
                metric = agg(metric)
            elif isinstance(agg, str):
                metric = getattr(np, agg)(metric, axis=-1)
            elif isinstance(agg, (int, slice)):
                metric = metric[..., agg]
        return metric


    def local_corrs(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ Compute average correlation between each column in data and nearest traces.

        Parameters
        ----------
        locality : {4, 8}
            Defines number of nearest traces to average correlations from.

        Returns
        -------
        array-like
            Matrix of (n_ilines, n_xlines) shape with computed metric for each point.
        """
        metric, title = compute_local_corrs(data=self.data, bad_traces=self.bad_traces,
                                            kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict

    def support_corrs(self, supports=10, safe_strip=0, line_no=None, **kwargs):
        """ Compute correlations with support traces.

        Parameters
        ----------
        supports : int, sequence, ndarray or str
            Defines mode of generating support traces.
            If int, then that number of random non-zero traces positions are generated.
            If sequence or ndarray, then must be of shape (N, 2) and is used as positions of support traces.
            If str, then must define either `iline` or `xline` mode. In each respective one, iline/xline given by
            `line_no` argument is used to generate supports.
        safe_strip : int
            Used only for `int` mode of `supports` parameter and defines minimum distance
            from borders for sampled points.
        line_no : int
            Used only for `str` mode of `supports` parameter to define exact iline/xline to use.

        Returns
        -------
        array-like
            Matrix of either (n_ilines, n_xlines, n_supports) or (n_ilines, n_xlines) shape with
            computed metric for each point.
        """
        metric, title = compute_support_corrs(data=self.data, supports=supports, bad_traces=self.bad_traces,
                                              safe_strip=safe_strip, line_no=line_no, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict

    def local_btch(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ !!. """
        metric, title = compute_local_btch(data=self.probs, bad_traces=self.bad_traces,
                                           kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': 0.0, 'zmax': 1.0,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    def support_btch(self, supports=10, safe_strip=0, **kwargs):
        """ !!. """
        metric, title = compute_support_btch(data=self.probs, supports=supports, bad_traces=self.bad_traces,
                                             safe_strip=safe_strip, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': 0.0, 'zmax': 1.0,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


    def local_kl(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ !!. """
        metric, title = compute_local_kl(data=self.probs, bad_traces=self.bad_traces,
                                         kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    def support_kl(self, supports=10, safe_strip=0, **kwargs):
        """ !!. """
        metric, title = compute_support_kl(data=self.probs, supports=supports, bad_traces=self.bad_traces,
                                           safe_strip=safe_strip, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


    def local_js(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ !!. """
        metric, title = compute_local_js(data=self.probs, bad_traces=self.bad_traces,
                                         kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    def support_js(self, supports=10, safe_strip=0, **kwargs):
        """ !!. """
        metric, title = compute_support_js(data=self.probs, supports=supports, bad_traces=self.bad_traces,
                                           safe_strip=safe_strip, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


    def local_hellinger(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ !!. """
        metric, title = compute_local_hellinger(data=self.probs, bad_traces=self.bad_traces,
                                                kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    def support_hellinger(self, supports=10, safe_strip=0, **kwargs):
        """ !!. """
        metric, title = compute_support_hellinger(data=self.probs, supports=supports, bad_traces=self.bad_traces,
                                                  safe_strip=safe_strip, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


    def local_wasserstein(self, kernel_size=3, reduce_func='nanmean', **kwargs):
        """ !!. """
        metric, title = compute_local_wasserstein(data=self.probs, bad_traces=self.bad_traces,
                                                  kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}, k={kernel_size}, reduce={reduce_func}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    def support_wasserstein(self, supports=10, safe_strip=0, **kwargs):
        """ !!. """
        metric, title = compute_support_wasserstein(data=self.probs, supports=supports, bad_traces=self.bad_traces,
                                                    safe_strip=safe_strip, **kwargs)
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict

    # Aliases for Wasserstein distance
    local_emd = local_wasserstein
    support_emd = support_wasserstein


    def hilbert(self, mode='median', kernel_size=3, eps=1e-5, **kwargs):
        """ Compute phase along the data. """
        _ = kwargs
        # full_matrix = self.horizon.full_matrix

        analytic = hilbert(self.data, axis=-1)
        phase = (np.angle(analytic))
        phase = phase % (2 * np.pi) - np.pi
        # phase[full_matrix == Horizon.FILL_VALUE, :] = 0

        horizon_phase = phase[:, :, phase.shape[-1] // 2]
        horizon_phase = correct_pi(horizon_phase, eps)

        if mode == 'mean':
            median_phase = compute_running_mean(horizon_phase, kernel_size)
        else:
            median_phase = medfilt(horizon_phase, kernel_size)
        # median_phase[full_matrix == Horizon.FILL_VALUE] = 0

        img = np.minimum(median_phase - horizon_phase, 2 * np.pi + horizon_phase - median_phase)
        # img[full_matrix == Horizon.FILL_VALUE] = 0
        img = np.where(img < -np.pi, img + 2 * np. pi, img)

        metric = np.zeros((*img.shape, 2+self.data.shape[2]))
        metric[:, :, 0] = img
        metric[:, :, 1] = median_phase
        metric[:, :, 2:] = phase

        title = 'phase by {}'.format(mode)
        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.name, self.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            # **kwargs
        }
        return metric, plot_dict


    def quality_map(self, quantiles, metric_names=None, computed_metrics=None, reduce_func='nanmean',
                    smoothing_params=None, local_params=None, support_params=None, **kwargs):
        """ !!. """
        _ = kwargs
        computed_metrics = computed_metrics or []
        smoothing_params = smoothing_params or self.SMOOTHING_DEFAULTS
        local_params = local_params or self.LOCAL_DEFAULTS
        support_params = support_params or self.SUPPORT_DEFAULTS

        smoothing_params = {**self.SMOOTHING_DEFAULTS, **smoothing_params}
        local_params = {**self.LOCAL_DEFAULTS, **local_params}
        support_params = {**self.SUPPORT_DEFAULTS, **support_params}

        for metric_name in metric_names:
            if metric_name.startswith('local'):
                kwds = copy(local_params)
            elif metric_name.startswith('supp'):
                kwds = copy(support_params)

            metric = self.evaluate(metric_name, plot=False, **kwds)
            computed_metrics.append(metric)

        digitized_metrics = []
        for metric_matrix in computed_metrics:
            smoothed = smooth_out(metric_matrix, **smoothing_params)
            digitized = digitize(smoothed, quantiles)
            digitized_metrics.append(digitized)

        quality_map = np.stack(digitized_metrics, axis=-1)
        quality_map = getattr(np, reduce_func)(quality_map, axis=-1)
        quality_map = smooth_out(quality_map, **smoothing_params)

        title = 'quality map'
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'Reds',
            'zmin': 0.0, 'zmax': np.max(quality_map),
            'ignore_value': np.nan,
            # **kwargs
        }
        return quality_map, plot_dict


    def make_grid(self, quality_map, frequencies, iline=True, xline=True, margin=0, **kwargs):
        """ !!. """
        _ = kwargs
        if margin:
            bad_traces = np.copy(self.geometry.zero_traces)
            bad_traces[:, 0] = 1
            bad_traces[:, -1] = 1
            bad_traces[0, :] = 1
            bad_traces[-1, :] = 1

            kernel = np.ones((2 + 2*margin, 2 + 2*margin), dtype=np.uint8)
            bad_traces = cv2.dilate(bad_traces.astype(np.uint8), kernel, iterations=1).astype(bad_traces.dtype)
            quality_map[(bad_traces - self.geometry.zero_traces) == 1] = 0.0

        pre_grid = np.rint(quality_map)
        grid = gridify(pre_grid, frequencies, iline, xline)

        if margin:
            grid[(bad_traces - self.geometry.zero_traces) == 1] = 0
        return grid



class HorizonMetrics(BaseSeismicMetric):
    """ Evaluate metric(s) on horizon(s).
    During initialization, data along the horizon is cut with the desired parameters.
    To get the value of a particular metric, use :meth:`.evaluate`::
        HorizonMetrics(horizon).evaluate('support_corrs', supports=20, agg='mean')

    To plot the results, set `plot` argument of :meth:`.evaluate` to True.

    Parameters
    horizons : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to evaluate.
        Can be either one horizon, then this horizon is evaluated on its own,
        or sequence of two horizons, then they are compared against each other,
        or nested sequence of horizon and list of horizons, then the first horizon is compared against the
        best match from the list.
    other parameters
        Passed direcly to :meth:`.Horizon.get_cube_values` or :meth:`.Horizon.get_cube_values_line`.
    """
    AVAILABLE_METRICS = [
        'local_corrs', 'support_corrs',
        'local_kl', 'support_kl',
        'local_hellinger', 'support_hellinger',
        'local_wasserstein', 'support_wasserstein',
        'hilbert',
    ]

    def __init__(self, horizons, orientation=None, window=23, offset=0, scale=False, chunk_size=256, line=1):
        super().__init__()
        horizons = list(horizons) if isinstance(horizons, tuple) else horizons
        horizons = horizons if isinstance(horizons, list) else [horizons]
        self.horizons = horizons

        # Save parameters for later evaluation
        self.orientation, self.line = orientation, line
        self.window, self.offset, self.scale, self.chunk_size = window, offset, scale, chunk_size

        # The first horizon is used to evaluate metrics
        self.horizon = horizons[0]
        self.name = self.horizon.name
        self.cube_name = self.horizon.cube_name

        if orientation is None: # metrics are computed on full cube (spatially)
            self._data = None # evaluated later
            self._probs = None
            self.bad_traces = np.copy(self.horizon.geometry.zero_traces)
            self.bad_traces[self.horizon.full_matrix == Horizon.FILL_VALUE] = 1
            self.spatial = True

        else: # metrics are computed on a specific slide
            self._data, self.bad_traces = self.horizon.get_cube_values_line(orientation=orientation, line=line,
                                                                            window=window, offset=offset, scale=scale)
            self._probs = None
            self.spatial = False

    @property
    def data(self):
        """ Create `data` attribute at the first time of evaluation. """
        if self._data is None:
            self._data = self.horizon.get_cube_values(window=self.window, offset=self.offset,
                                                      scale=self.scale, chunk_size=self.chunk_size)
        self._data[self._data == Horizon.FILL_VALUE] = np.nan
        return self._data

    @property
    def probs(self):
        """ !!. """
        if self._probs is None:
            # mins = np.min(self.data, axis=-1, keepdims=True)
            # maxs = np.max(self.data, axis=-1, keepdims=True)
            # shift_scaled = (self.data - mins) / (maxs - mins)
            # self._probs = shift_scaled / np.sum(shift_scaled, axis=-1, keepdims=True) + self.EPS

            hist_matrix = NumbaNumpy.histo_reduce(self.data, self.horizon.geometry.bins)
            self._probs = hist_matrix / np.sum(hist_matrix, axis=-1, keepdims=True) + self.EPS
        return self._probs



    def compare(self, offset=0, absolute=True, hist=True, printer=print, **kwargs):
        """ Compare horizons on against the best match from the list of horizons.

        Parameters
        ----------
        offset : number
            Value to shift horizon down. Can be used to take into account different counting bases.
        absolute : bool
            Whether to use absolute values for differences.
        hist : bool
            Whether to plot histogram of differences.
        printer : callable
            Function to print results, for example `print` or any other callable that can log data.
        """
        _ = kwargs
        if len(self.horizons) != 2:
            raise ValueError('Can compare two horizons exactly or one to the best match from list of horizons. ')
        if isinstance(self.horizons[1], Horizon):
            self.horizons[1] = [self.horizons[1]]

        lst = []
        for hor in self.horizons[1]:
            if hor.geometry.name == self.horizon.geometry.name:
                overlap_info = Horizon.verify_merge(self.horizon, hor, adjacency=3)[1]
                lst.append((hor, overlap_info))
        lst.sort(key=lambda x: x[1].get('mean', 999999))
        other, overlap_info = lst[0] # the best match

        self_full_matrix = self.horizon.full_matrix
        other_full_matrix = other.full_matrix
        metric = np.where((self_full_matrix != other.FILL_VALUE) & (other_full_matrix != other.FILL_VALUE),
                          offset + self_full_matrix - other_full_matrix, np.nan)
        if absolute:
            metric = np.abs(metric)

        window_rate = np.mean(np.abs(metric[~np.isnan(metric)]) < (5 / other.geometry.sample_rate))
        max_abs_error = np.nanmax(np.abs(metric))
        max_abs_error_count = np.sum(metric == max_abs_error) + np.sum(metric == -max_abs_error)
        at_1 = len(np.asarray((self_full_matrix != other.FILL_VALUE) &
                              (other_full_matrix == other.FILL_VALUE)).nonzero()[0])
        at_2 = len(np.asarray((self_full_matrix == other.FILL_VALUE) &
                              (other_full_matrix != other.FILL_VALUE)).nonzero()[0])

        if printer is not None:
            msg = f"""
            Comparing horizons:       {self.horizon.name}
                                    {other.name}
            {'—'*45}

            Rate in 5ms:                         {window_rate:8.4}
            Mean/std of errors:       {np.nanmean(metric):8.4} / {np.nanstd(metric):8.4}
            Max abs error/count:      {max_abs_error:8.4} / {max_abs_error_count:8}
            {'—'*45}

            Lengths of horizons:                 {len(self.horizon):8}
                                                {len(other):8}
            {'—'*45}
            Average heights of horizons:         {(offset + self.horizon.h_mean):8.4}
                                                {other.h_mean:8.4}
            {'—'*45}
            Coverage of horizons:                {self.horizon.coverage:8.4}
                                                {other.coverage:8.4}
            {'—'*45}
            Solidity of horizons:                {self.horizon.solidity:8.4}
                                                {other.solidity:8.4}
            {'—'*45}
            Number of holes in horizons:         {self.horizon.number_of_holes:8}
                                                {other.number_of_holes:8}
            {'—'*45}
            Additional traces labeled:           {at_1:8}
            (present in one, absent in other)    {at_2:8}
            {'—'*45}
            """
            printer(dedent(msg))

        if hist and not np.isnan(max_abs_error):
            _ = plt.hist(metric.ravel(), bins=100)

        title = 'Height differences between {} and {}'.format(self.horizon.name, other.name)
        plot_dict = {
            'spatial': True,
            'title': '{} on cube {}'.format(title, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': 0, 'zmax': np.max(metric),
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict




class GeometryMetrics(BaseSeismicMetric):
    """ !!. """
    AVAILABLE_METRICS = [
        'local_corrs', 'support_corrs',
        'local_kl', 'support_kl',
        'local_hellinger', 'support_hellinger',
        'local_wasserstein', 'support_wasserstein',
    ]


    def __init__(self, geometries):
        super().__init__()

        geometries = list(geometries) if isinstance(geometries, tuple) else geometries
        geometries = geometries if isinstance(geometries, list) else [geometries]
        self.geometries = geometries

        self.geometry = geometries[0]
        self._data = None
        self._probs = None
        self._bad_traces = None

        self.spatial = True
        self.name = 'hist_matrix'
        self.cube_name = self.geometry.name

    @property
    def data(self):
        """ !!. """
        if self._data is None:
            self._data = self.geometry.hist_matrix
        return self._data

    @property
    def bad_traces(self):
        """ !!. """
        if self._bad_traces is None:
            self._bad_traces = self.geometry.zero_traces
        return self._bad_traces


    @property
    def probs(self):
        """ !!. """
        if self._probs is None:
            self._probs = self.data / np.sum(self.data, axis=-1, keepdims=True) + self.EPS
        return self._probs


    def tracewise(self, func, l=3, pbar=True, **kwargs):
        """ !!. """
        if len(self.geometries) != 2:
            raise ValueError()
        pbar = tqdm if pbar else lambda iterator, *args, **kwargs: iterator
        metric = np.full((*self.geometry.uniques, l), np.nan)

        s_1 = self.geometries[0].dataframe['trace_index']
        s_2 = self.geometries[1].dataframe['trace_index']

        for idx, trace_index_1 in pbar(s_1.iteritems(), total=len(s_1)):
            trace_index_2 = s_2[idx]

            header = self.geometries[0].segyfile.header[trace_index_1]
            keys = [header.get(field) for field in self.geometries[0].fields]
            store_key = [self.geometries[0].vals_inversed[i][item] for i, item in enumerate(keys)]
            store_key = tuple(store_key)

            trace_1 = self.geometries[0].load_trace_segy(trace_index_1)
            trace_2 = self.geometries[1].load_trace_segy(trace_index_2)

            metric[store_key] = func(trace_1, trace_2, **kwargs)

        title = f"tracewise {func}"
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict


    def tracewise_unsafe(self, func, l=3, pbar=True, **kwargs):
        """ !!. """
        if len(self.geometries) != 2:
            raise ValueError()
        pbar = tqdm if pbar else lambda iterator, *args, **kwargs: iterator
        metric = np.full((*self.geometry.uniques, l), np.nan)

        for idx in pbar(range(len(self.geometries[0].dataframe))):
            header = self.geometries[0].segyfile.header[idx]
            keys = [header.get(field) for field in self.geometries[0].fields]
            store_key = [self.geometries[0].vals_inversed[i][item] for i, item in enumerate(keys)]
            store_key = tuple(store_key)

            trace_1 = self.geometries[0].load_trace_segy(idx)
            trace_2 = self.geometries[1].load_trace_segy(idx)
            metric[store_key] = func(trace_1, trace_2, **kwargs)

        title = f"tracewise unsafe {func}"
        plot_dict = {
            'spatial': self.spatial,
            'title': f'{title} for {self.name} on cube {self.cube_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            # **kwargs
        }
        return metric, plot_dict




# Njitted NumPy funcions
@njit
def geomean(array):
    """ !!. """
    n = np.sum(~np.isnan(array))
    return np.power(np.nanprod(array), (1 / n))

@njit
def harmean(array):
    """ !!. """
    n = np.sum(~np.isnan(array))
    return n / np.nansum(1 / array)

@njit
def histo_reduce(data, bins):
    """ !!. """
    i_range, x_range = data.shape[:2]

    hist_matrix = np.full((i_range, x_range, len(bins) - 1), np.nan)
    for il in prange(i_range):
        for xl in prange(x_range):
            hist_matrix[il, xl] = np.histogram(data[il, xl], bins=bins)[0]
    return hist_matrix


class NumbaNumpy:
    """ !!. """
    #pylint: disable = unnecessary-lambda, undefined-variable
    nanmin = njit()(lambda array: np.nanmin(array))
    nanmax = njit()(lambda array: np.nanmax(array))
    nanmean = njit()(lambda array: np.nanmean(array))
    nanstd = njit()(lambda array: np.nanstd(array))

    min = nanmin
    max = nanmax
    mean = nanmean
    std = nanstd

    geomean = geomean
    harmean = harmean

    histo_reduce = lambda data, bins: histo_reduce(data, bins)




# Functions to compute metric from data-array
def compute_local_func(function, name, data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    _ = kwargs

    reduce_func = getattr(NumbaNumpy, reduce_func)

    bad_traces = np.copy(bad_traces)
    bad_traces[np.std(data, axis=-1) == 0.0] = 1

    padded = np.pad(data, ((kernel_size, kernel_size), (kernel_size, kernel_size), (0, 0)), constant_values=np.nan)
    bad_traces = np.pad(bad_traces, kernel_size, constant_values=1.0)

    metric = apply_local_func(function, reduce_func, padded, bad_traces, kernel_size)
    metric = metric[kernel_size:-kernel_size, kernel_size:-kernel_size]
    title = f'local {name}'
    return metric, title


@njit
def apply_local_func(compute_func, reduce_func, data, bad_traces, kernel_size):
    """ !!. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate
    k = int(np.floor(kernel_size / 2))
    i_range, x_range = data.shape[:2]
    metric = np.full((i_range, x_range), np.nan)

    for il in prange(i_range):
        for xl in prange(x_range):
            if bad_traces[il, xl] == 0:
                trace = data[il, xl, :]

                metric_element = np.full((kernel_size, kernel_size), np.nan)
                for _idx in prange(-k, k+1):
                    for _jdx in prange(-k, k+1):
                        if bad_traces[il+_idx, xl+_jdx] == 0:
                            trace_ = data[il+_idx, xl+_jdx]
                            metric_element[k+_idx, k+_jdx] = compute_func(trace, trace_)
                metric_element[k, k] = np.nan

                if np.sum(~np.isnan(metric_element)):
                    metric[il, xl] = reduce_func(metric_element)
    return metric


def compute_support_func(function_ndarray, function_str, name,
                         data, supports, bad_traces, safe_strip=0, line_no=None, **kwargs):
    """ !!. """
    _ = kwargs
    bad_traces = np.copy(bad_traces)
    bad_traces[np.std(data, axis=-1) == 0.0] = 1

    if isinstance(supports, (int, tuple, list, np.ndarray)):
        if isinstance(supports, int):
            title = f'{name} with {supports} random supports'
            if safe_strip:
                bad_traces[:, :safe_strip], bad_traces[:, -safe_strip:] = 1, 1
                bad_traces[:safe_strip, :], bad_traces[-safe_strip:, :] = 1, 1

            non_zero_traces = np.where(bad_traces == 0)
            indices = np.random.choice(len(non_zero_traces[0]), supports)
            supports = np.array([non_zero_traces[0][indices], non_zero_traces[1][indices]]).T

        elif isinstance(supports, (tuple, list, np.ndarray)):
            title = f'{name} with {len(supports)} supports'
            if min(len(item) == 2 for item in supports) is False:
                raise ValueError('Each of `supports` sequence must contain coordinate of trace (il, xl). ')
            supports = np.array(supports)

        metric = function_ndarray(data, supports, bad_traces)

    elif isinstance(supports, str):
        if function_str is None:
            raise ValueError(f'{name} does not work in `line` mode!')

        title = f'{name} on {line_no} {supports}'
        if supports.startswith('i'):
            support_il = line_no or data.shape[0] // 2
            metric = function_str(data, bad_traces, support_il=support_il)

        if supports.startswith('x'):
            support_xl = line_no or data.shape[1] // 2
            metric = function_str(data, bad_traces, support_xl=support_xl)

    else:
        raise ValueError('`Supports` must be either int, sequence, ndarray or string. ')
    return metric, title



def compute_local_corrs(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_corrs, 'correlation',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

@njit
def _compute_local_corrs(array_1, array_2):
    """ !!. """
    result = np.sum((array_1 - np.mean(array_1)) * (array_2 - np.mean(array_2))) / (np.std(array_1) * np.std(array_2))
    return result / len(array_1)


def compute_support_corrs(data, supports, bad_traces, safe_strip=0, line_no=None, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_corrs,
                                function_str=_compute_line_corrs,
                                name='correlation',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, line_no=line_no, **kwargs)

def _compute_support_corrs(data, supports, bad_traces):
    """ NumPy function to compute correlations with a number of support traces. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    data_n = data - np.mean(data, axis=-1, keepdims=True)
    data_stds = np.std(data, axis=-1)

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]
    support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
    support_stds = np.std(support_traces, axis=-1)

    corrs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        cov = np.sum(support_n[i] * data_n, axis=-1) / depth
        temp = cov / (support_stds[i] * data_stds)
        temp[bad_traces == 1] = np.nan
        corrs[:, :, i] = temp
    return corrs

def _compute_line_corrs(data, bad_traces, support_il=None, support_xl=None):
    depth = data.shape[-1]

    data_n = data - np.mean(data, axis=-1, keepdims=True)
    data_stds = np.std(data, axis=-1)
    bad_traces[data_stds == 0] = 1

    if support_il is not None and support_xl is not None:
        raise ValueError('Use `compute_support_corrs` for given trace. ')

    if support_il is not None:
        support_traces = data[[support_il], :, :]
        support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
        support_stds = np.std(support_traces, axis=-1)
        bad_traces[:, support_stds[0, :] == 0] = 1
    if support_xl is not None:
        support_traces = data[:, [support_xl], :]
        support_n = support_traces - np.mean(support_traces, axis=-1, keepdims=True)
        support_stds = np.std(support_traces, axis=-1)
        bad_traces[support_stds[:, 0] == 0, :] = 1

    cov = np.sum(support_n * data_n, axis=-1) / depth
    corrs = cov / (support_stds * data_stds)
    corrs[bad_traces == 1] = 0
    return corrs


def compute_local_btch(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_btch, 'Bhattacharyya-divergence',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

@njit
def _compute_local_btch(array_1, array_2):
    """ !!. """
    return np.sum(np.sqrt(array_1 * array_2))


def compute_support_btch(data, supports, bad_traces, safe_strip=0, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_btch,
                                function_str=None,
                                name='Bhattacharyya-divergence',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, **kwargs)

def _compute_support_btch(data, supports, bad_traces):
    """ !!. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    divs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        supports_ = support_traces[i]
        temp = np.sum(np.sqrt(supports_ * data), axis=-1)
        temp[bad_traces == 1] = np.nan
        divs[:, :, i] = temp
    return divs




def compute_local_kl(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_kl, 'KL-divergence',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

@njit
def _compute_local_kl(array_1, array_2):
    """ !!. """
    return 1 - np.sum(array_1 * np.log2(array_1 / array_2))


def compute_support_kl(data, supports, bad_traces, safe_strip=0, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_kl,
                                function_str=None,
                                name='KL-divergence',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, **kwargs)

def _compute_support_kl(data, supports, bad_traces):
    """ !!. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    divs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        supports_ = support_traces[i]
        temp = 1 - np.sum(supports_ * np.log2(supports_/data), axis=-1)
        temp[bad_traces == 1] = np.nan
        divs[:, :, i] = temp
    return divs



def compute_local_js(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_js, 'JS-divergence',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

@njit
def _compute_local_js(array_1, array_2):
    """ !!. """
    average = (array_1 + array_2) / 2
    log_average = np.log2(average)
    div_1 = np.sum(array_1 * (np.log2(array_1) - log_average))
    div_2 = np.sum(array_2 * (np.log2(array_2) - log_average))
    return 1 - (div_1 + div_2) / 2


def compute_support_js(data, supports, bad_traces, safe_strip=0, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_js,
                                function_str=None,
                                name='JS-divergence',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, **kwargs)

def _compute_support_js(data, supports, bad_traces):
    """ !!. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    divs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        supports_ = support_traces[i]

        average = (supports_ + data) / 2
        log_average = np.log2(average)
        div_1 = np.sum(supports_ * (np.log2(supports_) - log_average), axis=-1)
        div_2 = np.sum(data * (np.log2(data) - log_average), axis=-1)
        temp = 1 - (div_1 + div_2) / 2
        temp[bad_traces == 1] = np.nan
        divs[:, :, i] = temp
    return divs



def compute_local_hellinger(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_hellinger, 'hellinger distance',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

SQRT_2 = np.sqrt(2)
@njit
def _compute_local_hellinger(array_1, array_2):
    """ !!. """
    return 1 - np.sqrt(np.sum(np.sqrt(array_1) - np.sqrt(array_2)) ** 2) / SQRT_2


def compute_support_hellinger(data, supports, bad_traces, safe_strip=0, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_hellinger,
                                function_str=None,
                                name='hellinger distance',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, **kwargs)

def _compute_support_hellinger(data, supports, bad_traces):
    """ !!. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    dist = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        temp = 1 - np.sqrt(np.sum((np.sqrt(support_traces[i]) - np.sqrt(data)) ** 2, axis=-1)) / SQRT_2
        temp[bad_traces == 1] = np.nan
        dist[:, :, i] = temp
    return dist



def compute_local_wasserstein(data, bad_traces, kernel_size=3, reduce_func='nanmean', **kwargs):
    """ !!. """
    return compute_local_func(_compute_local_wasserstein, 'wesserstein distance',
                              data=data, bad_traces=bad_traces,
                              kernel_size=kernel_size, reduce_func=reduce_func, **kwargs)

@njit
def _compute_local_wasserstein(array_1, array_2):
    """ !!. """
    sorter_1 = np.argsort(array_1)
    sorter_2 = np.argsort(array_2)

    concatted = np.concatenate((array_1, array_2))
    concatted = np.sort(concatted)
    deltas = np.diff(concatted)

    cdf_indices_1 = np.searchsorted(array_1[sorter_1], concatted[:-1], 'right')
    cdf_indices_2 = np.searchsorted(array_2[sorter_2], concatted[:-1], 'right')

    cdf_1 = cdf_indices_1 / array_1.size
    cdf_2 = cdf_indices_2 / array_2.size
    return 1 - np.sum(np.multiply(np.abs(cdf_1 - cdf_2), deltas))


def compute_support_wasserstein(data, supports, bad_traces, safe_strip=0, **kwargs):
    """ !!. """
    return compute_support_func(function_ndarray=_compute_support_wasserstein,
                                function_str=None,
                                name='wasserstein distance',
                                data=data, supports=supports, bad_traces=bad_traces,
                                safe_strip=safe_strip, **kwargs)

def _compute_support_wasserstein(data, supports, bad_traces):
    """ !!. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    support_traces = np.zeros((n_supports, depth))
    for i in range(n_supports):
        coord = supports[i]
        support_traces[i, :] = data[coord[0], coord[1], :]

    divs = np.zeros((i_range, x_range, n_supports))
    for i in range(n_supports):
        temp = emd_array(support_traces[i], data)
        temp[bad_traces == 1] = np.nan
        divs[:, :, i] = temp
    return divs

@njit
def emd_array(array_1d, array_3d):
    """ !!. """
    temp = np.zeros(array_3d.shape[:2])

    for i in prange(array_3d.shape[0]):
        for j in range(array_3d.shape[1]):
            temp[i, j] = _compute_local_wasserstein(array_1d, array_3d[i, j, :])
    return temp



def smooth_out(matrix, kernel_size=3, sigma=2.0, iters=3, **kwargs):
    """ !!. """
    _ = kwargs
    k = int(np.floor(kernel_size / 2))

    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    x_points, y_points = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(x_points) + np.square(y_points)) / np.square(sigma))
    gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))

    smoothed = np.copy(matrix)
    smoothed = np.pad(smoothed, kernel_size, constant_values=np.nan)

    for _ in range(iters):
        smoothed = apply_local_smoothing(smoothed, k, gaussian_kernel.ravel())
    smoothed = smoothed[kernel_size:-kernel_size, kernel_size:-kernel_size]
    smoothed[np.isnan(matrix)] = np.nan
    return smoothed

@njit
def apply_local_smoothing(matrix, k, raveled_kernel):
    """ !!. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate
    i_range, x_range = matrix.shape
    smoothed = np.full((i_range, x_range), np.nan)

    for iline in prange(i_range):
        for xline in prange(x_range):

            if not np.isnan(matrix[iline, xline]):
                element = matrix[iline-k:iline+k+1, xline-k:xline+k+1]

                s, sum_weights = 0.0, 0.0
                for item, weight in zip(element.ravel(), raveled_kernel):
                    if not np.isnan(item):
                        s += item * weight
                        sum_weights += weight

                if sum_weights != 0.0:
                    val = s / sum_weights
                    smoothed[iline, xline] = val

    return smoothed


def digitize(matrix, quantiles):
    """ !!. """
    bins = np.nanquantile(matrix.ravel(), np.sort(quantiles)[::-1])

    if len(bins) > 1:
        digitized = np.digitize(matrix, [*bins, np.nan]).astype(float)
    else:
        digitized = np.zeros_like(matrix, dtype=np.float64)
        digitized[matrix <= bins[0]] = 1.0

    digitized[np.isnan(matrix)] = np.nan
    return digitized


def gridify(matrix, frequencies, iline=True, xline=True):
    """ !!. """
    values = np.unique(matrix[~np.isnan(matrix)])
    if len(values) != len(frequencies):
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        multiplier = np.power(max_freq/min_freq, 1/(len(values) - 1))
        frequencies = [np.rint(max_freq / (multiplier ** i))
                       for i, _ in enumerate(values)]
    else:
        frequencies = np.sort(frequencies)[::-1]

    grid = np.zeros_like(matrix)
    for value, freq in zip(values, frequencies):
        idx_1, idx_2 = np.nonzero(matrix == value)

        if iline:
            mask = (idx_1 % freq == 0)
            grid[idx_1[mask], idx_2[mask]] = 1
        if xline:
            mask = (idx_2 % freq == 0)
            grid[idx_1[mask], idx_2[mask]] = 1

    grid[np.isnan(matrix)] = np.nan
    return grid



@njit
def correct_pi(horizon_phase, eps):
    """ Jit-accelerated function to <>. """
    for i in range(horizon_phase.shape[0]):
        prev = horizon_phase[i, 0]
        for j in range(1, horizon_phase.shape[1] - 1):
            if np.abs(np.abs(prev) - np.pi) <= eps and np.abs(np.abs(horizon_phase[i, j + 1]) - np.pi) <= eps:
                horizon_phase[i, j] = prev
            prev = horizon_phase[i, j]
    return horizon_phase