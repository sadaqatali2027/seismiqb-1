""" Contains metrics for various labels (horizons, facies, etc) and cubes. """
from textwrap import dedent

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from scipy.signal import hilbert, medfilt

from ..batchflow.models.metrics import Metrics

from .horizon import Horizon
from .utils import compute_running_mean
from .plot_utils import plot_image



class HorizonMetrics(Metrics):
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
        if orientation is None: # metrics are computed on full cube (spatially)
            self._data = None # evaluated later
            self.bad_traces = np.copy(self.horizon.geometry.zero_traces)
            self.bad_traces[self.horizon.full_matrix == Horizon.FILL_VALUE] = 1
            self.spatial = True

        else: # metrics are computed on a specific slide
            self._data, self.bad_traces = self.horizon.get_cube_values_line(orientation=orientation, line=line,
                                                                            window=window, offset=offset, scale=scale)
            self.spatial = False


    @property
    def data(self):
        """ Create `data` attribute at the first time of evaluation. """
        if self._data is None:
            self._data = self.horizon.get_cube_values(window=self.window, offset=self.offset,
                                                      scale=self.scale, chunk_size=self.chunk_size)
        return self._data


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
            ignore_value = plot_dict.pop('ignore_value', None)
            spatial = plot_dict.pop('spatial', True)
            _ = backend, plot_kwargs, plot_dict.pop('zmin', -1), plot_dict.pop('zmax', 1)

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


    def local_corrs(self, locality=4, **kwargs):
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
        _ = kwargs

        if locality == 4:
            locs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        elif locality == 8:
            locs = [[-1, -1], [0, -1], [1, -1],
                    [-1, 0], [1, 0],
                    [-1, 1], [0, 1], [1, 1]]
        locs = np.array(locs)

        bad_traces = np.copy(self.bad_traces)
        bad_traces[np.std(self.data, axis=-1) == 0.0] = 1
        metric = _compute_local_corrs(self.data, bad_traces, locs)
        title = 'local correlation'

        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict


    def support_corrs(self, supports=1, safe_strip=0, line_no=None, **kwargs):
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
        _ = kwargs
        bad_traces = np.copy(self.bad_traces)
        bad_traces[np.std(self.data, axis=-1) == 0.0] = 1

        if isinstance(supports, (int, tuple, list, np.ndarray)):
            if isinstance(supports, int):
                title = 'correlation with {} random supports'.format(supports)
                if safe_strip:
                    bad_traces[:, :safe_strip], bad_traces[:, -safe_strip:] = 1, 1
                    bad_traces[:safe_strip, :], bad_traces[-safe_strip:, :] = 1, 1

                non_zero_traces = np.where(bad_traces == 0)
                indices = np.random.choice(len(non_zero_traces[0]), supports)
                supports = np.array([non_zero_traces[0][indices], non_zero_traces[1][indices]]).T

            elif isinstance(supports, (tuple, list, np.ndarray)):
                title = 'correlation with {} supports'.format(len(supports))
                if min(len(item) == 2 for item in supports) is False:
                    raise ValueError('Each of `supports` sequence must contain coordinate of trace (il, xl). ')
                supports = np.array(supports)

            metric = _compute_support_corrs_np(self.data, supports, bad_traces)

        elif isinstance(supports, str):
            title = 'correlation on {} {}'.format(line_no, supports)
            if supports.startswith('i'):
                support_il = line_no or self.data.shape[0] // 2
                metric = _compute_line_corrs_np(self.data, bad_traces, support_il=support_il)

            if supports.startswith('x'):
                support_xl = line_no or self.data.shape[1] // 2
                metric = _compute_line_corrs_np(self.data, bad_traces, support_xl=support_xl)

        else:
            raise ValueError('`Supports` must be either int, sequence, ndarray or string. ')

        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            'ignore_value': 0.0,
            # **kwargs
        }
        return metric, plot_dict


    def hilbert(self, mode='median', kernel_size=3, eps=1e-5, **kwargs):
        """ Compute phase along the horizon. """
        _ = kwargs
        full_matrix = self.horizon.full_matrix

        analytic = hilbert(self.data, axis=-1)
        phase = (np.angle(analytic))
        phase = phase % (2 * np.pi) - np.pi
        phase[full_matrix == Horizon.FILL_VALUE, :] = 0

        horizon_phase = phase[:, :, phase.shape[-1] // 2]
        horizon_phase = correct_pi(horizon_phase, eps)

        if mode == 'mean':
            median_phase = compute_running_mean(horizon_phase, kernel_size)
        else:
            median_phase = medfilt(horizon_phase, kernel_size)
        median_phase[full_matrix == Horizon.FILL_VALUE] = 0

        img = np.minimum(median_phase - horizon_phase, 2 * np.pi + horizon_phase - median_phase)
        img[full_matrix == Horizon.FILL_VALUE] = 0
        img = np.where(img < -np.pi, img + 2 * np. pi, img)

        metric = np.zeros((*img.shape, 2+self.data.shape[2]))
        metric[:, :, 0] = img
        metric[:, :, 1] = median_phase
        metric[:, :, 2:] = phase

        title = 'phase by {}'.format(mode)
        plot_dict = {
            'spatial': self.spatial,
            'title': '{} for {} on cube {}'.format(title, self.horizon.name, self.horizon.cube_name),
            'cmap': 'seismic',
            'zmin': -1, 'zmax': 1,
            # **kwargs
        }
        return metric, plot_dict


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


@njit
def _compute_local_corrs(data, bad_traces, locs):
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for il in range(i_range):
        for xl in range(x_range):
            if bad_traces[il, xl] == 0:
                trace = data[il, xl, :]

                s, c = 0.0, 0
                for i in range(len(locs)):
                    loc = locs[i]
                    il_, xl_ = il + loc[0], xl + loc[1]

                    if (0 <= il_ < i_range) and (0 <= xl_ < x_range):
                        if bad_traces[il_, xl_] == 0:
                            trace_ = data[il_, xl_, :]
                            s += np.corrcoef(trace, trace_)[0, 1]
                            c += 1
                if c != 0:
                    corrs[il, xl] = s / c
    return corrs


def _compute_support_corrs_np(data, supports, bad_traces):
    """ NumPy function to compute correlations with a number of support traces. """
    n_supports = len(supports)
    i_range, x_range, depth = data.shape

    data_n = data - np.mean(data, axis=-1, keepdims=True)
    data_stds = np.std(data, axis=-1)
    bad_traces[data_stds == 0] = 1

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
        temp[bad_traces == 1] = 0
        corrs[:, :, i] = temp
    return corrs


def _compute_line_corrs_np(data, bad_traces, support_il=None, support_xl=None):
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
