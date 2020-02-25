""" Contains container for storing dataset of seismic crops. """
#pylint: disable=too-many-lines
import os
from glob import glob

import dill
import numpy as np

from ..batchflow import Dataset, Sampler
from ..batchflow import NumpySampler, ConstantSampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from ._const import FILL_VALUE_MAP
from .utils import round_to_array, lru_cache
from .plot_utils import show_sampler, plot_slide, plot_image, plot_image_roll

from .horizon import get_horizon_amplitudes, get_line_horizon_amplitudes
from .horizon import compute_local_corrs, compute_support_corrs, compute_hilbert
from .horizon import mask_to_horizon, compare_horizons
from .horizon import Horizon


class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures.

    Attributes
    ----------
    geometries : dict
        Mapping from cube names to instances of :class:`~.SeismicGeometry`, which holds information
        about that cube structure. :meth:`~.load_geometries` is used to infer that structure.
        Note that no more that one trace is loaded into the memory at a time.

    labels : dict
        Mapping from cube names to numba-dictionaries, which are mappings from (xline, iline) pairs
        into arrays of heights of horizons for a given cube.
        Note that this arrays preserve order: i-th horizon is always placed into the i-th element of the array.
    """
    #pylint: disable=too-many-public-methods
    def __init__(self, index, batch_class=SeismicCropBatch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes.
        """
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.geometries = {ix: SeismicGeometry(self.index.get_fullpath(ix)) for ix in self.indices}
        self.point_clouds = {ix: np.array([]) for ix in self.indices}
        self.labels = {ix: dict() for ix in self.indices}
        self.samplers = {ix: None for ix in self.indices}
        self.sampler = None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None


    def load_geometries(self, path=None, logs=True):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        path : str
            Path to the dill-file to load geometries from.
        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        if isinstance(path, str):
            with open(path, 'rb') as file:
                self.geometries = dill.load(file)
        else:
            for ix in self.indices:
                self.geometries[ix].load()
                if logs:
                    self.geometries[ix].log()
        return self

    def convert_to_h5py(self, postfix='', dtype=np.float32):
        """ Converts every cube in dataset from `.sgy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_h5py(postfix=postfix, dtype=dtype)
        return self

    def save_geometries(self, save_to):
        """ Save dill-serialized geometries for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.geometries, file)
        return self


    def create_labels(self, paths=None, filter_zeros=True, dst='labels', **kwargs):
        """ Make labels in inline-xline coordinates using cloud of points and supplied transforms.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.
        dst : str
            Name of attribute to put labels in.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded labels.
        """
        for ix in self.indices:
            horizon_list = [Horizon(path, self.geometries[ix], **kwargs) for path in paths[ix]]
            horizon_list.sort(key=lambda horizon: horizon.h_mean)
            if filter_zeros:
                _ = [getattr(horizon, 'filter_points')() for horizon in horizon_list]
            getattr(self, dst)[ix] = horizon_list
        return self

    def dump_labels(self, dir_name=None, src_labels='labels'):
        """ Dump labels-dict into separate txt files.

        Parameters
        ----------
        src_labels : str
            Attribute with dictionary: (iline, xline) -> array of heights.
            Each entry in the array is saved into separate file.
        dir_name : str
            Relative (to cube location) path to directory with saved horizons.
        """


    def create_sampler(self, mode='hist', p=None, transforms=None, dst='sampler', **kwargs):
        """ Create samplers for every cube and store it in `samplers`
        attribute of passed dataset. Also creates one combined sampler
        and stores it in `sampler` attribute of passed dataset.

        Parameters
        ----------
        mode : str or Sampler
            Type of sampler to be created.
            If 'hist' or 'horizon', then sampler is estimated from given labels.
            If 'numpy', then sampler is created with `kwargs` parameters.
            If instance of Sampler is provided, it must generate points from unit cube.
        p : list
            Weights for each mixture in final sampler.
        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from absolute coordinates (X, Y world-wise) to
            cube local specific and take array of shape (N, 4) as input.

        Notes
        -----
        Passed `dataset` must have `geometries` and `labels` attributes if you want to create HistoSampler.
        """
        #pylint: disable=cell-var-from-loop
        lowcut, highcut = [0, 0, 0], [1, 1, 1]
        transforms = transforms or dict()

        samplers = {}
        if not isinstance(mode, dict):
            mode = {ix: mode for ix in self.indices}

        for ix in self.indices:
            if isinstance(mode[ix], Sampler):
                sampler = mode[ix]

            elif mode[ix] == 'numpy':
                sampler = NumpySampler(**kwargs)

            elif mode[ix] == 'hist' or mode[ix] == 'horizon':
                sampler = 0 & NumpySampler('n', dim=3)
                for i, horizon in enumerate(self.labels[ix]):
                    horizon.create_sampler(**kwargs)
                    sampler = sampler | horizon.sampler
            else:
                sampler = NumpySampler('u', low=0, high=1, dim=3)

            sampler = sampler.truncate(low=lowcut, high=highcut)
            samplers.update({ix: sampler})
        self.samplers = samplers

        # One sampler to rule them all
        p = p or [1/len(self) for _ in self.indices]

        sampler = 0 & NumpySampler('n', dim=4)
        for i, ix in enumerate(self.indices):
            sampler_ = (ConstantSampler(ix)
                        & samplers[ix].apply(lambda d: d.astype(np.object)))
            sampler = sampler | (p[i] & sampler_)
        setattr(self, dst, sampler)
        return self

    def modify_sampler(self, dst, mode='iline', low=None, high=None,
                       each=None, each_start=None,
                       to_cube=False, post=None, finish=False, src='sampler'):
        """ Change given sampler to generate points from desired regions.

        Parameters
        ----------
        src : str
            Attribute with Sampler to change.
        dst : str
            Attribute to store created Sampler.
        mode : str
            Axis to modify: ilines/xlines/heights.
        low : float
            Lower bound for truncating.
        high : float
            Upper bound for truncating.
        each : int
            Keep only i-th value along axis.
        each_start : int
            Shift grid for previous parameter.
        to_cube : bool
            Transform sampled values to each cube coordinates.
        post : callable
            Additional function to apply to sampled points.
        finish : bool
            If False, instance of Sampler is put into `dst` and can be modified later.
            If True, `sample` method is put into `dst` and can be called via `D` named-expressions.

        Examples
        --------
        Split into train / test along ilines in 80/20 ratio:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='i', high=0.8)
        >>> cubeset.modify_sampler(dst='test_sampler', mode='i', low=0.9)

        Sample only every 50-th point along xlines starting from 70-th xline:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='x', each=50, each_start=70)

        Notes
        -----
        It is advised to have gap between `high` for train sampler and `low` for test sampler.
        That is done in order to take into account additional seen entries due to crop shape.
        """

        # Parsing arguments
        sampler = getattr(self, src)

        mapping = {'ilines': 0, 'xlines': 1, 'heights': 2,
                   'iline': 0, 'xline': 1, 'i': 0, 'x': 1, 'h': 2}
        axis = mapping[mode]

        low, high = low or 0, high or 1
        each_start = each_start or each

        # Keep only points from region
        if (low != 0) or (high != 1):
            sampler = sampler.truncate(low=low, high=high, prob=high-low,
                                       expr=lambda p: p[:, axis+1])

        # Keep only every `each`-th point
        if each is not None:
            def get_shape(name):
                return self.geometries[name].cube_shape[axis]

            def get_ticks(name):
                shape = self.geometries[name].cube_shape[axis]
                return np.arange(each_start, shape, each)

            def filter_out(array):
                shapes = np.array(list(map(get_shape, array[:, 0])))
                ticks = np.array(list(map(get_ticks, array[:, 0])))
                arr = np.rint(array[:, axis+1].astype(float)*shapes).astype(int)
                array[:, axis+1] = round_to_array(arr, ticks) / shapes
                return array

            sampler = sampler.apply(filter_out)

        # Change representation of points from unit cube to cube coordinates
        if to_cube:
            def get_shapes(name):
                return self.geometries[name].cube_shape

            def coords_to_cube(array):
                shapes = np.array(list(map(get_shapes, array[:, 0])))
                array[:, 1:] = np.rint(array[:, 1:].astype(float) * shapes).astype(int)
                return array

            sampler = sampler.apply(coords_to_cube)

        # Apply additional transformations to points
        if callable(post):
            sampler = sampler.apply(post)

        if finish:
            setattr(self, dst, sampler.sample)
        else:
            setattr(self, dst, sampler)

    def show_sampler(self, idx=0, src_sampler='sampler', n=100000, eps=3, show_unique=False, **kwargs):
        """ Generate a lot of points and look at their (iline, xline) positions.

        Parameters
        ----------
        idx : str, int
            If str, then name of cube to use.
            If int, then number of cube in the index to use.
        src_sampler : str
            Name of attribute with sampler in it.
            Must generate points in cubic coordinates, which can be achieved by `modify_sampler` method.
        n : int
            Number of points to generate.
        eps : int
            Window of painting.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        geom = self.geometries[cube_name]
        sampler = getattr(self, src_sampler)
        show_sampler(sampler, cube_name, geom, n=n, eps=eps, show_unique=show_unique, **kwargs)



    @lru_cache(1024, storage=os.environ.get('SEISMIQB_CUBESET_CACHEDIR'), anchor=True)
    def load(self, horizon_dir=None, p=None, bins=None, filter_zeros=True, **kwargs):
        """ Load everything: geometries, point clouds, labels, samplers.

        Parameters
        ----------
        horizon_dir : str
            Relative path from each cube to directory with horizons.
        p : sequence of numbers
            Proportions of different cubes in sampler.
        filter_zeros : bool
            Whether to remove labels on zero-traces.
        """
        _ = kwargs
        horizon_dir = horizon_dir or '/BEST_HORIZONS/*'

        paths_txt = {}
        for i in range(len(self)):
            dir_path = '/'.join(self.index.get_fullpath(self.indices[i]).split('/')[:-1])
            dir_ = dir_path + horizon_dir
            paths_txt[self.indices[i]] = glob(dir_)

        self.load_geometries()
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros)
        self.create_sampler(p=p, bins=bins)
        return self




    def make_grid(self, cube_name, crop_shape, ilines_range, xlines_range, h_range, strides=None, batch_size=16):
        """ Create regular grid of points in cube.
        This method is usually used with `assemble_predict` action of SeismicCropBatch.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.
        crop_shape : array-like
            Shape of model inputs.
        ilines_range : array-like of two elements
            Location of desired prediction, iline-wise.
        xlines_range : array-like of two elements
            Location of desired prediction, xline-wise.
        h_range : array-like of two elements
            Location of desired prediction, depth-wise.
        strides : array-like
            Distance between grid points.
        batch_size : int
            Amount of returned points per generator call.

        Returns
        -------
        SeismicCubeset
            Same instance with grid generator and grid information in attributes.

        """
        geom = self.geometries[cube_name]
        strides = strides or crop_shape

        # Assert ranges are valid
        if ilines_range[0] < 0 or \
           xlines_range[0] < 0 or \
           h_range[0] < 0:
            raise ValueError('Ranges must contain in the cube.')

        if ilines_range[1] >= geom.ilines_len or \
           xlines_range[1] >= geom.xlines_len or \
           h_range[1] >= geom.depth:
            raise ValueError('Ranges must contain in the cube.')

        # Make separate grids for every axis
        def _make_axis_grid(axis_range, stride, length, crop_shape):
            grid = np.arange(*axis_range, stride)
            grid_ = [x for x in grid if x + crop_shape < length]
            if len(grid) != len(grid_):
                grid_ += [axis_range[1] - crop_shape]
            return sorted(grid_)

        ilines = _make_axis_grid(ilines_range, strides[0], geom.ilines_len, crop_shape[0])
        xlines = _make_axis_grid(xlines_range, strides[1], geom.xlines_len, crop_shape[1])
        hs = _make_axis_grid(h_range, strides[2], geom.depth, crop_shape[2])

        # Every point in grid contains reference to cube
        # in order to be valid input for `crop` action of SeismicCropBatch
        grid = []
        for il in ilines:
            for xl in xlines:
                for h in hs:
                    point = [cube_name, il, xl, h]
                    grid.append(point)
        grid = np.array(grid, dtype=object)

        # Creating and storing all the necessary things
        grid_gen = (grid[i:i+batch_size]
                    for i in range(0, len(grid), batch_size))

        offsets = np.array([min(grid[:, 1]),
                            min(grid[:, 2]),
                            min(grid[:, 3])])

        predict_shape = (ilines_range[1] - ilines_range[0],
                         xlines_range[1] - xlines_range[0],
                         h_range[1] - h_range[0])

        grid_array = grid[:, 1:].astype(int) - offsets

        self.grid_gen = lambda: next(grid_gen)
        self.grid_iters = - (-len(grid) // batch_size)
        self.grid_info = {'grid_array': grid_array,
                          'predict_shape': predict_shape,
                          'crop_shape': crop_shape,
                          'cube_name': cube_name,
                          'range': [ilines_range, xlines_range, h_range]}
        return self


    def get_point_cloud(self, src, dst, threshold=0.5, averaging='mean', coordinates='cubic', separate=True):
        """ Compute point cloud of horizons from a mask, save it into the 'cubeset'-attribute.

        Parameters
        ----------
        src : str or array
            Source-mask. Can be either a name of attribute or mask itself.
        dst : str
            Attribute of `cubeset` to write the horizons in.
        threshold : float
            Parameter of mask-thresholding.
        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).
        coordinates : str
            Coordinates to use for keys of point-cloud. Can be either 'cubic'
            'lines' or None. In case of None, mask-coordinates are used. Mode 'cubic'
            requires 'grid_info'-attribute; can be run after `make_grid`-method. Mode 'lines'
            requires both 'grid_info' and 'geometries'-attributes to be loaded.
        separate : bool
            Whether to write horizonts in separate dictionaries or in one common.

        Returns
        -------
        dict or list of dict
            If separate is False, then one dictionary returned with keys being pairs of
            (iline, xline) and values being lists of heights.
            If separate is True, then list of dictionaries is returned, with every dictionary being
            mapping from pairs of (iline, xline) to height from each individual horizont.
        """
        # fetch mask-array
        mask = getattr(self, src) if isinstance(src, str) else src

        # prepare coordinate-transforms
        if coordinates is None:
            transforms = [lambda x: x for _ in range(3)]
        elif coordinates == 'cubic':
            shifts = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda x_, shift=shift: x_ + shift for shift in shifts]
        elif coordinates == 'lines':
            geom = self.geometries[self.grid_info['cube_name']]
            i_shift, x_shift, h_shift = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda i_: geom.ilines[i_ + i_shift], lambda x_: geom.xlines[x_ + x_shift],
                          lambda h_: h_ + h_shift]

        # get horizons
        setattr(self, dst, mask_to_horizon(mask, threshold, averaging, transforms, separate))

        if separate:
            horizons = getattr(self, dst)
            horizons.sort(key=len, reverse=True)
            for i, horizon in enumerate(horizons):
                setattr(self, dst+'_'+str(i), horizon)
        return self


    def compare_to_labels(self, horizon, idx=0, offset=1, plot=True):
        """ Compare given horizon to labels in dataset.

        Parameters
        ----------
        horizon : dict
            Mapping from (iline, xline) to heights.
        idx : str, int
            If str, then name of cube to use.
            If int, then number of cube in the index to use.
        offset : number
            Value to shift horizon up. Can be used to take into account different counting bases.
        plot : bool
            Whether to plot histogram of errors.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = self.labels[cube_name]
        sample_rate = self.geometries[cube_name].sample_rate

        compare_horizons(horizon, labels, printer=print, plot=plot,
                         sample_rate=sample_rate, offset=offset)
        return self


    def show_slide(self, idx=0, n_line=0, plot_mode='overlap', mode='iline', **kwargs):
        """ Show full slide of the given cube on the given line.

        Parameters
        ----------
        idx : str, int
            Number of cube in the index to use.
        mode : str
            Axis to cut along. Can be either `iline` or `xline`.
        n_line : int
            Number of line to show.
        plot_mode : str
            Way of showing results. Can be either `overlap`, `separate`, `facies`.
        """
        components = ('images', 'masks') if list(self.labels.values())[0] else ('images',)
        plot_slide(self, *components, idx=idx, n_line=n_line, plot_mode=plot_mode, mode=mode, **kwargs)



    def show_horizon_amplitudes(self, idx=0, horizon_idx=0, labels_src=None, scale=False, _return=False):
        """ Show trace values on a horizon.

        Parameters
        ----------
        idx : str, int
            If str, then name of cube to use.
            If int, then number of cube in the index to use.
        horizon_idx : int
            Index of used horizon from `labels` dictionary.
        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.
        scale : bool, callable
            If bool, then values are scaled to [0, 1] range.
            If callable, then it is applied to iline-oriented slices of data from the cube.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]
        hor_name = os.path.basename(geom.horizon_list[horizon_idx])

        data, depth_map = get_horizon_amplitudes(labels, geom, horizon_idx, 1, scale=scale)

        plot_image(data, 'Horizon {} on cube {}'.format(hor_name, cube_name), cmap='seismic')
        print('Average value of height is {}'.format(np.mean(depth_map[depth_map != FILL_VALUE_MAP])))
        print('Std of amplitudes is {}'.format(np.std(data[depth_map != FILL_VALUE_MAP])))

        if _return:
            return {'data': data, 'depth_map': depth_map}
        return None


    def show_horizon_rgb(self, idx=0, horizon_idx=0, labels_src=None, width=1):
        """ Show trace values on the horizon and surfaces directly under it.

        Parameters
        ----------
        idx : str, int
            If str, then name of cube to use.
            If int, then number of cube in the index to use.
        horizon_idx : int
            Index of used horizon from `labels` dictionary.
        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.
        width : int
            Space between surfaces to cut.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]
        hor_name = os.path.basename(geom.horizon_list[horizon_idx])

        data, depth_map = get_horizon_amplitudes(labels, geom, horizon_idx, 1 + width*2, width)
        data = data[:, :, (0, width, -1)]
        data -= data.min(axis=(0, 1)).reshape(1, 1, -1)
        data *= 1 / data.max(axis=(0, 1)).reshape(1, 1, -1)
        data[depth_map == FILL_VALUE_MAP, :] = 0
        data = data[:, :, ::-1]
        data *= np.asarray([1, 0.5, 0.25]).reshape(1, 1, -1)

        plot_image(data, 'RGB horizon {} on cube {}'.format(hor_name, cube_name), rgb=True)
        print('AVG', np.mean(depth_map[depth_map != FILL_VALUE_MAP]))



    def compute_horizon_metric(self, idx=0, horizon_idx=0, labels_src=None, data_slice=None,
                               window=3, mode='local_corrs',
                               filter_zero_traces=True, _fill_value=0, aggregate=None,
                               show=True, savefig=False, show_plot=True, show_scalar=True,
                               _return=False, **kwargs):
        """ Compute metric along the horizon amplitudes.
        Generally, this function does following:
            Cuts data from the cube values along the horizon surface.
            Applies metric computation function to the cut data.
            Filters locations of zero traces, if needed.
            Aggregates metric into (n_ilines, n_xlines) shape, if needed.
            Plots visual representation of metric, if needed.

        Parameters
        ----------
        idx : str, int
            If str, then name of cube to use.
            If int, then number of cube in the index to use.
        horizon_idx : int
            Index of used horizon from `labels` dictionary.
        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon dictionary.
        data_slice : None or slice
            Slice to cut from data from the cube.
        window : int
            Width of trace used for computing metric.
        mode : str or callable
            Type of metric to compute. Can be either callable, `local`, `support` or `hilbert`:
                If callable, then function applied to data along the horizon. Must have following
                signature: (data, depth_map, filtering_matrix, **kwargs).

                If `local`, then for each trace average of correlation with nearest 4 or 8 traces is computed.
                Additional parameter `locality` can be passed via `kwargs`:
                    locality : {4, 8}
                        Defines number of nearest traces to average correlations from.

                If `support`, then compute correlations between one or multiple fixed traces and rest of the cube.
                Additional parameters `supports`, `safe_strip`, `line_no` can be passed via `kwargs`:
                    supports : int, sequence, ndarray or str
                        Defines mode of generating support traces.
                        If int, then that number of random non-zero traces positions are generated.
                        If sequence or array, then must be of shape (N, 2) and is used as positions of support traces.
                        If str, then must defines either `iline` or `xline` mode. In each respective one,
                        given iline/xline is used to generate supports.
                    safe_strip : int
                        Used only for `int` mode of `supports` parameter and defines minimum distance
                        from borders for sampled points.
                    line_no : int
                        Used only for `str` mode of `supports` parameter to define exact iline/xline to use.

                If `hilbert`, then analytic transform is performed to get phases along the horizon.
                Additional parameters `hilbert_mode`, `kernel_size`, `eps` can be passed via kwargs:
                    hilbert_mode : str
                        Way of averaging phase along trace. Can either be `median` or `mean`.
                    kernel_size : int
                        Kernel size of averaging.
                    eps : float
                        Tolerance of pi normalization.

        filter_zero_traces : bool
            Whether to fill points of zero traces with `_fill_value`.
        aggregate : int, str or callable
            Function to transform metric from ndarray of (n_ilines, n_xlines, N) shape to (n_ilines, n_xlines) shape.
            If callable, then directly applied to the output of metric computation function.
            If str, then must be a function from `numpy` module. Applied along the last axis only.
            If int, then index of slice along the last axis to return.
        show : bool
            Whether to create image of metric.
        savefig : bool or str
            If str, then path for image saving.
            If False, then image is not saved.
        show_plot: bool
            Whether to show created image in output stream.
        show_scalar : bool
            Whether to show averaged value.
        _return : bool
            Whether to return metric ndarray.
        **kwargs : dict
            Other named arguments.
        """
        #pylint: disable=too-many-branches
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]

        data, depth_map = get_horizon_amplitudes(labels, geom, horizon_idx, window, 0)

        if data_slice:
            data = data[data_slice]
            depth_map = depth_map[data_slice]

        bad_traces = np.copy(geom.zero_traces)
        bad_traces[np.where(depth_map == FILL_VALUE_MAP)] = 1

        if callable(mode):
            metric = mode(data, depth_map, bad_traces, **kwargs)
            title = 'custom metric'

        elif mode in ['local_corrs'] or 'local' in mode:
            metric = compute_local_corrs(data, bad_traces, **kwargs)
            title = 'local correlation'

        elif 'support' in mode:
            metric = compute_support_corrs(data, bad_traces, **kwargs)

            supports = kwargs.get('supports', 1)
            if isinstance(supports, int):
                title = 'correlation with {} random supports'.format(supports)
            elif isinstance(supports, (tuple, list, np.ndarray)):
                title = 'correlation with {} supports'.format(len(supports))
            elif isinstance(supports, str):
                title = 'correlation along {} {}'.format(kwargs.get('line_no', 'middle'), supports)

        elif 'hilbert' in mode:
            metric = compute_hilbert(data, depth_map, **kwargs)
            title = 'phase by {}'.format(kwargs.get('hilbert_mode', 'median'))

        if filter_zero_traces:
            metric[np.where(depth_map == FILL_VALUE_MAP)] = _fill_value

        if aggregate is not None:
            if callable(aggregate):
                metric = aggregate(metric)
            elif isinstance(aggregate, str):
                metric = getattr(np, aggregate)(metric, axis=-1)
            elif isinstance(aggregate, (int, slice)):
                metric = metric[:, :, aggregate]

        if show:
            hor_name = os.path.basename(geom.horizon_list[horizon_idx])
            plot_image_roll(metric, '{} for {} on cube {}'.format(title, hor_name, cube_name),
                            cmap='seismic', savefig=savefig, show_plot=show_plot)
        if show_scalar:
            scalar_metric = np.mean(metric[depth_map != FILL_VALUE_MAP])
            print('{} aggregated into single scalar is {:.3} (computed only on non-zero traces)'
                  .format(title, scalar_metric))
        if _return:
            return {'metric': metric, 'data': data, 'depth_map': depth_map}
        return None

    def compute_horizon_corrs(self, idx=0, horizon_idx=0, labels_src=None, window=3, **kwargs):
        """ Compute correlations with the nearest (locally) traces along the horizon.
        Alias for `compute_horizon_metric` method with some predefined parameters. """
        return self.compute_horizon_metric(idx=idx, horizon_idx=horizon_idx, labels_src=labels_src, window=window,
                                           mode='local_corrs', **kwargs)


    def compute_horizon_random_corrs(self, idx=0, horizon_idx=0, labels_src=None, window=3, supports=20,
                                     aggregate=None, **kwargs):
        """ Compute correlations with the nearest (locally) traces along the horizon.
        Alias for `compute_horizon_metric` method with some predefined parameters."""
        return self.compute_horizon_metric(idx=idx, horizon_idx=horizon_idx, labels_src=labels_src, window=window,
                                           mode='support', supports=supports, aggregate=aggregate, **kwargs)


    def compute_horizon_phase(self, idx=0, horizon_idx=0, labels_src=None, window=3, hilbert_mode='median',
                              aggregate=1, **kwargs):
        """ Compute phase along the horizon.
        Alias for `compute_horizon_metric` method with some predefined parameters."""
        return self.compute_horizon_metric(idx=idx, horizon_idx=horizon_idx, labels_src=labels_src, window=window,
                                           mode='hilbert', hilbert_mode=hilbert_mode, aggregate=aggregate, **kwargs)


    def compute_line_horizon_metric(self, idx=0, horizon_idx=0, labels_src=None, data_slice=None,
                                    orientation='iline', line=None, window=3, mode='support',
                                    filter_zero_traces=True, _fill_value=0, aggregate=None,
                                    show=True, savefig=False, show_plot=True, show_scalar=True,
                                    _return=False, **kwargs):
        """ Compute metric along the horizon along given line (either inline or xline).

        Parameters
        ----------
        orientation : str
            Iline or xline orientation to slice along.
        line : int
            Number of line to cut.
        other arguments : dict
            Same arguments as for :meth:`.compute_horizon_metric` method.
        """
        #pylint: disable=too-many-branches
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]

        slide, data, depth_map, zero_traces = get_line_horizon_amplitudes(labels, geom, horizon_idx,
                                                                          orientation, line, window, 0)
        if data_slice:
            data = data[data_slice]
            depth_map = depth_map[data_slice]

        bad_traces = np.copy(zero_traces)
        bad_traces[0, np.where(depth_map == FILL_VALUE_MAP)] = 1

        if callable(mode):
            metric = mode(data, depth_map, zero_traces, **kwargs)
            title = 'custom metric'

        elif mode in ['local_corrs'] or 'local' in mode:
            metric = compute_local_corrs(data, zero_traces, **kwargs)
            title = 'local correlation'

        elif 'support' in mode:
            metric = compute_support_corrs(data, zero_traces, **kwargs)

            supports = kwargs.get('supports', 1)
            if isinstance(supports, int):
                title = 'correlation with {} random supports'.format(supports)
            elif isinstance(supports, (tuple, list, np.ndarray)):
                title = 'correlation with {} supports'.format(len(supports))

        elif 'hilbert' in mode:
            metric = compute_hilbert(data, depth_map, **kwargs)
            title = 'phase by {}'.format(kwargs.get('hilbert_mode', 'median'))

        metric = np.squeeze(metric)

        if filter_zero_traces:
            metric[np.where(depth_map == FILL_VALUE_MAP)] = _fill_value

        if aggregate is not None:
            if callable(aggregate):
                metric = aggregate(metric)
            elif isinstance(aggregate, str):
                metric = getattr(np, aggregate)(metric, axis=-1)
            elif isinstance(aggregate, (int, slice)):
                metric = metric[:, :, aggregate]

        _ = title, show, savefig, show_plot

        if show_scalar:
            scalar_metric = np.mean(metric[depth_map != FILL_VALUE_MAP])
            print('{} aggregated into single scalar is {:.3} (computed only on non-zero traces)'
                  .format(title, scalar_metric))

        if _return:
            return {'metric': metric, 'slide': slide, 'data': data, 'depth_map': depth_map, 'zero_traces': zero_traces}
        return None
