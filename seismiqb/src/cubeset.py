""" Contains container for storing dataset of seismic crops. """
#pylint: disable=too-many-lines
from glob import glob

import numpy as np

from ..batchflow import Dataset, Sampler, DatasetIndex
from ..batchflow import NumpySampler, ConstantSampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .horizon import Horizon, UnstructuredHorizon
from .metrics import HorizonMetrics
from .utils import IndexedDict, round_to_array
from .plot_utils import show_sampler, plot_slide, plot_image



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
        """ Initialize additional attributes. """
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.crop_index, self.crop_points = None, None

        self.geometries = IndexedDict({ix: SeismicGeometry(self.index.get_fullpath(ix), process=False)
                                       for ix in self.indices})
        self.labels = IndexedDict({ix: dict() for ix in self.indices})
        self.samplers = IndexedDict({ix: None for ix in self.indices})
        self._sampler = None
        self._p, self._bins = None, None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None


    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                  bar=False, bar_desc=None, iter_params=None, sampler=None):
        """ Allows to pass `sampler` directly to `next_batch` method to avoid re-creating of batch
        during pipeline run.
        """
        #pylint: disable=blacklisted-name
        if n_epochs is not None or shuffle or drop_last:
            raise ValueError('SeismicCubeset does not comply with `n_epochs`, `shuffle`\
                              and `drop_last`. Use `n_iters` instead! ')
        if sampler:
            sampler = sampler if callable(sampler) else sampler.sample
            points = sampler(batch_size * n_iters)

            self.crop_points = points
            self.crop_index = DatasetIndex(points[:, 0])
            return self.crop_index.gen_batch(batch_size, n_iters=n_iters, iter_params=iter_params,
                                             bar=bar, bar_desc=bar_desc)
        return super().gen_batch(batch_size, shuffle=shuffle, n_iters=n_iters, n_epochs=n_epochs,
                                 drop_last=drop_last, bar=bar, bar_desc=bar_desc, iter_params=iter_params)


    def load_geometries(self, logs=True, **kwargs):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        for ix in self.indices:
            self.geometries[ix].process(**kwargs)
            if logs:
                self.geometries[ix].log()
        return self

    def convert_to_hdf5(self, postfix=''):
        """ Converts every cube in dataset from `.segy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_hdf5(postfix=postfix)
        return self


    def create_labels(self, paths=None, filter_zeros=True, dst='labels', labels_class=None, **kwargs):
        """ Create labels (horizons, facies, etc) from given paths.

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
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: dict() for ix in self.indices}))

        for ix in self.indices:
            if labels_class is None:
                if self.geometries[ix].structured:
                    labels_class = Horizon
                else:
                    labels_class = UnstructuredHorizon

            horizon_list = [labels_class(path, self.geometries[ix], **kwargs) for path in paths[ix]]
            horizon_list.sort(key=lambda horizon: horizon.h_mean)
            if filter_zeros:
                _ = [getattr(horizon, 'filter_points')() for horizon in horizon_list]
            getattr(self, dst)[ix] = horizon_list
        return self

    @property
    def sampler(self):
        """ Lazily create sampler at the time of first access. """
        if self._sampler is None:
            self.create_sampler(p=self._p, bins=self._bins)
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        self._sampler = sampler


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
            def filter_out(array):
                for cube_name in np.unique(array[:, 0]):
                    shape = self.geometries[cube_name].cube_shape[axis]
                    ticks = np.arange(each_start, shape, each)
                    name_idx = np.asarray(array[:, 0] == cube_name).nonzero()

                    arr = np.rint(array[array[:, 0] == cube_name][:, axis+1].astype(float)*shape).astype(int)
                    array[name_idx, np.full_like(name_idx, axis+1)] = round_to_array(arr, ticks).astype(float) / shape
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

    def show_slices(self, idx=0, src_sampler='sampler', n=10000, normalize=False, shape=None,
                    make_slices=True, side_view=False, **kwargs):
        """ Show actually sampled slices of desired shape. """
        sampler = getattr(self, src_sampler)
        if callable(sampler):
            #pylint: disable=not-callable
            points = sampler(n)
        else:
            points = sampler.sample(n)
        batch = (self.p.crop(points=points, shape=shape, make_slices=make_slices, side_view=side_view)
                 .next_batch(self.size))

        unsalted = np.array([batch.unsalt(item) for item in batch.indices])
        background = np.zeros_like(self.geometries[idx].zero_traces)

        for slice_ in np.array(batch.slices)[unsalted == self.indices[idx]]:
            idx_i, idx_x, _ = slice_
            background[idx_i, idx_x] += 1

        if normalize:
            background = (background > 0).astype(int)
        plot_image(background, f'Sampled slices on {self.indices[idx]}', rgb=normalize, **kwargs)
        return batch


    def load(self, horizon_dir=None, filter_zeros=True, dst_labels='labels', p=None, bins=None, **kwargs):
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

        self.load_geometries(**kwargs)
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros, dst=dst_labels)
        self._p, self._bins = p, bins # stored for later sampler creation
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
        self.grid_info = {
            'grid_array': grid_array,
            'predict_shape': predict_shape,
            'crop_shape': crop_shape,
            'cube_name': cube_name,
            'geom': geom,
            'range': [ilines_range, xlines_range, h_range]
        }
        return self


    def mask_to_horizons(self, src, threshold=0.5, averaging='mean', minsize=0,
                         dst='predicted_horizons', prefix='predict'):
        """ Convert mask to a list of horizons.

        Parameters
        ----------
        src : str or array
            Source-mask. Can be either a name of attribute or mask itself.
        dst : str
            Attribute to write the horizons in.
        threshold : float
            Parameter of mask-thresholding.
        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).
        minsize : int
            Minimum length of a horizon to be saved.
        prefix : str
            Name of horizon to use.
        """
        #TODO: add `chunks` mode
        mask = getattr(self, src) if isinstance(src, str) else src
        horizons = Horizon.from_mask(mask, self.grid_info, threshold=threshold,
                                     averaging=averaging, minsize=minsize, prefix=prefix)
        setattr(self, dst, horizons)
        return self


    def merge_horizons(self, src, mean_threshold=2.0, adjacency=3, minsize=50):
        """ !!. """
        horizons = getattr(self, src)
        horizons = Horizon.merge_list(horizons, mean_threshold=mean_threshold, adjacency=adjacency, minsize=minsize)
        if isinstance(src, str):
            setattr(self, src, horizons)


    def compare_to_labels(self, horizon, src_labels='labels', offset=0, absolute=True,
                          printer=print, hist=True, plot=True):
        """ Compare given horizon to labels in dataset.

        Parameters
        ----------
        horizon : :class:`.Horizon`
            Horizon to evaluate.
        offset : number
            Value to shift horizon down. Can be used to take into account different counting bases.
        """
        for idx in self.indices:
            if horizon.geometry.name == self.geometries[idx].name:
                horizons_to_compare = getattr(self, src_labels)[idx]
                break
        HorizonMetrics([horizon, horizons_to_compare]).evaluate('compare', agg=None,
                                                                absolute=absolute, offset=offset,
                                                                printer=printer, hist=hist, plot=plot)


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
