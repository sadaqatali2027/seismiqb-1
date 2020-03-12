""" Contains container for storing dataset of seismic crops. """
#pylint: disable=too-many-lines
import os
from glob import glob

import numpy as np

from ..batchflow import Dataset, Sampler
from ..batchflow import NumpySampler, ConstantSampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .horizon import Horizon, HorizonMetrics
from .utils import IndexedDict, lru_cache, round_to_array
from .plot_utils import show_sampler, plot_slide

from ..batchflow import Dataset, Sampler, Pipeline
from ..batchflow import B, V, D


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

        self.geometries = IndexedDict({ix: SeismicGeometry(self.index.get_fullpath(ix)) for ix in self.indices})
        self.labels = IndexedDict({ix: dict() for ix in self.indices})
        self.samplers = IndexedDict({ix: None for ix in self.indices})
        self.sampler = None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None


    def load_geometries(self, logs=True):
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
            self.geometries[ix].load()
            if logs:
                self.geometries[ix].log()
        return self

    def convert_to_h5py(self, postfix='', dtype=np.float32):
        """ Converts every cube in dataset from `.sgy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_h5py(postfix=postfix, dtype=dtype)
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
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: dict() for ix in self.indices}))

        for ix in self.indices:
            horizon_list = [Horizon(path, self.geometries[ix], **kwargs) for path in paths[ix]]
            horizon_list.sort(key=lambda horizon: horizon.h_mean)
            if filter_zeros:
                _ = [getattr(horizon, 'filter_points')() for horizon in horizon_list]
            getattr(self, dst)[ix] = horizon_list
        return self


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


    @lru_cache(3, storage=os.environ.get('SEISMIQB_CACHEDIR'), anchor=True, attributes='indices')
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

        self.load_geometries()
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros, dst=dst_labels)
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
        mask = getattr(self, src) if isinstance(src, str) else src
        horizons = Horizon.from_mask(mask, self.grid_info, threshold=threshold,
                                     averaging=averaging, minsize=minsize, prefix=prefix)
        setattr(self, dst, horizons)
        return self


    def merge_horizons(self, src, mean_threshold=2.0, q_threshold=2.0, q=0.9, adjacency=3, ):
        """ Iterate over a list of horizons and merge what can be merged. Can be called after
        running a pipeline with `get_point_cloud`-action. Changes the list of horizons inplace.

        Parameters
        ----------
        src : str or list
            Source-horizons. Can be either a name of attribute or list itself.
        height_margin : int
            if adjacent horizons do not diverge for more than this distance, they can be merged together.
        border_margin : int
            max distance between a pair of horizon-borders when the horizons can be adjacent.
        """
        # fetch list of horizons
        horizons = getattr(self, src) if isinstance(src, str) else src

        # iterate over list of horizons to merge what can be merged
        i = 0
        flag = True
        while flag:
            # the procedure continues while at least a pair of horizons is mergeable
            flag = False
            while True:
                if i >= len(horizons):
                    break

                j = i + 1
                while True:
                    # attempt to merge each horizon to i-th horizon with fixed i
                    if j >= len(horizons):
                        break

                    merge_code, _ = Horizon.verify_merge(horizons[i], horizons[j],
                                                         mean_threshold=mean_threshold, q_threshold=q_threshold, q=q,
                                                         adjacency=adjacency)
                    if merge_code in [2, 3]:
                        force_merge = (merge_code == 3)
                        merged, _, _ = Horizon.adjacent_merge(horizons[i], horizons[j], inplace=True,
                                                              force_merge=force_merge, adjacency=adjacency,
                                                              mean_threshold=mean_threshold,
                                                              q_threshold=q_threshold, q=q)
                        if merged:
                            _ = horizons.pop(j)
                            flag = True
                        else:
                            j += 1
                    else:
                        j += 1
                i += 1



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

    def subset_labels(self, points, crop_shape=(2, 64, 64), cube_index=0, show_prior_mask=False):
        """Save prior mask to a cubeset attribute `prior_mask`.
        Parameters
        ----------
        points : tuple or list
            upper left coordinates of the starting crop in the seismic cube coordinates.
        crop_shape : array-like
            shape of the saved prior mask.
        cube_index : int
            index of the cube in `ds.indices` list.
        show_prior_mask : bool
            whether to show prior mask
        """
        ds_points = np.array([[self.indices[cube_index], *points, None]])[:, :4]

        start_predict_pipeline = (Pipeline()
                                  .load_component(src=[D('geometries'), D('labels')],
                                                  dst=['geometries', 'labels'])
                                  .crop(points=ds_points, shape=crop_shape)
                                  .load_cubes(dst='images')
                                  .create_masks(dst='masks', width=1, horizons=1, src_labels='labels')
                                  .rotate_axes(src=['images', 'masks'])
                                  .add_axis(src='masks', dst='masks')
                                  .scale(mode='normalize', src='images')) << self

        batch = start_predict_pipeline.next_batch(len(self.indices), n_epochs=None)

        if show_prior_mask:
            batch.plot_components('masks')

        i_shift, x_shift, h_shift = [slices[0] for slices in batch.slices[0]]
        transforms = [lambda i_: self.geometries[self.indices[cube_index]].ilines[i_ + i_shift],
                      lambda x_: self.geometries[self.indices[cube_index]].xlines[x_ + x_shift],
                      lambda h_: h_ + h_shift]
        self.mask_to_horizons(np.moveaxis(batch.masks[0][:, :, :, 0], -1, 0),
                             threshold=0.5, dst='prior_mask', coordinates=None, transforms=transforms, separate=True)
        if len(self.prior_mask[0]) == 0:
            raise ValueError("Prior mask is empty")
        numba_horizon = convert_to_numba_dict(self.prior_mask[0])
        setattr(self, 'prior_mask', {self.indices[cube_index]: numba_horizon})
        return self

    def make_slice_prediction(self, model_pipeline, points, crop_shape, max_iters=10, width=10, stride=32,
                              cube_index=0, threshold=0.02, show_count=None, slide_direction='xline', mode='right'):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        """ Extend horizon on one slice by sequential predict on overlapping crops.
        Parameters
        ----------
        ds : Cubeset
            Instance of the Cubeset. Must have non-empy attributes `predicted labels` and `labels` (for debugging plots)
        points : tuple or list
            upper left coordinates of the starting crop in the seismic cube coordinates.
        crop_shape : array-like
            shape of the crop fed to the model.
        max_iters : int
            max_number of extension steps. If we meet end of the cube we will make less steps.
        width : int
            width of compared windows.
        stride : int
            stride size.
        cube_index : int
            index of the cube ds.indices.
        threshold : float
            threshold for predicted mask
        show_count : int
            Number of extension steps to show
        slide_direction : str
            Either `xline` or `iline`. Direction of the predicted slice.
        mode : str
            if left increase next point's line coordinates otherwise decrease it.
        Returns
        -------
        SeismicCubeset
            Same instance with updated `predicted_labels` and `grid_info` attributes.
        """
        show_count = max_iters if show_count is None else show_count
        geom = self.geometries[self.indices[cube_index]]
        grid_array = []
        if isinstance(points[0], (list, tuple)):
            max_iline = points[0][1] if points[0][1] is not None else geom.ilines_len
            max_xline = points[1][1] if points[1][1] is not None else geom.xlines_len
            points = [points[0][0], points[1][0], points[2][0]]
        else:
            max_iline, max_xline = geom.ilines_len, geom.xlines_len

        # compute strides for xline, iline cases
        line_stride = -stride if mode == 'left' else stride
        if slide_direction == 'iline':
            axes = (1, 0, 2)
            strides_candidates = [[line_stride, 0, -stride], [line_stride, 0, stride], [line_stride, 0, 0],
                                  [0, 0, -stride], [0, 0, stride]]
        elif slide_direction == 'xline':
            axes = (0, 1, 2)
            strides_candidates = [[0, line_stride, -stride], [0, line_stride, stride], [0, line_stride, 0],
                                  [0, 0, -stride], [0, 0, stride]]
        else:
            raise ValueError("Slide direction can be either iline or xline.")

        load_components_ppl = (Pipeline()
                               .load_component(src=[D('geometries'), D('labels')],
                                               dst=['geometries', 'labels'])
                               .add_components('predicted_labels'))

        predict_ppl = (Pipeline()
                       .load_component(src=[D('predicted_labels')], dst=['predicted_labels'])
                       .load_cubes(dst='images')
                       .create_masks(dst='masks', width=1, n_horizons=1, src_labels='labels')
                       .create_masks(dst='cut_masks', width=1, n_horizons=1, src_labels='predicted_labels')
                       .apply_transform(np.transpose, axes=axes, src=['images', 'masks', 'cut_masks'])
                       .rotate_axes(src=['images', 'masks', 'cut_masks'])
                       .analytic_transform(src='images', dst='hilbert')
                       .scale(mode='normalize', src='images')
                       .add_axis(src='masks', dst='masks')
                       .import_model('extension', model_pipeline)
                       .init_variable('result_preds', init_on_each_run=list())
                       .concat_components(src=('images', 'cut_masks'), dst='model_inputs')
                       .predict_model('extension', fetches='sigmoid',
                                      images=B('model_inputs'),
                                      save_to=V('result_preds', mode='e')))

        for i in range(max_iters):
            if (points[0] + crop_shape[0] > max_iline or
                    points[1] + crop_shape[1] > max_xline or points[2] + crop_shape[2] > geom.depth):
                print("End of the cube or area")
                break

            grid_array.append(points)
            ds_points = np.array([[self.indices[cube_index], *points, None]])[:, :4]
            crop_ppl = Pipeline().crop(points=ds_points, shape=crop_shape, passdown='predicted_labels')

            next_predict_pipeline = (load_components_ppl + crop_ppl + predict_ppl) << self
            btch = next_predict_pipeline.next_batch(len(self.indices), n_epochs=None)
            result = next_predict_pipeline.get_variable('result_preds')[0][..., 0]
            if np.sum(btch.images) < 1e-2:
                print('Empty traces')
                break

            # transform cube coordinates to ilines-xlines
            transforms = [lambda i_: self.geometries[self.indices[cube_index]].ilines[i_ + points[0]],
                          lambda x_: self.geometries[self.indices[cube_index]].xlines[x_ + points[1]],
                          lambda h_: h_ + points[2]]

            if slide_direction == 'iline':
                self.mask_to_horizons(np.moveaxis(result, -1, 1), threshold=threshold, dst='predicted_mask',
                                     coordinates=None, separate=True, transforms=transforms)
            else:
                self.mask_to_horizons(np.moveaxis(result, -1, 0), threshold=threshold, dst='predicted_mask',
                                     coordinates=None, separate=True, transforms=transforms)
            try:
                numba_horizons = convert_to_numba_dict(self.predicted_mask[0])
            except IndexError:
                print('Empty predicted mask on step %s' % i)
                break

            assembled_horizon_dict = update_horizon_dict(self.predicted_labels[self.indices[cube_index]],
                                                         numba_horizons)
            setattr(self, 'predicted_labels', {self.indices[cube_index]: assembled_horizon_dict})
            points, compared_slices_ = compute_next_points(points, result[:, :, 0].T,
                                                           crop_shape, strides_candidates, width)

            if i < show_count:
                print('----------------')
                print(i)
                print('argmax ', np.argmax(np.array(compared_slices_)))
                print('next stride ', strides_candidates[np.argmax(np.array(compared_slices_))])
                print('selected next points ', points)
                plot_extension_history(next_predict_pipeline, btch)
            if len(self.predicted_labels) == 0:
                break

        # assemble grid_info
        self.grid_info = {self.indices[cube_index]:
                          make_grid_info(grid_array, self.indices[cube_index], crop_shape)}
        return self

    def update_labels(self, src='predicted_labels', update_src='prior_mask', cube_index=0):
        """ Update dict-like component with another dict
        Parameters
        ----------
        src : str
            Component to be updated.
        update_src : str
            Component with a dict to add.
        """
        dict_update = getattr(self, update_src)[self.indices[cube_index]]
        if hasattr(self, src):
            dict_update = update_horizon_dict(dict_update, getattr(self, src)[self.indices[cube_index]])
        setattr(self, src, {self.indices[cube_index]: dict_update})
        return self
