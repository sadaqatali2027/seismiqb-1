""" Contains container for storing dataset of seismic crops. """
import os
from glob import glob
from copy import copy

import dill
import numpy as np

from ..batchflow import Dataset, Sampler
from ..batchflow import HistoSampler, NumpySampler, ConstantSampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch
from .utils import read_point_cloud, filter_point_cloud, make_labels_dict, make_labels_dict_f, filter_labels
from .utils import _get_horizons, compare_horizons, dump_horizon, round_to_array
from .utils import horizon_to_depth_map, depth_map_to_labels, get_horizon_amplitudes, compute_corrs, FILL_VALUE_A
from .plot_utils import show_labels, show_sampler, plot_slide, plot_image



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


    def load_point_clouds(self, paths=None, path=None, **kwargs):
        """ Load point-cloud of labels for each cube in dataset.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.
        path : str
            Path to the dill-file to load point clouds from.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded point clouds.
        """
        if isinstance(path, str):
            with open(path, 'rb') as file:
                self.point_clouds = dill.load(file)
        else:
            for ix in self.indices:
                self.point_clouds[ix] = read_point_cloud(paths[ix], **kwargs)
                self.geometries[ix].horizon_list = paths[ix]
        return self

    def filter_point_clouds(self, src='point_clouds'):
        """ Remove points corresponding to zero-traces. """
        for ix in self.indices:
            geom = getattr(self, 'geometries').get(ix)
            ilines_offset, xlines_offset = geom.ilines_offset, geom.xlines_offset
            zero_matrix = geom.zero_traces
            getattr(self, src)[ix] = filter_point_cloud(getattr(self, src)[ix], zero_matrix,
                                                        ilines_offset, xlines_offset)

    def save_point_clouds(self, save_to):
        """ Save dill-serialized point clouds for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.point_clouds, file)
        return self


    def create_labels(self, path=None, transforms=None, src='point_clouds', dst='labels'):
        """ Make labels in inline-xline coordinates using cloud of points and supplied transforms.

        Parameters
        ----------
        path : str
            Path to the dill-file to load labels from.
        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from (i, x, h, n) to (i, x, d, n) and take array of shape (N, 4) as input,
            where d (depth) is corrected h (height) (divided by sample rate and moved by time-delay value).
        src : str
            Attribute with saved point clouds.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded labels.
        """
        point_clouds = getattr(self, src) if isinstance(src, str) else src
        transforms = transforms or dict()
        if not hasattr(self, dst):
            setattr(self, dst, {})

        if isinstance(path, str):
            try:
                with open(path, 'rb') as file:
                    setattr(self, dst, dill.load(file))
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
        else:
            for ix in self.indices:
                point_cloud = point_clouds.get(ix)
                geom = getattr(self, 'geometries').get(ix)
                transform = transforms.get(ix) or geom.height_correction
                if point_cloud.shape[1] == 4: # horizon: il, xl, height, index
                    getattr(self, dst)[ix] = make_labels_dict(transform(point_cloud))
                elif point_cloud.shape[1] == 5: # facies: il, xl, start_point, end_point, index
                    transformed_pc = copy(point_cloud)
                    transformed_pc[:, [0, 1, 2, 4]] = (transform(point_cloud[:, [0, 1, 2, 4]]).astype(np.int64))
                    transformed_pc[:, 3] = (transform(point_cloud[:, [4, 4, 3, 4]])[:, 2].astype(np.int64))
                    getattr(self, dst)[ix] = make_labels_dict_f(transformed_pc)
        return self

    def filter_labels(self, src='labels'):
        """ Remove labels corresponding to zero-traces.

        Parameters
        ----------
        src : str
            Attribute with labels-dictionary.
        """
        for ix in self.indices:
            geom = getattr(self, 'geometries').get(ix)
            ilines_offset, xlines_offset = geom.ilines_offset, geom.xlines_offset
            zero_matrix = geom.zero_traces
            filter_labels(getattr(self, src)[ix], zero_matrix, ilines_offset, xlines_offset)

    def save_labels(self, save_to, src='labels'):
        """ Save dill-serialized labels for a dataset of seismic-cubes on disk. """
        if isinstance(save_to, str):
            try:
                with open(save_to, 'wb') as file:
                    dill.dump(getattr(self, src), file)
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
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
        for ix in self.indices:
            labels = getattr(self, src_labels).get(ix)
            geom = self.geometries[ix]

            save_dir = os.path.join(os.path.dirname(geom.path), dir_name)
            try:
                os.mkdir(save_dir)
            except FileExistsError:
                pass

            for idx, path in enumerate(geom.horizon_list):
                name = os.path.basename(path)
                name = os.path.join(save_dir, os.path.basename(path))
                dump_horizon(labels, geom, name, idx=idx, offset=0)

    def show_labels(self, idx=0, hor_idx=None):
        """ Draw points with hand-labeled horizons from above. """
        show_labels(self, idx=idx, hor_idx=hor_idx)


    def create_sampler(self, mode='hist', p=None, transforms=None, dst='sampler', **kwargs):
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
            elif mode[ix] == 'hist':
                point_cloud = getattr(self, 'point_clouds')[ix]

                if point_cloud.shape[1] == 5:
                    point_cloud = np.vstack([point_cloud[:, 0],
                                             point_cloud[:, 1]
                                             (point_cloud[:, 2] + point_cloud[:, 3])//2,
                                             point_cloud[:, -1]]).T

                geom = getattr(self, 'geometries')[ix]
                offsets = np.array([geom.ilines_offset, geom.xlines_offset, 0])
                cube_shape = np.array(geom.cube_shape)
                to_cube = lambda points: (points[:, :3] - offsets)/cube_shape
                default = lambda points: to_cube(geom.height_correction(points))

                transform = transforms.get(ix) or default
                cube_array = transform(point_cloud)

                # Size of ticks along each respective axis
                default_bins = cube_shape // np.array([5, 20, 20])
                bins = kwargs.get('bins') or default_bins
                sampler = HistoSampler(np.histogramdd(cube_array, bins=bins))
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
                arr = (array[:, axis+1]*shapes).astype(int)
                array[:, axis+1] = round_to_array(arr, ticks) / shapes
                return array

            sampler = sampler.apply(filter_out)

        # Change representation of points from unit cube to cube coordinates
        if to_cube:
            def get_shapes(name):
                return self.geometries[name].cube_shape

            def coords_to_cube(array):
                shapes = np.array(list(map(get_shapes, array[:, 0])))
                array[:, 1:] = (array[:, 1:] * shapes).astype(int)
                return array

            sampler = sampler.apply(coords_to_cube)

        # Apply additional transformations to points
        if callable(post):
            sampler = sampler.apply(post)

        if finish:
            setattr(self, dst, sampler.sample)
        else:
            setattr(self, dst, sampler)

    def show_sampler(self, idx=0, src_sampler='sampler', n=100000, eps=3):
        """ Generate a lot of points and look at their (iline, xline) positions.

        Parameters
        ----------
        idx : int
            Number of cube to show sampler on.
        src_sampler : str
            Name of attribute with sampler in it.
            Must generate points in cubic coordinates, which can be achieved by `modify_sampler` method.
        n : int
            Number of points to generate.
        eps : int
            Window of painting.
        """
        show_sampler(self, idx=idx, src_sampler=src_sampler, n=n, eps=eps)


    def load(self, horizon_dir=None, p=None, filter_zeros=True):
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
        horizon_dir = horizon_dir or '/BEST_HORIZONS/*'

        paths_txt = {}
        for i in range(len(self)):
            dir_path = '/'.join(self.index.get_fullpath(self.indices[i]).split('/')[:-1])
            dir_ = dir_path + horizon_dir
            paths_txt[self.indices[i]] = glob(dir_)

        self.load_geometries()
        self.load_point_clouds(paths=paths_txt)
        if filter_zeros:
            self.filter_point_clouds()
        self.create_labels()
        self.create_sampler(p=p)
        return self

    def save_attr(self, name, save_to):
        """ Save attribute of dataset to disk. """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(getattr(self, name), file)
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
        setattr(self, dst, _get_horizons(mask, threshold, averaging, transforms, separate))

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
        """ Show full slide of the given cube on the given iline. """
        components = ('images', 'masks') if list(self.labels.values())[0] else ('images',)
        plot_slide(self, *components, idx=idx, n_line=n_line, plot_mode=plot_mode, mode=mode, **kwargs)


    def apply_to_horizon(self, idx=0, horizon_idx=0, labels_src=None, transform=None):
        """ Apply specific transform to individual horizon inside dictionary with labels.
        Under the hood, horizon is converted into depth map, which is essentially a matrix in
        (xline, inline) coordinates with values corresponding to height of the horizon at the given point,
        then transform is applied, and then depth map is converted back into horizon.
        Note that this method changes values of horizon inplace.

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
        transform : callable
            Function to apply to depth map.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]

        depth_map = horizon_to_depth_map(labels, geom, horizon_idx, 0)
        if callable(transform):
            depth_map = transform(depth_map)
        depth_map_to_labels(depth_map, geom, labels, horizon_idx)


    def show_horizon_depth_map(self, idx=0, horizon_idx=0, labels_src=None, _return=False):
        """ Show depth map of a horizon.

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
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]
        hor_name = os.path.basename(geom.horizon_list[horizon_idx])

        depth_map = horizon_to_depth_map(labels, geom, horizon_idx)
        depth_map[depth_map == FILL_VALUE_A] = 0

        plot_image(depth_map, 'Heights {} on cube {}'.format(hor_name, cube_name), cmap='seismic')
        print('Average value of height is {}'.format(np.mean(depth_map[depth_map != FILL_VALUE_A])))

        if _return:
            return depth_map
        return None


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
        print('Average value of height is {}'.format(np.mean(depth_map[depth_map != FILL_VALUE_A])))
        print('Std of amplitudes is {}'.format(np.std(data[depth_map != FILL_VALUE_A])))

        if _return:
            return data
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
        data[depth_map == FILL_VALUE_A, :] = 0
        data = data[:, :, ::-1]
        data *= np.asarray([1, 0.5, 0.25]).reshape(1, 1, -1)

        plot_image(data, 'RGB horizon {} on cube {}'.format(hor_name, cube_name), rgb=True)
        print('AVG', np.mean(depth_map[depth_map != FILL_VALUE_A]))


    def compute_horizon_corrs(self, idx=0, horizon_idx=0, labels_src=None, window=3, _return=False):
        """ Compute correlations with the nearest traces along the horizon.

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
        window : int
            Width of trace used for computing correlations.
        """
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        labels = labels_src or self.labels[cube_name]
        geom = self.geometries[cube_name]
        hor_name = os.path.basename(geom.horizon_list[horizon_idx])

        data, depth_map = get_horizon_amplitudes(labels, geom, horizon_idx, window, 0)
        corrs = compute_corrs(data)
        corrs[np.where(depth_map == FILL_VALUE_A)] = 0

        plot_image(corrs, 'Correlation for {} on cube {}'.format(hor_name, cube_name), cmap='seismic')
        print('Average correlation is {}'.format(np.mean(corrs[depth_map != FILL_VALUE_A])))

        if _return:
            return corrs
        return None


    def compare_horizons(self, hor_1, hor_2, hor_1_idx=0, hor_2_idx=0, idx=0, axis=-1, cmap='Set1', _return=False):
        """ Compare two horizons on l1 metric and on derivative differences.

        Parameters
        ----------
        idx : int
            Number of cube to use.
        hor_1, hor_2 : dict
            Dictionaries with labeled horizons.
        hor_1_idx, hor_2_idx : int
            Index of used horizon from each respective dictionary.
        axis : int
            Axis to take derivative along.
        cmap : str
            Colormap of showing the results.
        """
        #pylint: disable=comparison-with-callable
        cube_name = idx if isinstance(idx, str) else self.indices[idx]
        geom = self.geometries[cube_name]

        # Create all depth maps
        depth_map_1 = horizon_to_depth_map(hor_1, geom, hor_1_idx, 0)
        depth_map_2 = horizon_to_depth_map(hor_2, geom, hor_2_idx, 0)
        indicator = (np.minimum(depth_map_1, depth_map_2) == FILL_VALUE_A).astype(int)
        depth_map_1[np.where(indicator == 1)] = 0
        depth_map_2[np.where(indicator == 1)] = 0

        # l1: mean absolute difference between depth maps
        metric_1 = np.abs(depth_map_1 - depth_map_2)
        metric_1[np.where(indicator == 1)] = 0
        plot_image(metric_1, 'l1 metric on cube {}'.format(cube_name), cmap=cmap)
        print('Average value of l1 is {}\n\n'.format(np.mean(metric_1[np.where(indicator == 0)])))

        # ~: mean absolute difference between gradients of depth maps
        metric_2 = np.abs(np.diff(depth_map_1, axis=axis, prepend=0) - np.diff(depth_map_2, axis=axis, prepend=0))
        metric_2[np.where(indicator == 1)] = 0
        plot_image(metric_2, '~ metric on cube {}'.format(cube_name), cmap=cmap)
        print('Average value of ~ is {}'.format(np.mean(metric_2[np.where(indicator == 0)])))

        if _return:
            return metric_1, metric_2
        return None
