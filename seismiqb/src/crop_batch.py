""" Seismic Crop Batch."""
import string
import random
from copy import copy

import numpy as np
import segyio
import cv2
from scipy.signal import butter, lfilter, hilbert

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel
from ..batchflow.batch_image import transform_actions # pylint: disable=no-name-in-module,import-error
from .utils import create_mask, create_mask_f, aggregate
from .horizon import mask_to_horizon, check_if_joinable, merge_horizon_into_another
from .plot_utils import plot_batch_components



AFFIX = '___'
SIZE_POSTFIX = 7
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX



@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class SeismicCropBatch(Batch):
    """ Batch with ability to generate 3d-crops of various shapes."""
    components = ('slices',)

    def _init_component(self, *args, **kwargs):
        """ Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if isinstance(dst, str):
            dst = (dst,)
        for comp in dst:
            if not hasattr(self, comp):
                setattr(self, comp, np.array([None] * len(self.index)))
        return self.indices


    @staticmethod
    def salt(path):
        """ Adds random postfix of predefined length to string.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        path : str
            supplied string with random postfix.
        Notes
        -----
        Action `crop` makes a new instance of SeismicCropBatch with
        different (enlarged) index. Items in that index should point to cube
        location to cut crops from. Since we can't store multiple copies of the same
        string in one index (due to internal usage of dictionary), we need to augment
        those strings with random postfix (which we can remove later).
        """
        chars = string.ascii_uppercase + string.digits
        return path + AFFIX + ''.join(random.choice(chars) for _ in range(SIZE_POSTFIX))

    @staticmethod
    def has_salt(path):
        """ Check whether path is salted. """
        return path[::-1].find(AFFIX) == SIZE_POSTFIX

    @staticmethod
    def unsalt(path):
        """ Removes postfix that was made by `salt` method.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        str
            string without postfix.
        """
        if AFFIX in path:
            return path[:-SIZE_SALT]
        return path

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        return super().__getattr__(name)

    def get(self, item=None, component=None):
        """ Overload `get` in order to use it for some attributes (that are looking like geometries or labels). """
        if sum([attribute in component for attribute in ['label', 'geom']]):
            if isinstance(item, str) and self.has_salt(item):
                item = self.unsalt(item)
            res = getattr(self, component)
            if isinstance(res, dict) and item in res:
                return res[item]
            return res

        item = self.get_pos(None, component, item)
        return super().get(item, component)

    @action
    def load_component(self, src, dst):
        """ Store `src` data in `dst` component. """
        if isinstance(src, dict):
            src = [src]
        if isinstance(dst, str):
            dst = [dst]

        for data, name in zip(src, dst):
            setattr(self, name, data)
        return self


    @action
    def crop(self, points, shape, dilations=(1, 1, 1), loc=(0, 0, 0), side_view=False, dst='slices', passdown=None):
        """ Generate positions of crops. Creates new instance of `SeismicCropBatch`
        with crop positions in one of the components (`slices` by default).

        Parameters
        ----------
        points : array-like
            Upper rightmost points for every crop and name of cube to
            cut it from. Order is: name, iline, xline, height. For example,
            ['Cube.sgy', 13, 500, 200] stands for crop has [13, 500, 200]
            as its upper rightmost point and must be cut from 'Cube.sgy' file.
        shape : sequence
            Desired shape of crops.
        dilations : sequence
            Intervals between successive slides along each dimension.
        loc : sequence of numbers
            Location of the point relative to the cut crop. Must be a location on unit cube.
        side_view : bool or float
            Determines whether to generate crops of transposed shape (xline, iline, height).
            If False, then shape is never transposed.
            If True, then shape is transposed with 0.5 probability.
            If float, then shape is transposed with that probability.
        dst : str, optional
            Component of batch to put positions of crops in.
        passdown : str of list of str
            Components of batch to keep in the new one.

        Notes
        -----
        Based on the first column of `points`, new instance of SeismicCropBatch is created.
        In order to keep multiple references to the same .sgy cube, each index is augmented
        with prefix of fixed length (check `salt` method for details).

        Returns
        -------
        SeismicCropBatch
            Batch with positions of crops in specified component.
        """
        new_index = [self.salt(ix) for ix in points[:, 0]]
        new_dict = {ix: self.index.get_fullpath(self.unsalt(ix))
                    for ix in new_index}
        new_batch = type(self)(FilesIndex.from_index(index=new_index, paths=new_dict, dirs=False))

        passdown = passdown or []
        passdown = [passdown] if isinstance(passdown, str) else passdown
        passdown.extend(['geometries', 'labels'])

        for component in passdown:
            if hasattr(self, component):
                setattr(new_batch, component, getattr(self, component))

        if side_view:
            side_view = side_view if isinstance(side_view, float) else 0.5
        shape = np.asarray(shape)
        shapes = []
        for _ in points:
            if not side_view:
                shapes.append(shape)
            else:
                flag = np.random.random() > side_view
                if flag:
                    shapes.append(shape)
                else:
                    shapes.append(shape[[1, 0, 2]])
        shapes = np.array(shapes)

        slices = []
        for point, shape_ in zip(points, shapes):
            slice_ = self._make_slice(point, shape_, dilations, loc)
            slices.append(slice_)
        setattr(new_batch, dst, slices)
        return new_batch

    def _make_slice(self, point, shape, dilations, loc=(0, 0, 0)):
        """ Creates list of `np.arange`'s for desired location. """
        if isinstance(point[1], float) or isinstance(point[2], float) or isinstance(point[3], float):
            ix = point[0]
            cube_shape = np.array(self.get(ix, 'geometries').cube_shape)
            slice_point = np.rint(point[1:].astype(float) * (cube_shape - np.array(shape))).astype(int)
        else:
            slice_point = point[1:]

        slice_ = []
        for i in range(3):
            start_point = int(max(slice_point[i] - loc[i]*shape[i]*dilations[i], 0))
            end_point = start_point + shape[i]*dilations[i]
            slice_.append(np.arange(start_point, end_point, dilations[i]))
        return slice_

    @property
    def crop_shape(self):
        """ Shape of crops, made by action `crop`. """
        _, shapes_count = np.unique([image.shape for image in self.images], return_counts=True, axis=0)
        if len(shapes_count) == 1:
            return self.images[0].shape
        raise RuntimeError('Crops have different shapes')

    @property
    def crop_shape_dice(self):
        """ Extended crop shape. Useful for model with Dice-coefficient as loss-function. """
        return (*self.crop_shape, 1)


    @action
    def load_cubes(self, dst, fmt='h5py', src='slices', view=None):
        """ Load data from cube in given positions.

        Parameters
        ----------
        fmt : 'h5py' or 'sgy'
            Cube storing format.
        src : str
            Component of batch with positions of crops to load.
        dst : str
            Component of batch to put loaded crops in.

        Returns
        -------
        SeismicCropBatch
            Batch with loaded crops in desired component.
        """
        if fmt.lower() in ['sgy', 'segy']:
            _ = view
            return self._load_cubes_sgy(src=src, dst=dst)
        if fmt.lower() in ['h5py', 'h5']:
            return self._load_cubes_h5py(src=src, dst=dst, view=view)

        return self


    def _sgy_init(self, *args, **kwargs):
        """ Create `dst` component and preemptively open all the .sgy files.
        Should always be used in pair with `_sgy_post`!

        Note
        ----
        This init function is helpful for actions that work directly with .sgy
        files through `segyio` API: all the file handlers are created only once per batch,
        rather than once for every item in the batch.
        """
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if isinstance(dst, str):
            dst = (dst,)
        for comp in dst:
            if not hasattr(self, comp):
                setattr(self, comp, np.array([None] * len(self.index)))

        segyfiles = {}
        for ix in self.indices:
            path_data = self.index.get_fullpath(ix)
            if segyfiles.get(self.unsalt(ix)) is None:
                segyfile = segyio.open(path_data, 'r', strict=False)
                segyfile.mmap()
                segyfiles[self.unsalt(ix)] = segyfile
        return [dict(ix=ix, segyfile=segyfiles[self.unsalt(ix)])
                for ix in self.indices]


    def _sgy_post(self, segyfiles, *args, **kwargs):
        """ Close opened .sgy files."""
        _, _ = args, kwargs
        for segyfile in segyfiles:
            segyfile.close()
        return self


    def _stitch_clouds(self, all_clouds, *args, dst=None, height_margin=2, border_margin=1, **kwargs):
        """ Stitch a set of point-clouds to a point cloud form dst if possible.
        Post for `get_point_cloud`-action.
        """
        _, _ = args, kwargs
        if dst is None:
            raise ValueError("dst should be initialized with empty list.")

        # remember, all_clouds contains lists of horizons
        for horizons_set in all_clouds:
            for horizon_candidate in horizons_set:
                for horizon_target in dst:
                    if check_if_joinable(horizon_candidate, horizon_target, height_margin=height_margin,
                                         border_margin=border_margin):
                        merge_horizon_into_another(horizon_candidate, horizon_target)
                        break
                else:
                    # if a horizon cannot be stitched to a horizon from dst, we enrich dst with it
                    dst.append(horizon_candidate)
        return self


    @inbatch_parallel(init='_sgy_init', post='_sgy_post', target='threads')
    def _load_cubes_sgy(self, ix, segyfile, dst, src='slices'):
        """ Load data from .sgy-cube in given positions. """
        geom = self.get(ix, 'geometries')
        slice_ = self.get(ix, src)
        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]

        crop = np.zeros((len(ilines_), len(xlines_), len(hs_)))
        for i, iline_ in enumerate(ilines_):
            for j, xline_ in enumerate(xlines_):
                il_, xl_ = geom.ilines[iline_], geom.xlines[xline_]
                try:
                    tr_ = geom.il_xl_trace[(il_, xl_)]
                    crop[i, j, :] = segyfile.trace[tr_][hs_]
                except KeyError:
                    pass

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = crop
        return segyfile


    @inbatch_parallel(init='_init_component', target='threads')
    def _load_cubes_h5py(self, ix, dst, src='slices', view=None):
        """ Load data from .hdf5-cube in given positions. """
        geom = self.get(ix, 'geometries')
        slice_ = self.get(ix, src)

        if view is None:
            slice_lens = np.array([len(item) for item in slice_])
            axis = np.argmin(slice_lens)
        else:
            mapping = {0: 0, 1: 1, 2: 2,
                       'i': 0, 'x': 1, 'h': 2,
                       'iline': 0, 'xline': 1, 'height': 2, 'depth': 2}
            axis = mapping[view]

        if axis == 0:
            crop = self.__load_h5py_i(geom, *slice_)
        elif axis == 1 and 'cube_x' in geom.h5py_file:
            crop = self.__load_h5py_x(geom, *slice_)
        elif axis == 2 and 'cube_h' in geom.h5py_file:
            crop = self.__load_h5py_h(geom, *slice_)
        else: # backward compatibility
            crop = self.__load_h5py_i(geom, *slice_)

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = crop
        return self

    def __load_h5py_i(self, geom, ilines, xlines, heights):
        h5py_cube = geom.h5py_file['cube']
        dtype = h5py_cube.dtype

        crop = np.zeros((len(ilines), len(xlines), len(heights)), dtype=dtype)
        for i, iline in enumerate(ilines):
            slide = self.__load_slide(h5py_cube, iline)
            crop[i, :, :] = slide[xlines, :][:, heights]
        return crop

    def __load_h5py_x(self, geom, ilines, xlines, heights):
        h5py_cube = geom.h5py_file['cube_x']
        dtype = h5py_cube.dtype

        crop = np.zeros((len(ilines), len(xlines), len(heights)), dtype=dtype)
        for i, xline in enumerate(xlines):
            slide = self.__load_slide(h5py_cube, xline)
            crop[:, i, :] = slide[heights, :][:, ilines].transpose([1, 0])
        return crop

    def __load_h5py_h(self, geom, ilines, xlines, heights):
        h5py_cube = geom.h5py_file['cube_h']
        dtype = h5py_cube.dtype

        crop = np.zeros((len(ilines), len(xlines), len(heights)), dtype=dtype)
        for i, height in enumerate(heights):
            slide = self.__load_slide(h5py_cube, height)
            crop[:, :, i] = slide[ilines, :][:, xlines]
        return crop

    def __load_slide(self, cube, index):
        """ (Potentially) cached function for slide loading.

        Notes
        -----
        One must use thread-safe cache implementation.
        """
        return cube[index, :, :]

    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def create_masks(self, ix, dst, src='slices', mode='horizon', width=3, src_labels='labels', n_horizons=-1):

        """ Create masks from labels-dictionary in given positions.

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.
        dst : str
            Component of batch to put loaded masks in.
        mode : str
            Either `horizon` or `stratum`.
            Type of created mask. If `horizon` then only horizons, i.e. borders
            between geological strata will be loaded. In this case binary is created.
            If  `stratum` then every stratum between horizons in the point-cloud
            dictionary will be labeled with different class. Classes are in range from
            1 to number_of_horizons + 1.
        width : int
            Width of horizons in the `horizon` mode.
        src_labels : str
            Component of batch with labels dict.
        n_horizons : int or array-like of ints
            Maximum number of horizons per crop.
            If -1, all possible horizons will be added.
            If array-like then elements are interpreted as indices of the desired horizons
            and must be ints in range [0, len(horizons) - 1].
            Note if you want to pass an index of a single horizon it must a list with one
            element.

        Returns
        -------
        SeismicCropBatch
            Batch with loaded masks in desired components.

        Notes
        -----
        Can be run only after labels-dict is loaded into labels-component.
        """
        #pylint: disable=protected-access
        geom = self.get(ix, 'geometries')
        il_xl_h = self.get(ix, src_labels)

        slice_ = self.get(ix, src)
        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]
        if not hasattr(il_xl_h._dict_type.value_type, '__len__'):
            if isinstance(n_horizons, int):
                horizons_idx = [-1]
            else:
                horizons_idx = n_horizons
                n_horizons = -1
            mask = create_mask(ilines_, xlines_, hs_, il_xl_h,
                               geom.ilines_offset, geom.xlines_offset, geom.depth,
                               mode, width, horizons_idx, n_horizons)
        else:
            mask = create_mask_f(ilines_, xlines_, hs_, il_xl_h,
                                 geom.ilines_offset, geom.xlines_offset, geom.depth)

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = mask
        return self


    @action
    @inbatch_parallel(init='indices', target='threads', post='_stitch_clouds')
    def get_point_cloud(self, ix, src_masks='masks', src_slices='slices', dst='predicted_labels',
                        threshold=0.5, averaging='mean', coordinates='cubic', order=(2, 0, 1),
                        height_margin=2, border_margin=1):
        """ Convert labels from horizons-mask into point-cloud format. Fetches point-clouds from
        a batch of masks, then merges resulting clouds to those stored in `dst`, whenever possible.

        Parameters
        ----------
        src_masks : str
            component of batch that stores masks.
        src_slices : str
            component of batch that stores slices of crops.
        dst : str/object
            component of batch to store the resulting labels, o/w a storing object.
        threshold : float
            parameter of mask-thresholding.
        averaging : str
            method of pandas.groupby used for finding the center of a horizon.
        coordinates : str
            coordinates-mode to use for keys of point-cloud. Can be either 'cubic'
            or 'lines'. In case of `lines`-option, `geometries` must be loaded as
            a component of batch.
        order : tuple of int
            axes-param for `transpose`-operation, applied to a mask before fetching point clouds.
            Default value of (2, 0, 1) is applicable to standart pipeline with one `rotate_axes`
            applied to images-tensor.
        height_margin : int
            if adjacent horizons do not diverge for more than this distance, they can be merged together.
        border_margin : int
            max distance between a pair of horizon-borders when the horizons can be adjacent.

        Returns
        -------
        SeismicCropBatch
            batch with fetched labels.
        """
        _, _, _ = dst, height_margin, border_margin

        # threshold the mask, reshape and rotate the mask if needed
        mask = (getattr(self, src_masks)[self.get_pos(None, src_masks, ix)] > threshold).astype(np.float32)
        mask = np.reshape(mask, mask.shape[:3])
        mask = np.transpose(mask, axes=order)

        # prepare args
        if isinstance(dst, str):
            dst = getattr(self, dst)

        i_shift, x_shift, h_shift = [self.get(ix, src_slices)[k][0] for k in range(3)]
        geom = self.get(ix, 'geometries')
        if coordinates == 'lines':
            transforms = (lambda i_: geom.ilines[i_ + i_shift], lambda x_: geom.xlines[x_ + x_shift],
                          lambda h_: h_ + h_shift)
        else:
            transforms = (lambda i_: i_ + i_shift, lambda x_: x_ + x_shift,
                          lambda h_: h_ + h_shift)

        # get horizons and merge them with matching aggregated ones
        horizons = mask_to_horizon(mask, threshold, averaging, transforms, separate=True)
        return horizons


    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def filter_out(self, ix, src=None, dst=None, mode=None, expr=None, low=None, high=None, length=None):
        """ Cut mask for horizont extension task.

        Parameters
        ----------
        src : str
            Component of batch with mask
        dst : str
            Component of batch to put cut mask in.
        mode : str
            Either point, line, iline or xline.
            If point, then only only one point per horizon will be labeled.
            If iline or xline then single iline or xline with labeled.
            If line then randomly either single iline or xline will be
            labeled.
        expr : callable, optional.
            Some vectorized function. Accepts points in cube, returns either float.
            If not None, low or high/length should also be supplied.
        """
        if not (src and dst):
            raise ValueError('Src and dst must be provided')

        pos = self.get_pos(None, src, ix)
        mask = getattr(self, src)[pos]
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            getattr(self, dst)[pos] = mask
            return self
        if mode is not None:
            new_mask = np.zeros_like(mask)
            point = np.random.randint(len(coords))
            if mode == 'point':
                new_mask[coords[0][point], coords[1][point], :] = mask[coords[0][point], coords[1][point], :]
            elif mode == 'iline' or (mode == 'line' and np.random.binomial(1, 0.5)) == 1:
                new_mask[coords[0][point], :, :] = mask[coords[0][point], :, :]
            elif mode in ['xline', 'line']:
                new_mask[:, coords[1][point], :] = mask[:, coords[1][point], :]
            else:
                raise ValueError('Mode should be either `point`, `iline`, `xline` or `line')
            mask = new_mask
        if expr is not None:
            coords = np.where(mask > 0)
            new_mask = np.zeros_like(mask)

            coords = np.array(coords).astype(np.float).T
            cond = np.ones(shape=coords.shape[0]).astype(bool)
            coords /= np.reshape(mask.shape, newshape=(1, 3))
            if low is not None:
                cond &= np.greater_equal(expr(coords), low)
            if high is not None:
                cond &= np.less_equal(expr(coords), high)
            if length is not None:
                low = 0 if not low else low
                cond &= np.less_equal(expr(coords), low + length)
            coords *= np.reshape(mask.shape, newshape=(1, 3))
            coords = np.round(coords).astype(np.int32)[cond]
            new_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = mask[coords[:, 0], coords[:, 1], coords[:, 2]]
            mask = new_mask

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = mask
        return self


    @action
    @inbatch_parallel(init='indices', target='threads')
    def scale(self, ix, mode, src=None, dst=None):
        """ Scale values in crop. """
        pos = self.get_pos(None, src, ix)
        comp_data = getattr(self, src)[pos]
        geom = self.get(ix, 'geometries')

        if mode == 'normalize':
            new_data = geom.scaler(comp_data)
        elif mode == 'denormalize':
            new_data = geom.descaler(comp_data)
        else:
            raise ValueError('Scaling mode is not recognized.')

        dst = dst or src
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self)))

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = new_data
        return self


    @action
    @inbatch_parallel(init='_init_component', post='_assemble', target='threads')
    def concat_components(self, ix, src, dst, axis=-1):
        """ Concatenate a list of components and save results to `dst` component.

        Parameters
        ----------
        src : array-like
            List of components to concatenate of length more than one.
        dst : str
            Component of batch to put results in.
        axis : int
            The axis along which the arrays will be joined.
        """
        _ = dst
        if not isinstance(src, (list, tuple, np.ndarray)) or len(src) < 2:
            raise ValueError('Src must contain at least two components to concatenate')
        result = []
        for component in src:
            pos = self.get_pos(None, component, ix)
            result.append(getattr(self, component)[pos])
        return np.concatenate(result, axis=axis)


    @action
    @inbatch_parallel(init='run_once')
    def assemble_crops(self, src, dst, grid_info, order=None):
        """ Glue crops together in accordance to the grid.

        Note
        ----
        In order to use this action you must first call `make_grid` method of SeismicCubeset.

        Parameters
        ----------
        src : array-like
            Sequence of crops.
        dst : str
            Component of batch to put results in.
        grid_info : dict
            Dictionary with information about grid. Should be created by `make_grid` method.

        Returns
        -------
        SeismicCropBatch
            Batch with assembled subcube in desired component.
        """
        # Do nothing until there is a crop for every point
        if len(src) != len(grid_info['grid_array']):
            return self

        order = order or (2, 0, 1)
        # Since we know that cube is 3-d entity, we can get rid of
        # unneccessary dimensions
        src = np.array(src)
        src = src if len(src.shape) == 4 else np.squeeze(src, axis=-1)
        assembled = aggregate(src, grid_info['grid_array'], grid_info['crop_shape'],
                              grid_info['predict_shape'], order)
        setattr(self, dst, assembled)
        return self


    def _side_view_reshape_(self, crop, shape):
        """ Changes axis of view to match desired shape.
        Must be used in combination with `side_view` argument of `crop` action.

        Parameters
        ----------
        shape : sequence
            Desired shape of resulting crops.
        """
        if (np.array(crop.shape) != np.array(shape)).any():
            return crop.transpose([1, 0, 2])
        return crop


    def _rotate_axes_(self, crop):
        """ The last shall be first and the first last.

        Notes
        -----
        Actions `crop`, `load_cubes`, `create_mask` make data in [iline, xline, height]
        format. Since most of the models percieve ilines as channels, it might be convinient
        to change format to [xline, heigh, ilines] via this action.
        """
        crop_ = np.swapaxes(crop, 0, 1)
        crop_ = np.swapaxes(crop_, 1, 2)
        return crop_

    def _add_axis_(self, crop):
        """ Add new axis.

        Notes
        -----
        Used in combination with `dice` and `ce` losses to tell model that input is
        3D entity, but 2D convolutions are used.
        """
        return crop[..., np.newaxis]

    def _additive_noise_(self, crop, scale):
        """ Add random value to each entry of crop. Added values are centered at 0.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution."""
        return crop + np.random.normal(loc=0, scale=scale, size=crop.shape)

    def _multiplicative_noise_(self, crop, scale):
        """ Multiply each entry of crop by random value, centered at 1.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution."""
        return crop * np.random.normal(loc=1, scale=scale, size=crop.shape)

    def _cutout_2d_(self, crop, patch_shape, n):
        """ Change patches of data to zeros.

        Parameters
        ----------
        patch_shape : array-like
            Shape or patches along each axis.
        n : float
            Number of patches to cut.
        """
        rnd = np.random.RandomState(int(n*100)).uniform
        patch_shape = patch_shape.astype(int)

        copy_ = copy(crop)
        for _ in range(int(n)):
            x_ = int(rnd(max(crop.shape[0] - patch_shape[0], 1)))
            h_ = int(rnd(max(crop.shape[1] - patch_shape[1], 1)))
            copy_[x_:x_+patch_shape[0], h_:h_+patch_shape[1], :] = 0
        return copy_

    def _rotate_(self, crop, angle):
        """ Rotate crop along the first two axes.

        Parameters
        ----------
        angle : float
            Angle of rotation.
        """
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), angle, 1)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

    def _flip_(self, crop, axis=0):
        """ Flip crop along the given axis.

        Parameters
        ----------
        axis : int
            Axis to flip along
        """
        return cv2.flip(crop, axis).reshape(crop.shape)

    def _scale_2d_(self, crop, scale):
        """ Zoom in or zoom out along the first two axes of crop.

        Parameters
        ----------
        scale : float
            Zooming factor.
        """
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), 0, scale)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

    def _affine_transform_(self, crop, alpha_affine=10):
        """ Perspective transform. Moves three points to other locations.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_affine : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_affine*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_affine >= min(shape)//16:
            alpha_affine = min(shape)//16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size]])

        pts2 = pts1 + rnd(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)


        matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    def _perspective_transform_(self, crop, alpha_persp):
        """ Perspective transform. Moves four points to other four.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_persp : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_persp*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_persp >= min(shape) // 16:
            alpha_persp = min(shape) // 16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size],
                           [center_[0] - square_size, center_[1] + square_size]])

        pts2 = pts1 + rnd(-alpha_persp, alpha_persp, size=pts1.shape).astype(np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    def _elastic_transform_(self, crop, alpha=40, sigma=4):
        """ Transform indexing grid of the first two axes.

        Parameters
        ----------
        alpha : float
            Maximum shift along each axis.
        sigma : float
            Smoothening factor.
        """
        state = np.random.RandomState(int(alpha*100))
        shape_size = crop.shape[:2]

        grid_scale = 4
        alpha //= grid_scale
        sigma //= grid_scale
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur((state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv2.GaussianBlur((state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)

        distorted_img = cv2.remap(crop, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT_101,
                                  interpolation=cv2.INTER_LINEAR)
        return distorted_img.reshape(crop.shape)

    def _bandwidth_filter_(self, crop, lowcut=None, highcut=None, fs=1, order=3):
        """ Keep only frequences between lowcut and highcut.

        Notes
        -----
        Use it before other augmentations, especially before ones that add lots of zeros.

        Parameters
        ----------
        lowcut : float
            Lower bound for frequences kept.
        highcut : float
            Upper bound for frequences kept.
        fs : float
            Sampling rate.
        order : int
            Filtering order.
        """
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = butter(order, lowcut / nyq, btype='low')
        else:
            b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, crop, axis=1)

    def _sign_(self, crop):
        """ Element-wise indication of the sign of a number. """
        return np.sign(crop)

    def _analytic_transform_(self, crop, axis=1, mode='phase'):
        """ Compute instantaneous phase or frequency via the Hilbert transform.

        Parameters
        ----------
        axis : int
            Axis of transformation. Intended to be used after `rotate_axes`, so default value
            is to make transform along depth dimension.
        mode : str
            If 'phase', compute instantaneous phase.
            If 'freq', compute instantaneous frequency.
        """
        analytic = hilbert(crop, axis=axis)
        phase = np.unwrap(np.angle(analytic))

        if mode == 'phase':
            return phase
        if 'freq' in mode:
            return np.diff(phase, axis=axis, prepend=0) / (2*np.pi)
        raise ValueError('Unknown `mode` parameter.')


    def plot_components(self, *components, idx=0, plot_mode='overlap', order_axes=None, **kwargs):
        """ Plot components of batch.

        Parameters
        ----------
        idx : int or None
            If int, then index of desired image in list.
            If None, then no indexing is applied.
        components : str or sequence of str
            Components to get from batch and draw.
        plot_mode : bool
            If 'overlap', then images are drawn one over the other.
            If 'facies', then images are drawn one over the other with transparency.
            If 'separate', then images are drawn on separate layouts.
        order_axes : sequence of int
            Determines desired order of the axis. The first two are plotted.
        cmaps : str or sequence of str
            Color maps for showing images.
        alphas : number or sequence of numbers
            Opacity for showing images.
        """
        plot_batch_components(self, *components, idx=idx, plot_mode=plot_mode, order_axes=order_axes, **kwargs)
