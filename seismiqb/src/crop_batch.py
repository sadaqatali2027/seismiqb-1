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
from .utils import create_mask, aggregate, make_labels_dict, _get_horizons
from .plot_utils import plot_batch_components


AFFIX = '___'
SIZE_POSTFIX = 7
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX


@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class SeismicCropBatch(Batch):
    """ Batch with ability to generate 3d-crops of various shapes."""
    components = ('slices', 'geometries', 'labels')

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

    def _assemble_labels(self, all_clouds, *args, dst=None, **kwargs):
        """ Assemble labels-dict from different crops in batch.
        """
        _ = args
        labels = dict()
        labels_ = dict()

        # init labels-dict
        for ix in self.indices:
            labels_[self.unsalt(ix)] = set()

        for ix, cloud in zip(self.indices, all_clouds):
            labels_[self.unsalt(ix)] |= set(cloud.keys())

        for cube, ilines_xlines in labels_.items():
            labels[cube] = dict()
            for il_xl in ilines_xlines:
                labels[cube][il_xl] = set()

        # fill labels with sets of horizons
        for ix, cloud in zip(self.indices, all_clouds):
            for il_xl, heights in cloud.items():
                labels[self.unsalt(ix)][il_xl] |= set(heights)

        # transforms sets of horizons to labels
        for cube in labels:
            for il_xl in labels[cube]:
                labels[cube][il_xl] = np.sort(list(labels[cube][il_xl]))

        # convert labels to numba.Dict if needed
        if kwargs.get('to_numba'):
            for cube, cloud_dict in labels.items():
                cloud = []
                for il_xl, horizons in cloud_dict.items():
                    (il, xl) = il_xl
                    for h in horizons.reshape(-1):
                        cloud.append([il, xl, h])

                cloud = np.array(cloud)
                labels[cube] = make_labels_dict(cloud)

        setattr(self, dst, labels)
        return self

    def get_pos(self, data, component, index):
        """ Get correct slice/key of a component-item based on its type.
        """
        candidates = self.components + ('segyfiles', )
        if component in candidates and component != 'slices':
            return self.unsalt(index)
        return super().get_pos(data, component, index)

    def __setattr__(self, name, value):
        if self.components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                if self._item_class is None:
                    self.make_item_class()
                self._data_named = self._item_class(data=self._data)   # pylint: disable=not-callable
                return
            if name in self.components:    # pylint: disable=unsupported-membership-test
                if self._data_named is None:
                    _ = self.data
                setattr(self._data_named, name, value)
                super().__setattr__('_data', self._data_named.data)
                return
        super().__setattr__(name, value)

    @action
    def add_components(self, components, init=None):
        """ Add new components

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data

        Raises
        ------
        ValueError
            If the component with the given name already exists
        """
        if isinstance(components, str):
            components = (components,)
            init = (init,)
        elif isinstance(components, (tuple, list)):
            components = tuple(components)
            if init is None:
                init = (None,) * len(components)
            else:
                init = tuple(init)

        if any(hasattr(self, c) for c in components):
            raise ValueError("An attribute with the same name exists")

        data = self._data
        if self.components is None:
            self.components = tuple()
            data = tuple()
            warnings.warn("All batch data is erased")

        exists_component = set(components) & set(self.components)
        if exists_component:
            raise ValueError("Component(s) with name(s) '{}' already exists".format("', '".join(exists_component)))

        self.components = self.components + components
        
        self.make_item_class(local=True)
        if data is not None:
            self._data = data + init
        print('in add components type of data is ', type(self._data))
        return self

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
    def crop(self, points, shape, dilations=None, dst='slices', passdown=None):
        """ Generate positions of crops. Creates new instance of `SeismicCropBatch`
        with crop positions in one of the components (`slices` by default).
        Parameters
        ----------
        points : array-like
            Upper rightmost points for every crop and name of cube to
            cut it from. Order is: name, iline, xline, height. For example,
            ['Cube.sgy', 13, 500, 200] stands for crop has [13, 500, 200]
            as its upper rightmost point and must be cut from 'Cube.sgy' file.
        shape : array-like
            Desired shape of crops.
        dst : str, optional
            Component of batch to put positions of crops in.
        passdown : str of list of str
            Components of batch to keep in the new one.
        Note
        ----
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
                if not component in new_batch.components:
                    new_batch.add_components(component)
                setattr(new_batch, component, getattr(self, component))

        dilations = dilations or [1, 1, 1]
        slices = []
        for point in points:
            slice_ = self._make_slice(point, shape, dilations)
            slices.append(slice_)
        setattr(new_batch, dst, slices)
        return new_batch


    def _make_slice(self, point, shape, dilations):
        """ Creates list of `np.arange`'s for desired location. """
        ix = point[0]

        if isinstance(point[1], float) or isinstance(point[2], float) or isinstance(point[3], float):
            geom = self.get(ix, 'geometries')
            slice_point = (point[1:] * (np.array(geom.cube_shape) - np.array(shape))).astype(int)
        else:
            slice_point = point[1:]

        slice_ = [np.arange(slice_point[0], slice_point[0]+shape[0]*dilations[0], dilations[0]),
                  np.arange(slice_point[1], slice_point[1]+shape[1]*dilations[1], dilations[1]),
                  np.arange(slice_point[2], slice_point[2]+shape[2]*dilations[2], dilations[2])]

        return slice_


    @action
    def load_cubes(self, dst, fmt='h5py', src='slices'):
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
            return self._load_cubes_sgy(src=src, dst=dst)
        if fmt.lower() in ['h5py', 'h5']:
            return self._load_cubes_h5py(src=src, dst=dst)

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
    def _load_cubes_h5py(self, ix, dst, src='slices'):
        """ Load data from .hdf5-cube in given positions. """
        geom = self.get(ix, 'geometries')
        h5py_cube = geom.h5py_file['cube']
        dtype = h5py_cube.dtype

        slice_ = self.get(ix, src)
        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]

        crop = np.zeros((len(ilines_), len(xlines_), len(hs_)), dtype=dtype)
        for i, iline_ in enumerate(ilines_):
            slide = h5py_cube[iline_, :, :]
            crop[i, :, :] = slide[xlines_, :][:, hs_]

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = crop
        return self

    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def create_masks(self, ix, dst, src='slices', mode='horizon', width=3, src_labels='labels', single_horizon=False):
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

        Returns
        -------
        SeismicCropBatch
            Batch with loaded masks in desired components.

        Notes
        -----
        Can be run only after labels-dict is loaded into labels-component.
        """
        geom = self.get(ix, 'geometries')
        il_xl_h = self.get(ix, src_labels)

        slice_ = self.get(ix, src)
        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]
        mask = create_mask(ilines_, xlines_, hs_, il_xl_h,
                           geom.ilines, geom.xlines, geom.depth, mode, width, single_horizon)

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = mask
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble_labels', target='threads')
    def get_point_cloud(self, ix, src_masks='masks', src_slices='slices', dst='predicted_labels',
                        threshold=0.5, averaging='mean', coordinates='cubic', to_numba=False):
        """ Convert labels from horizons-mask into point-cloud format.
        Parameters
        ----------
        src_masks : str
            component of batch that stores masks.
        src_slices : str
            component of batch that stores slices of crops.
        dst : str
            component of batch to store the resulting labels.
        threshold : float
            parameter of mask-thresholding.
        averaging : str
            method of pandas.groupby used for finding the center of a horizon.
        coordinates : str
            coordinates-mode to use for keys of point-cloud. Can be either 'cubic'
            or 'lines'. In case of `lines`-option, `geometries` must be loaded as
            a component of batch.
        to_numba : bool
            whether to convert the resulting point-cloud to numba-dict. The conversion
            takes additional time.
        Returns
        -------
        SeismicCropBatch
            batch with fetched labels.
        """
        _ = dst, to_numba

        # threshold the mask
        mask = getattr(self, src_masks)[self.get_pos(None, src_masks, ix)]

        # prepare args
        i_shift, x_shift, h_shift = [self.get(ix, src_slices)[k][0] for k in range(3)]
        geom = self.get(ix, 'geometries')
        if coordinates == 'lines':
            transforms = (lambda i_: geom.ilines[i_ + i_shift], lambda x_: geom.xlines[x_ + x_shift],
                          lambda h_: h_ + h_shift)
        else:
            transforms = (lambda i_: i_ + i_shift, lambda x_: x_ + x_shift,
                          lambda h_: h_ + h_shift)

        return _get_horizons(mask, threshold, averaging, transforms, separate=False)

    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def filter_out(self, ix, src=None, dst=None, mode=None, expr=None, low=None, high=None):
        """ Cut mask for horizont extension task.
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
            If not None, high or low should also be supplied.
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
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0]))


    def _flip_(self, crop, axis=0):
        """ Flip crop along the given axis.
        Parameters
        ----------
        axis : int
            Axis to flip along
        """
        return cv2.flip(crop, axis)


    def _scale_2d_(self, crop, scale):
        """ Zoom in or zoom out along the first two axes of crop.
        Parameters
        ----------
        scale : float
            Zooming factor.
        """
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), 0, scale)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0]))


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
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0]))


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
        return cv2.warpPerspective(crop, matrix, (shape[1], shape[0]))


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
        return distorted_img


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


    def plot_components(self, idx=0, *components, overlap=True, rotate_axes=0, cmaps=None, alphas=None):
        """ Plot components of batch.

        Parameters
        ----------
        idx : int or None
            If int, then index of desired image in list.
            If None, then no indexing is applied.

        overlap : bool
            Whether to draw images one over the other or not.

        components : str or sequence of str
            Components to get from batch and draw.

        cmaps : str or sequence of str
            Color maps for showing images.

        alphas : number or sequence of numbers
            Opacity for showing images.
        """
        plot_batch_components(self, idx, *components, overlap=overlap,
                              rotate_axes=rotate_axes, cmaps=cmaps, alphas=alphas)
