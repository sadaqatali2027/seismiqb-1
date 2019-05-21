""" Seismic Crop Batch."""
import string
import random

import numpy as np
import pandas as pd
import segyio
from skimage.measure import label, regionprops

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel

from .utils import create_mask


AFFIX = '___'
SIZE_POSTFIX = 7
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX



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

    def _assemble_labels(self, all_clouds, *args, dst=None, **kwargs):
        """ Assemble labels-dict from different crops in batch.
        """
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
        if component in ('geometries', 'labels', 'segyfiles'):
            return self.unsalt(index)
        return super().get_pos(data, component, index)


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
    def crop(self, points, shape, dst='slices', passdown=None):
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
                setattr(new_batch, component, getattr(self, component))

        slices = []
        for point in points:
            slice_ = self._make_slice(point, shape)
            slices.append(slice_)
        setattr(new_batch, dst, slices)
        return new_batch


    def _make_slice(self, point, shape):
        """ Creates list of `np.arange`'s for desired location. """
        ix = point[0]

        if isinstance(point[1], float) or isinstance(point[2], float) or isinstance(point[3], float):
            geom = self.get(ix, 'geometries')
            slice_point = (point[1:] * (np.array(geom.cube_shape) - np.array(shape))).astype(int)
        else:
            slice_point = point[1:]

        slice_ = [np.arange(slice_point[0], slice_point[0]+shape[0]),
                  np.arange(slice_point[1], slice_point[1]+shape[1]),
                  np.arange(slice_point[2], slice_point[2]+shape[2])]
        return slice_


    @action
    @inbatch_parallel(init='_sgy_init', post='_sgy_post', target='threads')
    def load_cubes(self, ix, segyfile, dst, src='slices'):
        """ Load data from cube in given positions.

        Notes
        -----
        Init function `_sgy_init` passes both index and handler to necessary
        .sgy file. Post function '_sgy_post' takes all of the handlers and
        closes I/O. That is done in order to open every file only once (since it is time-consuming).

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.

        dst : str
            Component of batch to put loaded crops in.

        Returns
        -------
        SeismicCropBatch
            Batch with loaded crops in desired component.
        """
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


    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def create_masks(self, ix, dst, src='slices', mode='horizon', width=3):
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
        il_xl_h = self.get(ix, 'labels')

        slice_ = self.get(ix, src)
        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]
        mask = create_mask(ilines_, xlines_, hs_, il_xl_h, geom.ilines, geom.xlines, geom.depth, mode, width)

        pos = self.get_pos(None, dst, ix)
        getattr(self, dst)[pos] = mask
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble_labels', target='threads')
    def get_point_cloud(self, ix, src_masks='masks', src_slices='slices', dst='predicted_labels',
                        threshold=0.5, averaging='mean', coords='cubic', to_numba=False):
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
        coords : str
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
        # threshold the mask
        mask = getattr(self, src_masks)[self.get_pos(None, src_masks, ix)]
        mask_ = np.zeros_like(mask, np.int32)
        mask_[mask < threshold] = 0
        mask_[mask >= threshold] = 1

        # get regions
        labels = label(mask_)
        regions = regionprops(labels)

        # get horizons
        horizons = dict()
        for region in regions:
            coords = region.coords
            coords = pd.DataFrame(coords, columns=['iline', 'xline', 'height'])
            horizon_ = getattr(coords.groupby(['iline', 'xline']), averaging)()
            for i_x in horizon_.index.values:
                i, x = i_x
                i_shift, x_shift, h_shift = [self.get(ix, src_slices)[k][0] for k in range(3)]
                il_xl = (i + i_shift, x + x_shift)
                if coords == 'lines':
                    geom = self.get(ix, 'geometries')
                    il_xl = geom.ilines[il_xl[0]], geom.xlines[il_xl[1]]

                if il_xl in horizons:
                    horizons[il_xl].append(horizon_.loc[i_x, 'height'] + h_shift)
                else:
                    horizons[il_xl] = [horizon_.loc[i_x, 'height'] + h_shift]

        return horizons

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
