""" Seismic Crop Batch."""
import string
import random

import numpy as np
import segyio

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel


AFFIX = '___'
SIZE_POSTFIX = 7
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX


class SeismicCropBatch(Batch):
    """ Batch with ability to generate 3d-crops of various shapes."""
    # pylint: disable=protected-access, C0103
    components = ('slices', )

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
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
        with crop positions in one of the components.

        Note
        ----
        dsa

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

        Returns
        -------
        SeismicCropBatch
            Batch with positions of crops in specified component.
        """
        new_index = [self.salt(ix) for ix in points[:, 0]]
        new_dict = {ix: self.index.get_fullpath(self.unsalt(ix))
                    for ix in new_index}
        new_batch = SeismicCropBatch(FilesIndex.from_index(index=new_index, paths=new_dict, dirs=False))

        passdown = passdown or []
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
        path_data = point[0]
        geom = self.geometries[path_data]
        lens = [geom.ilines_len, geom.xlines_len, geom.depth]

        slice_point = (point[1:] * (np.array(lens) - np.array(shape))).astype(int)
        slice_ = [np.arange(slice_point[0], slice_point[0]+shape[0]),
                  np.arange(slice_point[1], slice_point[1]+shape[1]),
                  np.arange(slice_point[2], slice_point[2]+shape[2])]
        return slice_


    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def load_cubes(self, ix, dst, src='slices'):
        """ Load data from cube in given positions.

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.

        dst : str
            Component of batch to put loaded crops in.

        Returns
        -------
        batch : SeismicCropBatch
            Batch with loaded crops in desired component.
        """
        pos = self.get_pos(None, 'indices', ix)
        path_data = self.index.get_fullpath(ix)

        geom = self.geometries[self.unsalt(ix)]
        slice_ = getattr(self, src)[pos]

        with segyio.open(path_data, 'r', strict=False) as segyfile:
            segyfile.mmap()
            ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]

            crop = np.zeros((len(ilines_), len(xlines_), len(hs_)))
            for i, iline_ in enumerate(ilines_):
                for j, xline_ in enumerate(xlines_):
                    il_, xl_ = geom.ilines[iline_], geom.xlines[xline_]
                    tr_ = geom.il_xl_trace[(il_, xl_)]

                    crop[i, j, :] = segyfile.trace[tr_][hs_]
        getattr(self, dst)[pos] = crop
        return self


    @action
    @inbatch_parallel(init='_init_component', target='threads')
    def load_masks(self, ix, dst, src='slices'):
        """ Load masks from dictionary in given positions.

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.

        dst : str
            Component of batch to put loaded crops in.

        dst_masks : str, optional
            Component of batch to put additional crops in (e.g. masks).

        Returns
        -------
        batch : SeismicCropBatch
            Batch with loaded masks in desired components.
        """
        pos = self.get_pos(None, 'indices', ix)
        ix = self.unsalt(ix)

        geom = self.geometries[ix]
        il_xl_h = self.labels[ix]
        slice_ = getattr(self, src)[pos]

        ilines_, xlines_, hs_ = slice_[0], slice_[1], slice_[2]

        mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))
        for i, iline_ in enumerate(ilines_):
            for j, xline_ in enumerate(xlines_):
                il_, xl_ = geom.ilines[iline_], geom.xlines[xline_]

                m_temp = np.zeros(geom.depth)
                if il_xl_h.get((il_, xl_)) is not None:
                    for height_ in il_xl_h[(il_, xl_)]:
                        try:
                            m_temp[height_-3:height_+3] += 1
                        except IndexError:
                            pass
                mask[i, j, :] = m_temp[hs_]

        getattr(self, dst)[pos] = mask
        return self


    @action
    @inbatch_parallel(init='indices', target='threads')
    def scale(self, path_data, mode, src=None, dst=None):
        """ Scale values in crop. """
        pos = self.get_pos(None, 'indices', path_data)
        path_data = self.unsalt(path_data)
        comp_data = getattr(self, src)[pos]
        geom = self.geometries[path_data]

        if mode == 'normalize':
            new_data = geom.scaler(comp_data)
        elif mode == 'denormalize':
            new_data = geom.descaler(comp_data)
        else:
            raise ValueError('Scaling mode is not recognized.')

        dst = dst or src
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        getattr(self, dst)[pos] = new_data
        return self


    @staticmethod
    def salt(path):
        """ Adds random postfix of predefined length to string.

        Note
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
        """ Removes postfix that was made by `salt` method. """
        if AFFIX in path:
            return path[:-SIZE_SALT]
        return path
