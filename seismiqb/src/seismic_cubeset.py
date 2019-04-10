""" Contains container for storing dataset of seismic crops. """
import dill

from ..batchflow import Dataset, Batch
from .seismic_geometry import SeismicGeometry

from .utils import read_point_cloud, apply, make_labels_dict

class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures.
    """
    def __init__(self, index, batch_class=Batch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes.
        """
        super().__init__(index, batch_class=Batch, preloaded=None, *args, **kwargs)
        self.geometries = {path: SeismicGeometry() for path in self.indices}
        self.samplers = {path: None for path in self.indices}
        self.labels = {path: dict() for path in self.indices}
        self.point_cloud = {path: np.array() for path in self.indices}

    def load_geometries(self, path=None):
        """ Load geometries into dataset-attribute.
        """
        if isinstance(path, str):
            with open(path, 'rb') as file:
                self.geometries = dill.load(file)

        else:
            for path in self.indices:
                # not pring but logging
                # print('Creating Geometry for file: ' + path_data)
                self.geometries[path].load(path)

        return self

    def save_geometries(self, to):
        """ Save dill-serialized geometries for a dataset of seismic-cubes on disk.
        """
        if isinstance(to, str):
            with open(to, 'wb') as file:
                dill.dump(self.geometries, file)
            # not pring but logging
            # print('Geometries are saved to ' + save_to)
        return self

    def load_point_cloud(self, paths, **kwargs):
        """ Load point-cloud of labels for each cube in dataset.
        """
        for path in self.indices:
            self.point_cloud[path] = read_point_cloud(paths[path], **kwargs)

        return self

    def make_labels(self, transforms=None, src='point_cloud'):
        """ Make labels in inline-xline coordinates using cloud of points and supplied transforms.
        """
        # apply transforms to point-cloud
        point_cloud = getattr(self, src) if isinstance(src, str) else point_cloud
        transforms = dict() if transforms is None else transforms

        for path in self.indices:
            self.labels[path] = make_labels_dict(apply(point_cloud.get(path), transforms.get(path)))

        return self
