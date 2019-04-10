""" Contains container for storing dataset of seismic crops along with necessary utils. """

class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures.
    """
    def __init__(self, index, batch_class=Batch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes.
        """
        super().__init__(index, batch_class=Batch, preloaded=None, *args, **kwargs)
        self.geometries = dict()
        self.samplers = dict()
