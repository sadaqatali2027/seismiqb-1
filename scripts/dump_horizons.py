"""Test"""
#pylint: disable=import-error, wrong-import-position
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from seismiqb import SeismicCubeset
from seismiqb.batchflow import Pipeline, FilesIndex, B, V, L, D
from seismiqb.batchflow.models.tf import TFModel

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Script-args
# Cube crop for prediction
CUBE_CROP = ([0, 2500], [0, 1200], [0, 1350])

# Small-crop shape
CROP_SHAPE = [3, 400, 400]

# Stride of crops
CROP_STRIDE = [3, 400, 400]

# Threshold of covered area by horizon to be dumped
AREA_SHARE = 0.6

# Threshold for clearing predictions
THRESHOLD = 0.5

PATH_TO_MODEL = "/notebooks/SEISMIC_DATA/SAVED/MODELS/CUBE_1_ONLY/"
PATH_TO_CUBE = "/notebooks/SEISMIC_DATA/CUBE_1/E_anon.hdf5"
PATH_TO_PREDICTIONS = "/notebooks/SEISMIC_DATA/CUBE_1/PREDICTED_SCRIPT/"


def dump_horizon(horizon, geometry, path, name):
    """ Convert horizon to point cloud and save it to desired place. """
    ixhs = []
    for k, v in horizon.items():
        ixhs.append([k[0], k[1], v])
    labels = pd.DataFrame(ixhs, columns=['inline', 'xline', 'height'])
    labels.sort_values(['inline', 'xline'], inplace=True)
    sample_rate, delay = geometry.sample_rate, geometry.delay
    inverse = lambda h: (h + 1) * sample_rate + delay
    labels.loc[:, 'height'] = inverse(labels.loc[:, 'height'])
    labels.to_csv(os.path.join(path, name + '.csv'), sep=' ', index=False)


def main():
    """ Main function. """
    # Init Cubeset and load cube-geometries
    dsix = FilesIndex(path=PATH_TO_CUBE, no_ext=True)
    ds = SeismicCubeset(dsix)
    ds = ds.load_geometries()

    # Make grid for small crops
    ds = ds.make_grid(ds.indices[0], CROP_SHAPE, *CUBE_CROP)

    # Pipeline: slice crops, normalize values in the cube, make predictions
    # via model, assemble crops according to the grid
    load_config = {'load/path': PATH_TO_MODEL}
    predict_pipeline = (Pipeline()
                        .load_component(src=D('geometries'), dst='geometries')
                        .crop(points=L(D('grid_gen')), shape=CROP_SHAPE)
                        .load_cubes(dst='data_crops')
                        .rotate_axes(src='data_crops')
                        .scale(mode='normalize', src='data_crops')
                        .init_model('dynamic', TFModel, 'loaded_model', load_config)
                        .init_variable('result_preds', init_on_each_run=list())
                        .predict_model('loaded_model', fetches='sigmoid', cubes=B('data_crops'),
                                       save_to=V('result_preds'), mode='e')
                        .assemble_crops(src=V('result_preds'), dst='assembled_pred',
                                        grid_info=D('grid_info'))
                        ) << ds

    for _ in tqdm(range(ds.grid_iters)):
        batch = predict_pipeline.next_batch(1, n_epochs=None)

    # fetch and dump horizons
    prediction = batch.assembled_pred
    ds.get_point_cloud(prediction, 'horizons', coordinates='lines', threshold=THRESHOLD)

    if (not os.path.exists(PATH_TO_PREDICTIONS)) and \
       (not os.path.isdir(PATH_TO_PREDICTIONS)):
        os.mkdir(PATH_TO_PREDICTIONS)

    ctr = 0
    for h in ds.horizons:
        if len(h) / np.prod(ds.geometries[ds.indices[0]].cube_shape) >= AREA_SHARE:
            dump_horizon(h, ds.geometries[ds.indices[0]],
                         PATH_TO_PREDICTIONS, 'horizon_' + str(ctr))
            ctr += 1



if __name__ == '__main__':
    main()
