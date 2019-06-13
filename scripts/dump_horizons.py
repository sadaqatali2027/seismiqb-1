import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# path to seismiqb-repo
PATH_TO_SEISMIQB = ""
sys.path.append(PATH_TO_SEISMIQB)

from seismiqb import SeismicCropBatch, SeismicGeometry, SeismicCubeset
from seismiqb.batchflow import Dataset, Pipeline, FilesIndex, B, V, C, L, D
from seismiqb.batchflow.models.tf import TFModel

# script-args
# cube crop for prediction
CUBE_CROP = ([0, 2500], [0, 1200], [0, 1350])

# small-crop shape
CROP_SHAPE = [3, 400, 400]

# horizon-predictions should cover at least this share of inlines-xlines to be dumped
AREA_SHARE = 0.6

# threshold for clearing up predictions
THRESHOLD = 0.5

PATH_TO_MODEL = "/notebooks/SEISMIC_DATA/CUBE_1/"
PATH_TO_CUBE = "/notebooks/SEISMIC_DATA/CUBE_1/E_anon.hdf5"
PATH_TO_PREDICTIONS = "/notebooks/SEISMIC_DATA/CUBE_1/"

# set GPU-device
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# init Dataset and load cube-geometries
dsix = FilesIndex(path=PATH_TO_CUBE, no_ext=True)
ds = SeismicCubeset(dsix)
ds = ds.load_geometries()

# make grid for small crops
ds = ds.make_grid(ds.indices[0], CROP_SHAPE, *CUBE_CROP)

# predict pipeline
predict_pipeline = (Pipeline()
                    .load_component(src=D('geometries'), dst='geometries')
                    .crop(points=L(D('grid_gen')), shape=CROP_SHAPE)
                    .load_cubes(dst='data_crops')
                    .rotate_axes(src='data_crops')
                    .scale(mode='normalize', src='data_crops')
                    .load_model({'load': PATH_TO_MODEL})
                    .init_variable('result_preds', init_on_each_run=list())
                    .predict_model('ED', fetches='sigmoid', cubes = B('data_crops'),
                                   save_to=V('result_preds'), mode='e')
                    .assemble_crops(src=V('result_preds'), dst='assembled_pred',
                                    grid_info=D('grid_info'))
                    ) << ds

# run the pipeline
for _ in tqdm(range(ds.grid_iters)):
    batch = predict_pipeline.next_batch(1, n_epochs=None)

# fetch and dump horizons
def _dump_horizon(horizon, geometry, path, name):
    ixhs = []
    for k, v in horizon.items():
        ixhs.append([k[0], k[1], v])
    labels = pd.DataFrame(ixhs, columns=['inline', 'xline', 'height'])
    labels.sort_values(['inline', 'xline'], inplace=True)
    sample_rate, delay = geometry.sample_rate, geometry.delay
    inverse = lambda h: (h + 1) * sample_rate + delay
    labels.loc[:, 'height'] = inverse(labels.loc[:, 'height'])
    labels.to_csv(os.path.join(path, name + '.csv'), sep=' ', index=False)

prediction = batch.assembled_pred
ds.get_point_cloud(prediction, 'horizons', coordinates='lines', threshold=THRESHOLD)
ctr = 0
for h in ds.horizons:
    if len(h) / ds.geometries[ds.indices[0]].cube_shape >= AREA_SHARE:
        _dump_horizon(h, ds.geometries[ds.indices[0]], PATH_TO_PREDICTIONS, 'horizon_' + str(ctr))
        ctr += 1
