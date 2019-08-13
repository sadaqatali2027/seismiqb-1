# Necessary imports
import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('..')
from seismiqb.batchflow import Dataset, Pipeline, FilesIndex
from seismiqb.batchflow import B, V, C, L, F, D, P, R
from seismiqb.batchflow.models.tf import DenseNetFC, TFModel
from seismiqb import SeismicCropBatch, SeismicGeometry, SeismicCubeset, plot_loss
# from models.extension_utils_images.py import make_data_extension
# from models.utils.py import predictions

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CROPS = 64
CROP_SHAPE = [2, 64, 64]
MODEL_SHAPE = [CROP_SHAPE[i] for i in(1, 2, 0)]
MODEL_SHAPE_DICE = tuple(MODEL_SHAPE + [1])
INPUT_SHAPE_EXT = MODEL_SHAPE[:2] + [MODEL_SHAPE[-1] * 2]

path_data_0 = '/notebooks/SEISMIC_DATA/CUBE_1/E_anon.hdf5'
path_data_1 = '/notebooks/SEISMIC_DATA/CUBE_3/P_cube.hdf5'
path_data_2 = '/notebooks/SEISMIC_DATA/CUBE_VUONGMK/Repaired_cube.hdf5'

PROPORTIONS = [[0.2, 0.8], [0.4, 0.6], [0.8, 0.2]]



def create_train_ppl():
    
    def make_data_extension(batch, **kwargs):
        data_x = []
        for i, cube in enumerate(batch.images):
            cut_mask_ = batch.cut_masks[i]
            data_x.append(np.concatenate([cube, cut_mask_], axis=-1))

        data_y = []

        for cube in batch.masks:
            data_y.append(cube)
        return {"feed_dict": {'images': data_x,
                              'masks': data_y}}
    def predictions(x):
        return tf.expand_dims(x, axis=-1, name='expand')

    model_config_common = {'initial_block/inputs': 'images',
                      'body': {'num_layers': [4]*5,
                               'block/growth_rate': 8, 'block/skip': True},
                      'loss': 'dice',
                      'optimizer': 'Adam',
                      'predictions': predictions,
                      'output': 'sigmoid',
                      'common': {'data_format': 'channels_last'}}

    model_config_ext = {'inputs/images/shape': INPUT_SHAPE_EXT,
                        'inputs/masks/shape': MODEL_SHAPE_DICE, **model_config_common}

    train_ppl = (Pipeline()
                     .load_component(src=[D('geometries'), D('labels')],
                                     dst=['geometries', 'labels'])
                     .crop(points=L(D('truncated_sampler'), NUM_CROPS), shape=CROP_SHAPE)
                     .load_cubes(dst='images')
                     .create_masks(dst='masks', width=1, single_horizon=True)
                     .rotate_axes(src=['images', 'masks'])
                     .scale(mode='normalize', src='images')
                     .filter_out(src='masks', dst='cut_masks',
                                 expr=lambda m: m[:, 0],
                                 low=R('uniform', low=0.0,high=0.3),
                                 high=R('uniform', low=0.40, high=0.5))
                     .filter_out(src='cut_masks', dst='cut_masks',
                                 expr=lambda m: np.sin(50 * m[:, 0]),
                                 low=0.0)
                     .additive_noise(scale=0.005, src='images', p=0.2)
                     .rotate(angle=P(R('uniform', -30, 30)),
                             src=['images', 'masks', 'cut_masks'], p=0.4)
                     .scale_2d(scale=P(R('uniform', 0.7, 1.3)),
                               src=['images', 'masks', 'cut_masks'], p=0.4)
                     .cutout_2d(patch_shape=P(R('uniform', 5, 15, size=2)),
                                n=P(R('uniform', 3, 7)), src='images')
                     .elastic_transform(alpha=P(R('uniform', 35, 45)),
                                        sigma=P(R('uniform', 4, 4.5)),
                                        src=['images', 'masks', 'cut_masks'], p=0.2)
                     .add_axis(src='masks', dst='masks')
                     .init_variables(dict(loss_history=dict(init_on_each_run=list),
                                     current_loss=dict(init_on_each_run=0)))
                     .init_model('dynamic', DenseNetFC, 'extension', model_config_ext)
                     .train_model('extension',
                                  fetches='loss',
                                  make_data=make_data_extension,
                                  save_to=V('current_loss'),
                                  use_lock=True)
                     .update_variable('loss_history', V('current_loss'), mode='a'))
    return train_ppl

def train_model(paths, epochs=2):
    for i in range(len(paths)):
        cube_name = paths[i].split('/')[-1].split('.')[0]
        _ex_paths = paths[:i] + paths[i+1:]
        _ds_idx = FilesIndex(path=_ex_paths, no_ext=True)
        _ex_ds = SeismicCubeset(_ds_idx).load(p=PROPORTIONS[i])
        _ex_ds.modify_sampler('truncated_sampler', low=0.0, high=1.0, finish=True)
        train_pipeline = create_train_ppl()
        train_pipeline = train_pipeline << _ex_ds
        print('Start training of cube %s...' % cube_name)
        for e in range(1, epochs+1):
            train_batch = train_pipeline.next_batch(2, n_epochs=None)
        (train_pipeline.save_model('extension', path='./test_' + cube_name + '/')
                      .next_batch(2, n_epochs=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model on different cubes.")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()
    paths_list = [path_data_0, path_data_1, path_data_2]
    train_model(paths_list, epochs=args.epochs)
