""" Predict horizons-mask using saved model and dump largest horizons on disk. """
#pylint: disable=import-error, wrong-import-position
import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from seismiqb import SeismicCubeset
from seismiqb.batchflow import Pipeline, FilesIndex, B, V, L, D
from seismiqb.batchflow.models.tf import TFModel

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
    labels.to_csv(os.path.join(path, name + '.csv'), sep=' ', index=False, header=False)


def main(path_to_cube, path_to_model, path_to_predictions, gpu_devices,
         cube_crop, crop_shape, crop_stride, area_share, threshold):
    """ Main function. """
    # set gpu-devices
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

    # Init Cubeset and load cube-geometries
    dsix = FilesIndex(path=path_to_cube, no_ext=True)
    ds = SeismicCubeset(dsix)
    ds = ds.load_geometries()

    # Make grid for small crops
    ds = ds.make_grid(ds.indices[0], crop_shape, *cube_crop, crop_stride)

    # Pipeline: slice crops, normalize values in the cube, make predictions
    # via model, assemble crops according to the grid
    load_config = {'load/path': path_to_model}
    predict_pipeline = (Pipeline()
                        .load_component(src=D('geometries'), dst='geometries')
                        .crop(points=L(D('grid_gen')), shape=crop_shape)
                        .load_cubes(dst='data_crops')
                        .rotate_axes(src='data_crops')
                        .scale(mode='normalize', src='data_crops')
                        .init_model('dynamic', TFModel, 'loaded_model', load_config)
                        .init_variable('result_preds', init_on_each_run=list())
                        .predict_model('loaded_model', fetches='sigmoid', cubes=B('data_crops'),
                                       save_to=V('result_preds', mode='e'))
                        .assemble_crops(src=V('result_preds'), dst='assembled_pred',
                                        grid_info=D('grid_info'))
                        ) << ds

    for _ in tqdm(range(ds.grid_iters)):
        batch = predict_pipeline.next_batch(1, n_epochs=None)

    # Fetch and dump horizons
    prediction = batch.assembled_pred
    ds.get_point_cloud(prediction, 'horizons', coordinates='lines', threshold=threshold)

    if (not os.path.exists(path_to_predictions)) and \
       (not os.path.isdir(path_to_predictions)):
        os.mkdir(path_to_predictions)

    ctr = 0
    for h in ds.horizons:
        if len(h) / np.prod(ds.geometries[ds.indices[0]].cube_shape) >= area_share:
            dump_horizon(h, ds.geometries[ds.indices[0]],
                         path_to_predictions, 'horizon_' + str(ctr))
            ctr += 1


if __name__ == '__main__':
    # fetch path to config
    parser = argparse.ArgumentParser(description="Predict horizons on a part of seismic-cube.")
    parser.add_argument("--config_path", type=str, default="./config.json")
    args = parser.parse_args()

    # fetch main-arguments from config and run main
    with open(args.config_path, 'r') as file:
        config = json.load(file)
        args = [config.get(key) for key in ["cubePath", "modelPath", "predictionsPath", "gpuDevices",
                                            "cubeCrop", "cropShape", "cropStride", "areaShare", "threshold"]]
        main(*args)
