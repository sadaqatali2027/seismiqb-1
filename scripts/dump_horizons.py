""" Predict horizons-mask using saved model and dump largest horizons on disk. """
#pylint: disable=import-error, wrong-import-position
import os
import sys
import argparse
import json
import logging

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


def main(path_to_cube, path_to_model, path_to_predictions, gpu_device,
         cube_crop, crop_shape, crop_stride, area_share, threshold, printer=None):
    """ Main function. """
    # Set gpu-device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # Init Cubeset and load cube-geometries
    dsix = FilesIndex(path=path_to_cube, no_ext=True)
    ds = SeismicCubeset(dsix)
    ds = ds.load_geometries()
    printer('Cube assembling is started')

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
    printer('Cube is assembled')

    # Fetch and dump horizons
    prediction = batch.assembled_pred
    ds.get_point_cloud(prediction, 'horizons', coordinates='lines', threshold=threshold)
    printer('Horizonts are labeled')

    if (not os.path.exists(path_to_predictions)) and \
       (not os.path.isdir(path_to_predictions)):
        os.mkdir(path_to_predictions)

    ds.horizons.sort(key=len, reverse=True)
    area = (cube_crop[0][1] - cube_crop[0][0]) * (cube_crop[1][1] - cube_crop[1][0])

    ctr = 0
    for h in ds.horizons[:10]:
        if len(h) / area >= area_share:
            dump_horizon(h, ds.geometries[ds.indices[0]],
                         path_to_predictions, 'Horizon_' + str(ctr))
            printer('Horizont {} is saved'.format(ctr))
            ctr += 1


if __name__ == '__main__':
    # Fetch path to config
    parser = argparse.ArgumentParser(description="Predict horizons on a part of seismic-cube.")
    parser.add_argument("--config_path", type=str, default="./configs/dump.json")
    args = parser.parse_args()

    # Fetch main-arguments from config
    with open(args.config_path, 'r') as file:
        config = json.load(file)
        args = [config.get(key) for key in ["cubePath", "modelPath", "predictionsPath", "gpuDevice",
                                            "cubeCrop", "cropShape", "cropStride", "areaShare", "threshold"]]

    # Logging to either stdout or file
    if config.get("print"):
        printer = print
    else:
        path_log = config.get("path_log") or os.path.join(os.getcwd(), "logs/dump.log")
        print('LOGGING TO ', path_log)
        handler = logging.FileHandler(path_log, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s     %(message)s'))

        logger = logging.getLogger('train_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        printer = logger.info

    main(*args, printer=printer)
