"""!!. """
import os
import logging
import random
from glob import glob
from copy import copy

import torch
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm

from ..batchflow import Pipeline, FilesIndex
from ..batchflow import B, V, C, F, D, P, R, W
from ..batchflow.models.torch import EncoderDecoder, ResBlock


from .cubeset import SeismicCubeset, Horizon
from .metrics import HorizonMetrics
from .plot_utils import plot_loss, plot_image


CUBE_PATHS = [
    '/data/seismic/CUBE_1/E_anon.hdf5',
    '/data/seismic/CUBE_2/M_cube.hdf5',
    '/data/seismic/CUBE_3/P_cube.hdf5',
    '/data/seismic/CUBE_4/R_cube.hdf5',
    '/data/seismic/CUBE_5/AMP.hdf5',
    '/data/seismic/CUBE_6/T_cube.hdf5',
    '/data/seismic/CUBE_7/S_cube.hdf5',

    '/data/seismic/CUBE_10/10_cube.hdf5',
    '/data/seismic/CUBE_11/Aya_3D_fixed.hdf5',
    '/data/seismic/CUBE_12/A_cube.hdf5',

    '/data/seismic/CUBE_15/15_cube.hdf5',
]

HORIZON_PATHS = {
    'E_anon': '/data/seismic/CUBE_1/RAW/*',
    'M_cube': '/data/seismic/CUBE_2/RAW/*',
    'P_cube': '/data/seismic/CUBE_3/RAW/prb*',
    'R_cube': '/data/seismic/CUBE_4/BEST_HORIZONS/*',
    'AMP': '/data/seismic/CUBE_5/RAW/B*',

    'S_cube': '/data/seismic/CUBE_7/RAW/*',

    'A_cube': '/data/seismic/CUBE_12/FULL_CONVERTED/*',

    '15_cube': '/data/seismic/CUBE_15/RAW/*',

    'T_cube': '',  # cube 6
    '10_cube': '', # cube 10
    '14_cube': '', # cube 14
}
HORIZON_PATHS = {key: glob(value) for key, value in HORIZON_PATHS.items()}


def dice_loss(pred, target, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        target: a tensor of shape [B, 1, H, W].
        pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = pred.shape[1]
    target = target.long()
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(pred)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(pred, dim=1)

    true_1_hot = true_1_hot.to(pred.device).type(pred.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - loss



MODEL_CONFIG = {
    # Model layout
    'initial_block': {
        'base_block': ResBlock,
        'filters': 16,
        'kernel_size': 5,
        'downsample': False,
        'attention': 'scse'
    },

    'body/encoder': {
        'num_stages': 4,
        'order': 'sbd',
        'blocks': {
            'base': ResBlock,
            'n_reps': 1,
            'filters': [32, 64, 128, 256],
            'attention': 'scse',
        },
    },
    'body/embedding': {
        'base': ResBlock,
        'n_reps': 1,
        'filters': 256,
        'attention': 'scse',
    },
    'body/decoder': {
        'num_stages': 4,
        'upsample': {
            'layout': 'tna',
            'kernel_size': 2,
        },
        'blocks': {
            'base': ResBlock,
            'filters': [128, 64, 32, 16],
            'attention': 'scse',
        },
    },
    'head': {
        'base_block': ResBlock,
        'filters': [16, 8],
        'attention': 'scse'
    },
    'output': 'sigmoid',
    # Train configuration
    'loss': dice_loss,
    'optimizer': {'name': 'Adam', 'lr': 0.01,},
    # 'optimizer': {'name': AdamCG, 'lr': 0.01,},
    # 'optimizer': {'name': Ranger, 'lr': 0.05,},
    "decay": {'name': 'exp', 'gamma': 0.1},
    "n_iters": 150,
    'microbatch': 4,
}


class Controller:
    """ !!."""
    #pylint: disable=unused-argument
    def __init__(self, batch_size=64, crop_shape=(1, 256, 256),
                 model_config=None, model_path=None, device=None,
                 show_plots=False, save_dir=None, logger=None, pbar=True):
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

        self.targets, self.predictions = None, None
        self.train_pipeline = None
        self.make_logger()

    # Utility functions
    def make_pbar(self, iterator, desc):
        """ !!. """
        if self.pbar:
            return tqdm(iterator, total=len(iterator), desc=desc, ncols=800)
        return iterator

    def make_save_path(self, *postfix):
        """ !!. """
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, *postfix[:-1])
            os.makedirs(path, exist_ok=True)
            return os.path.join(self.save_dir, *postfix)
        return None

    def make_logger(self):
        """ !!. """
        #pylint: disable=access-member-before-definition
        if self.logger is None and self.save_dir is not None:
            handler = logging.FileHandler(self.make_save_path('controller.log'), mode='w')
            handler.setFormatter(logging.Formatter('%(asctime)s      %(message)s'))

            logger = logging.getLogger('controller_logger')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            self.logger = logger.info

    def log(self, msg):
        """ !!. """
        if self.logger is not None:
            self.logger(msg)


    # Dataset creation: geometries, labels, grids, samplers
    def make_dataset(self, cube_paths, horizon_paths=None):
        """ !!. """
        cube_paths = cube_paths if isinstance(cube_paths, (tuple, list)) else [cube_paths]

        dsi = FilesIndex(path=cube_paths, no_ext=True)
        dataset = SeismicCubeset(dsi)

        dataset.load_geometries()
        dataset.create_labels(horizon_paths or HORIZON_PATHS)

        msg = '\n'
        for idx in dataset.indices:
            msg += f'{idx}\n'
            for hor in dataset.labels[idx]:
                msg += f'    {hor.name}'
        self.log(f'Created dataset ::: {msg}')
        return dataset


    def make_grid(self, dataset, frequencies, **kwargs):
        """ !!. """
        grid_coverages = []
        for idx in dataset.indices:
            geometry = dataset.geometries[idx]
            geometry.make_quality_grid(frequencies, **kwargs)
            plot_image(geometry.quality_grid, 'quality grid',
                       cmap='Reds', interpolation='bilinear', rgb=True,
                       show_plot=self.show_plots,
                       savefig=self.make_save_path(f'quality_grid_{idx}.png'),
                       )
            grid_coverage = (np.nansum(geometry.quality_grid) /
                             (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
            self.log(f'Created grid; grid coverage is {grid_coverage}')
            grid_coverages.append(grid_coverage)
        return grid_coverages


    def make_sampler(self, dataset, bins=None, use_grid=False, side_view=False, **kwargs):
        """ !!. """
        dataset.create_sampler(quality_grid=use_grid, bins=bins)
        dataset.modify_sampler('train_sampler', finish=True, **kwargs)
        dataset.train_sampler(random.randint(0, 1000000))
        for i, idx in enumerate(dataset.indices):
            dataset.show_slices(
                src_sampler='train_sampler', normalize=False, shape=self.crop_shape,
                idx=i, make_slices='adaptive' if use_grid is True else True, side_view=side_view,
                cmap='Reds', interpolation='bilinear', show_plot=self.show_plots, figsize=(25, 25),
                savefig=self.make_save_path(f'slices_{idx}.png')
            )

            dataset.show_slices(
                src_sampler='train_sampler', normalize=True, shape=self.crop_shape,
                idx=i, make_slices='adaptive' if use_grid is True else True,
                cmap='Reds', interpolation='bilinear', show_plot=self.show_plots, figsize=(25, 25),
                savefig=self.make_save_path(f'slices_n_{idx}.png')
            )


    # Train model on a created dataset
    def train(self, dataset, n_epochs=300, use_grid=False, side_view=False):
        """ !!. """
        if self.model_path is None:
            self.logger('Train started')
            pipeline_config = {
                'model_config': {**(self.model_config or MODEL_CONFIG), 'device': self.device},
                'make_slices': 'adaptive' if use_grid is True else True,
                'side_view': side_view,
            }
            train_pipeline = (self.get_train_template() << pipeline_config) << dataset

            train_pipeline.run(D('size'), n_iters=n_epochs, bar='n',
                               bar_desc=W(V('loss_history')[-1].format('Loss is: {:7.7}')))
            plot_loss(train_pipeline.v('loss_history'), show_plot=self.show_plots,
                      savefig=self.make_save_path('model_loss.png'))

            self.train_pipeline = train_pipeline

            last_loss = np.mean(train_pipeline.v('loss_history')[-50:])
            self.logger(f'Train finished; last loss is {last_loss}')
            return last_loss


    # Inference on a chosen set of data
    def inference(self, dataset, version=1, orientation='i', overlap_factor=2, heights_range=None, **kwargs):
        """ !!. """
        # 0 -- full cube assemble
        # 1 -- assemble on chunks, then merge
        # 2 -- crop-wise Horizon creation
        self.logger(f'Starting {orientation} inference_{version} with overlap of {overlap_factor}')
        self.targets = dataset.labels[0]
        method = getattr(self, f'inference_{version}')

        if len(orientation) == 1:
            horizons = method(dataset, orientation=orientation, overlap_factor=overlap_factor,
                              heights_range=heights_range, **kwargs)
        else:
            horizons_i = method(dataset, orientation='i', overlap_factor=overlap_factor,
                                heights_range=heights_range, **kwargs)
            horizons_x = method(dataset, orientation='x', overlap_factor=overlap_factor,
                                heights_range=heights_range, **kwargs)
            horizons = Horizon.merge_list(horizons_i + horizons_x, minsize=1000)
        self.predictions = horizons

        # Log some results
        if len(horizons):
            horizons.sort(key=len, reverse=True)
            self.logger(f'Num of predicted horizons: {len(horizons)}')
            self.logger(f'Total number of points in all of the horizons {sum(len(item) for item in horizons)}')
            self.logger(f'Len max: {len(horizons[0])}')
        else:
            self.logger('Zero horizons were predicted; possible problems..?')

    def make_inference_ranges(self, dataset, heights_range):
        """ Ranges of inference. """
        geometry = dataset.geometries[0]
        spatial_ranges = [[0, item-1] for item in geometry.cube_shape[:2]]
        if heights_range is None:
            if self.targets:
                min_height = max(0, min(horizon.h_min for horizon in self.targets) - 100)
                max_height = min(geometry.depth-1, max(horizon.h_max for horizon in self.targets) + 100)
                heights_range = [min_height, max_height]
            else:
                heights_range = [0, geometry.depth-1]
        return spatial_ranges, heights_range

    def make_inference_config(self, orientation, overlap_factor):
        """ Parameters depending on orientation. """
        config = {'train_pipeline': self.train_pipeline}
        if orientation == 'i':
            crop_shape_grid = self.crop_shape
            config['side_view'] = False
            config['order'] = (0, 1, 2)
        else:
            crop_shape_grid = np.array(self.crop_shape)[[1, 0, 2]]
            config['side_view'] = 1.0
            config['order'] = (1, 0, 2)
        strides_grid = [max(1, int(item//overlap_factor))
                        for item in crop_shape_grid]
        return config, crop_shape_grid, strides_grid


    def inference_0(self, dataset, heights_range=None, orientation='i', overlap_factor=2, **kwargs):
        """ !!. """
        _ = kwargs
        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = self.make_inference_ranges(dataset, heights_range)
        config, crop_shape_grid, strides_grid = self.make_inference_config(orientation, overlap_factor)

        # Actual inference
        dataset.make_grid(dataset.indices[0], crop_shape_grid,
                          *spatial_ranges, heights_range,
                          batch_size=self.batch_size,
                          strides=strides_grid)

        inference_pipeline = (self.get_inference_template_0() << config) << dataset
        for _ in self.make_pbar(range(dataset.grid_iters), desc=f'Inference on {geometry.name} | {orientation}'):
            batch = inference_pipeline.next_batch(D('size'))

        # Convert to Horizon instances
        return Horizon.from_mask(batch.assembled_pred, dataset.grid_info, threshold=0.5, minsize=50)

    def inference_1(self, dataset, heights_range=None, orientation='i', overlap_factor=2,
                    chunk_size=100, chunk_overlap=0.2, **kwargs):
        """ !!. """
        _ = kwargs
        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = self.make_inference_ranges(dataset, heights_range)
        config, crop_shape_grid, strides_grid = self.make_inference_config(orientation, overlap_factor)

        # Actual inference
        axis = np.argmin(crop_shape_grid[:2])
        iterator = range(spatial_ranges[axis][0], spatial_ranges[axis][1], int(chunk_size*(1 - chunk_overlap)))

        horizons = []
        for chunk in self.make_pbar(iterator, desc=f'Inference on {geometry.name}| {orientation}'):
            current_spatial_ranges = copy(spatial_ranges)
            current_spatial_ranges[axis] = [chunk, min(chunk + chunk_size, spatial_ranges[axis][-1])]

            dataset.make_grid(dataset.indices[0], crop_shape_grid,
                              *current_spatial_ranges, heights_range,
                              batch_size=self.batch_size,
                              strides=strides_grid)
            inference_pipeline = (self.get_inference_template_0() << config) << dataset
            for _ in range(dataset.grid_iters):
                batch = inference_pipeline.next_batch(D('size'))

            chunk_horizons = Horizon.from_mask(batch.assembled_pred, dataset.grid_info, threshold=0.5, minsize=50)
            horizons.extend(chunk_horizons)

        return Horizon.merge_list(horizons, mean_threshold=5.5, adjacency=3, minsize=500)


    def evaluate(self, n=5, add_prefix=False, dump=False):
        """ !!. """
        #pylint: disable=cell-var-from-loop
        results = []
        for i in range(n):
            info = {}
            horizon = self.predictions[i]
            cube_name, hor_name = horizon.geometry.short_name, f'{i+1}_horizon'
            prefix = [cube_name, hor_name, f'{i}_horizon'] if add_prefix else []

            horizon.show(show_plot=self.show_plots,
                         savefig=self.make_save_path(*prefix, 'horizon_img.png'))

            with open(self.make_save_path(*prefix, 'self_results.txt'), 'w') as result_txt:
                corrs = horizon.evaluate(printer=lambda msg: print(msg, file=result_txt),
                                         plot=True, show_plot=self.show_plots,
                                         savepath=self.make_save_path(*prefix, 'corrs.png'))
            self.logger(f'horizon {i}: len {len(horizon)}, cov {horizon.coverage}, # holes {horizon.number_of_holes}')

            hm = HorizonMetrics((horizon, self.targets))
            hellinger = hm.evaluate('support_hellinger', agg='nanmean', supports=15,
                                    plot=True, show_plot=self.show_plots,
                                    savepath=self.make_save_path(*prefix, 'hellinger.png'))
            self.logger(f'Mean hellinger: {np.nanmean(hellinger)}')

            if self.targets:
                _, oinfo = hm.evaluate('find_best_match', agg=None)
                info = {**info, **oinfo}

                with open(self.make_save_path(*prefix, 'results.txt'), 'w') as result_txt:
                    hm.evaluate('compare', agg=None, hist=False,
                                plot=True, show_plot=self.show_plots,
                                plot_kwargs={'cmap' :'Reds'},
                                printer=lambda msg: print(msg, file=result_txt),
                                savepath=self.make_save_path(*prefix, 'l1.png'))
                self.logger(f'horizon {i}: wr {info["window_rate"]}, mean {info["mean"]}')

            if dump:
                horizon.dump(path=self.make_save_path(*prefix, f'dumped_horizon'))

            info['hellinger'] = np.nanmean(hellinger)
            info['corrs'] = np.nanmean(corrs)
            results.append((info))

        return results



    def get_train_template(self):
        """ !!. """
        # model_config
        # crop_shape
        # make_slices
        train_template = (
            Pipeline()
            # Initialize pipeline variables and model
            .init_variable('loss_history', [])
            .init_model('dynamic', EncoderDecoder, 'ED', C('model_config'))

            # Load data/masks
            .crop(points=D('train_sampler')(self.batch_size),
                  shape=self.crop_shape, make_slices=C('make_slices'),
                  side_view=C('side_view', default=False))
            .create_masks(dst='masks', width=3)
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'], shape=self.crop_shape)
            .scale(mode='q', src='images')

            # Augmentations
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -15, 15)),
                    src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', 0.85, 1.15)),
                      src=['images', 'masks'], p=0.3)
            .elastic_transform(alpha=P(R('uniform', 35, 45)), sigma=P(R('uniform', 4, 4.5)),
                               src=['images', 'masks'], p=0.2)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))

            # Training
            .train_model('ED',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         save_to=V('loss_history', mode='a'))
        )
        return train_template


    def get_inference_template_0(self):
        """ !!. """
        # train_pipeline
        # crop_shape
        # side_view, order
        inference_template = (
            Pipeline()
            # Initialize everything
            .init_variable('result_preds', [])
            .import_model('ED', C('train_pipeline'))

            # Load data
            .crop(points=D('grid_gen')(), shape=self.crop_shape,
                  side_view=C('side_view', default=False))
            .load_cubes(dst='images')
            .adaptive_reshape(src='images', shape=self.crop_shape)
            .scale(mode='q', src='images')

            # Predict with model, then aggregate
            .predict_model('ED',
                           B('images'),
                           fetches='predictions',
                           save_to=V('result_preds', mode='e'))
            .assemble_crops(src=V('result_preds'), dst='assembled_pred',
                            grid_info=D('grid_info'), order=C('order', default=(0, 1, 2)))
        )
        return inference_template

    def get_inference_template_2(self):
        """ !!. """
