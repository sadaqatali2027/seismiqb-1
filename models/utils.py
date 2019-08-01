""" Util functions for tutorials. """
from glob import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_lines_range(batch, item):
    g = batch.get(batch.indices[item], 'geometries')
    ir = (g.ilines[np.min(batch.slices[item][0])],
          g.ilines[np.max(batch.slices[item][0])])
    ir = (np.array(ir) - g.ilines[0]) / (g.
                                         ilines[-1] - g.ilines[0])

    xr = (g.xlines[np.min(batch.slices[item][1])],
          g.xlines[np.max(batch.slices[item][1])])
    xr = (np.array(xr) - g.xlines[0]) / (g.xlines[-1] - g.xlines[0])
    return ir, xr


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

def show_extension_results(val_batch, val_pipeline, cubes_numbers, figsize=(25, 10)):
    for cube in cubes_numbers:
        print(val_batch.indices[cube][:-10])
        iline = 0

        truth_img =     val_pipeline.get_variable('ext_result')[0][cube, :, :, iline].T
        truth_labels =  val_pipeline.get_variable('ext_result')[1][cube, :, :, iline, 0].T
        predicted_img = val_pipeline.get_variable('ext_result')[2][cube, :, :, iline, 0].T
        cut_mask =      val_pipeline.get_variable('ext_result')[0][cube, :, :, iline + 2].T
        predicted_simple = val_pipeline.get_variable('result')[2][cube, :, :, iline, 0].T

        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1, 5, 1)
        plt.imshow(truth_img, cmap='gray')
        plt.title('Input tensor', fontsize=20)


        fig.add_subplot(1, 5, 2)
        plt.imshow(cut_mask, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)

        plt.title('Input mask', fontsize=20)

        fig.add_subplot(1, 5, 3)
        plt.imshow(truth_labels, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)
        plt.title('True mask', fontsize=20)

        fig.add_subplot(1, 5, 4)
        plt.imshow(predicted_simple, cmap="Greens")
        plt.title('Baseline prediction', fontsize=20)

        fig.add_subplot(1, 5, 5)
        plt.imshow(predicted_img, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)
        plt.imshow(predicted_img, cmap="Blues", alpha=0.1)

        plt.title('Extension prediction', fontsize=20)
        plt.show()


def load(dataset, p=None, postfix=None):
    postfix = postfix or '/FORMAT_HORIZONTS/*'

    paths_txt = {}
    for i in range(len(dataset)):
        dir_path = '/'.join(dataset.index.get_fullpath(dataset.indices[i]).split('/')[:-1])
        dir_ = dir_path + postfix
        paths_txt[dataset.indices[i]] = glob(dir_)

    dataset = (dataset.load_geometries()
                      .load_point_clouds(paths=paths_txt)
                      .load_labels()
                      .load_samplers(p=p))
    return dataset


def compare(dataset, horizont, cube_idx=0, offset=1):
    sample_rate = dataset.geometries[dataset.indices[cube_idx]].sample_rate
    labels = dataset.labels[dataset.indices[cube_idx]]

    res, not_present = [], 0
    vals, vals_true = [], []

    for key, val in horizont.items():
        if labels.get(key) is not None:
            true_horizonts = labels[key]
            diff = abs(true_horizonts - (val+offset))
            idx = np.argmin(diff)

            res.append(diff[idx])
            vals_true.append(true_horizonts[idx])
            vals.append(val)
        else:
            not_present += 1

    print('Mean value/std of error:                  {:8.7} / {:8.7}'.format(np.mean(res), np.std(res)))
    print('Horizont length:                          {}'.format(len(horizont)))
    print('Rate in 5 ms window:                      {:8.7}'.format(sum(np.array(res) < 5/sample_rate) / len(res)))
    print('Average height/std of true horizont:      {:8.7}'.format(np.mean(vals_true)))
    print('Average height/std of predicted horizont: {:8.7}'.format(np.mean(vals)))
    print('Number of values that were labeled by model and not labeled by experts: {}'.format(not_present))

    plt.title('Distribution of errors')
    _ = plt.hist(res, bins=100)

