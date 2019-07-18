""" Util functions for tutorials. """
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_lines_range(batch, item):
    g = batch.get(batch.indices[item], 'geometries')
    ir = (g.ilines[np.min(batch.slices[item][0])],
          g.ilines[np.max(batch.slices[item][0])])
    ir = (np.array(ir) - g.ilines[0]) / (g.ilines[-1] - g.ilines[0])

    xr = (g.xlines[np.min(batch.slices[item][1])],
          g.xlines[np.max(batch.slices[item][1])])
    xr = (np.array(xr) - g.xlines[0]) / (g.xlines[-1] - g.xlines[0])
    return ir, xr


def make_data_extension(batch, **kwargs):
    data_x = []
    for i, cube in enumerate(batch.data_crops):
        cut_mask_ = batch.cut_mask_crops[i]
        data_x.append(np.concatenate([cube, cut_mask_], axis=-1))

    data_y = []
    
    for cube in batch.mask_crops:
        data_y.append(cube)
    return {"feed_dict": {'cubes': data_x,
                          'masks': data_y}}

def predictions(x):
    return tf.expand_dims(x, axis=-1, name='expand')

def show_input_data(btch):
    cv = 0.2
    i = np.random.choice(64)

    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))
    axes[0].imshow(btch.data_crops[i][:, :, 0].T)
    axes[0].set_title('Image', fontsize=20)
    axes[1].imshow(btch.data_crops[i][:, :, 0].T, cmap='Greens')
    axes[1].imshow(btch.mask_crops[i][:, :, 0].T, cmap='Greens', alpha=0.3)
    axes[1].set_title('Mask', fontsize=20)
    axes[2].imshow(btch.data_crops[i][:, :, 0].T, cmap='Greens')
    axes[2].imshow(btch.cut_mask_crops[i][:, :, 0].T, cmap='Greens', alpha=0.3)
    axes[2].set_title('Thin out mask', fontsize=20)
    plt.show()

def show_extension_results(val_batch, val_pipeline, cubes_numbers):
    for cube in cubes_numbers:
        print(val_batch.indices[cube][:-10])
        iline = 0

        truth_img =     val_pipeline.get_variable('ext_result')[0][0][cube, :, :, iline].T
        truth_labels =  val_pipeline.get_variable('ext_result')[0][1][cube, :, :, iline, 0].T
        predicted_img = val_pipeline.get_variable('ext_result')[0][2][cube, :, :, iline, 0].T
        cut_mask =      val_pipeline.get_variable('ext_result')[0][0][cube, :, :, iline + 2].T
        predicted_simple = val_pipeline.get_variable('result')[0][2][cube, :, :, iline, 0].T

        fig = plt.figure(figsize=(25, 10))
        fig.add_subplot(1, 5, 1)
        plt.imshow(truth_img, cmap='gray')
        plt.title('Input tensor', fontsize=20)


        fig.add_subplot(1, 5, 2)
        plt.imshow(cut_mask, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)

        plt.title('Input mask', fontsize=20)

        fig.add_subplot(1, 5, 3)Extension model prediction
        plt.imshow(truth_labels, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)
        plt.title('True mask', fontsize=20)

        fig.add_subplot(1, 5, 4)
        plt.imshow(predicted_simple, cmap="Greens")
        plt.title('Baseline model prediction', fontsize=20)
        plt.show()

        fig.add_subplot(1, 5, 5)
        plt.imshow(predicted_img, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)
        plt.imshow(predicted_img, cmap="Blues", alpha=0.1)

        plt.title('Extension model prediction', fontsize=20)

