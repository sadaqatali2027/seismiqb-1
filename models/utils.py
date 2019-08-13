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
