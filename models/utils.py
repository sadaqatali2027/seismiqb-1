""" Util functions for tutorials. """
import numpy as np
import tensorflow as tf

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