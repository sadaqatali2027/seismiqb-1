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


def load(dataset, p=None, postfix=None):
    postfix = postfix or '/FORMAT_HORIZONTS/*'

    paths_txt = {ds.indices[i]: glob('/'.join(ds.get_fullpath(ds.indices[i])))}

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


def plot_loss(*lst, title=None):
    lst = lst if isinstance(lst[0], (tuple, list)) else [lst]

    plt.figure(figsize=(8, 5))
    for loss_history in lst:
        plt.plot(loss_history)

    plt.grid(True)
    plt.xlabel('Iterations', fontdict={'fontsize': 15})
    plt.ylabel('Loss', fontdict={'fontsize': 15})
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.show()



def plot_batch_components(batch, *components, n=5):
    n_comp = len(components)
    n_crops = len(getattr(batch, components[0]))

    indices = np.random.choice(n_crops, n)

    for idx in indices:
        print('Images from {}'.format(batch.indices[idx][:-10]))
        fig, ax = plt.subplots(1, n_comp, figsize=(8*n_comp, 10))
        for i, comp in enumerate(components):
            data = getattr(batch, comp)[idx]

            shape = data.shape
            if len(shape) == 2:
                data = data[:, :].T
            elif len(shape) == 3:
                data = data[:, :, 0].T
            elif len(shape) == 4:
                data = data[:, :, 0, 0].T

            ax[i].imshow(data)
            ax[i].set_title(comp)

        plt.show()




