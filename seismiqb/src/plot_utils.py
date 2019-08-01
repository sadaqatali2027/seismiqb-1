""" Utility functions for plotting. """
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from ..batchflow import Pipeline, D

def plot_loss(graph_lists, labels, ylabel='Loss', figsize=(10, 10), title=None):
    """ Plot losses. """
    if not isinstance(graph_lists[0], (tuple, list)):
        graph_lists = [graph_lists]

    plt.figure(figsize=figsize)
    for i, arr in enumerate(graph_lists):
        plt.plot(arr, label=labels[i])
    plt.xlabel('Iterations')
    plt.ylabel(ylabel, fontdict={'fontsize': 15})
    plt.grid(True)
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.legend()
    plt.show()

def plot_batch_components(batch, idx=0, *components, overlap=True, rotate_axes=0, cmaps=None, alphas=None):
    """ Plot components of batch.

    Parameters
    ----------
    batch : Batch
        Batch to get data from.

    idx : int or None
        If int, then index of desired image in list.
        If None, then no indexing is applied.

    overlap : bool
        Whether to draw images one over the other or not.

    components : str or sequence of str
        Components to get from batch and draw.

    cmaps : str or sequence of str
        Color maps for showing images.

    alphas : number or sequence of numbers
        Opacity for showing images.
    """
    if idx is not None:
        print('Images from {}'.format(batch.indices[idx][:-10]))
        imgs = [getattr(batch, comp)[idx] for comp in components]
    else:
        imgs = [getattr(batch, comp) for comp in components]

    if overlap:
        plot_images_o(imgs, ', '.join(components), rotate_axes=rotate_axes, cmaps=cmaps, alphas=alphas)
    else:
        plot_images_s(imgs, components, rotate_axes=rotate_axes, cmaps=cmaps, alphas=alphas)


def plot_images_s(imgs, titles, rotate_axes, cmaps=None, alphas=None):
    """ Plot one or more images on separate layouts. """
    cmaps = cmaps or ['gray'] + ['viridis']*len(imgs)
    cmaps = cmaps if isinstance(cmaps, (tuple, list)) else [cmaps]

    alphas = alphas or [1**-i for i in range(len(imgs))]
    alphas = alphas if isinstance(alphas, (tuple, list)) else [alphas**-i for i in range(len(imgs))]

    _, ax = plt.subplots(1, len(imgs), figsize=(8*len(imgs), 10))
    for i, (img, title, cmap, alpha) in enumerate(zip(imgs, titles, cmaps, alphas)):
        img = _to_img(img, rotate_axes=rotate_axes, convert=False)

        ax_ = ax[i] if len(imgs) > 1 else ax
        ax_.imshow(img, alpha=alpha, cmap=cmap)
        ax_.set_title(title, fontdict={'fontsize': 15})
    plt.show()


def plot_images_o(imgs, title, rotate_axes, cmaps=None, alphas=None):
    """ Plot one or more images with overlap. """
    cmaps = cmaps or ['gray'] + ['Reds']*len(imgs)
    alphas = alphas or [1**-i for i in range(len(imgs))]

    plt.figure(figsize=(15, 15))
    for i, (img, cmap, alpha) in enumerate(zip(imgs, cmaps, alphas)):
        img = _to_img(img, rotate_axes=rotate_axes, convert=(i > 0))
        plt.imshow(img, alpha=alpha, cmap=cmap)

    plt.title(title, fontdict={'fontsize': 15})
    plt.show()


def _to_img(data, rotate_axes=0, convert=False):
    for _ in range(rotate_axes):
        data = np.moveaxis(data, 0, -1)

    shape = data.shape
    if len(shape) == 2:
        data = data[:, :].T
    elif len(shape) == 3:
        data = data[:, :, 0].T
    elif len(shape) == 4:
        data = data[:, :, 0, 0].T

    if convert:
        background = np.zeros((*data.shape, 4))
        background[:, :, 0] = data
        background[:, :, -1] = (data != 0).astype(int)
        return background
    return data


def plot_slide(dataset, idx=0, iline=0, *components, overlap=True):
    """ Slidify! """
    cube_name = dataset.indices[idx]
    cube_shape = dataset.geometries[cube_name].cube_shape
    point = np.array([[cube_name, iline, 0, 0]], dtype=object)

    pipeline = (Pipeline()
                .load_component(src=[D('geometries'), D('labels')],
                                dst=['geometries', 'labels'])
                .crop(points=point,
                      shape=[1] + cube_shape[1:])
                .load_cubes(dst='data_crops')
                .create_masks(dst='mask_crops', width=2)
                .rotate_axes(src=['data_crops', 'mask_crops'])
                .scale(mode='normalize', src='data_crops')
                .add_axis(src='mask_crops', dst='mask_crops'))

    batch = (pipeline << dataset).next_batch(len(dataset), n_epochs=None)
    plot_batch_components(batch, 0, overlap, *components)
    return batch


def show_labels(dataset, ix=0):
    """ Not empty! """
    name = dataset.indices[ix]
    geom = dataset.geometries[name]
    labels = dataset.labels[name]
    possible_coordinates = [[il, xl] for il in geom.ilines for xl in geom.xlines]

    background = np.zeros((geom.ilines_len, geom.xlines_len))
    img = labels_matrix(background, np.array(possible_coordinates), labels,
                        geom.ilines_offset, geom.xlines_offset)

    _, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title('Known labels for cube (yellow is known)', fontdict={'fontsize': 20})
    plt.xlabel('XLINES', fontdict={'fontsize': 20})
    plt.ylabel('ILINES', fontdict={'fontsize': 20})
    plt.show()

@njit
def labels_matrix(background, possible_coordinates, labels,
                  ilines_offset, xlines_offset):
    """ Jitify! """
    for i in range(len(possible_coordinates)):
        point = possible_coordinates[i, :]
        if labels.get((point[0], point[1])) is not None:
            background[point[0] - ilines_offset, point[1] - xlines_offset] += len(labels.get((point[0], point[1])))
    return background

def plot_stratum_predictions(cubes, targets, predictions, n_rows=None):
    """ Plot a set of stratum predictions along with cubes and targets.
    """
    n_rows = n_rows or len(cubes)
    cubes = np.squeeze(cubes)

    # transform ohe to labels
    targets, predictions = np.argmax(targets, axis=-1), np.argmax(predictions, axis=-1)

    # plot crops
    figs, axes = plt.subplots(n_rows, 3, figsize=(3 * 4, n_rows * 4))
    for i, nr in enumerate(range(n_rows)):
        vmin, vmax = np.min(targets[nr]) - 1, np.max(targets[nr]) + 1
        axes[i, 0].imshow(cubes[nr].T)
        axes[i, 1].imshow(targets[nr].T, vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(predictions[nr].T, vmin=vmin, vmax=vmax)

        axes[i, 0].set_title('Input crop')
        axes[i, 1].set_title('True mask')
        axes[i, 2].set_title('Predicted mask')
