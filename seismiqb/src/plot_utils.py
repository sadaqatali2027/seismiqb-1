""" Utility functions for plotting. """
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from ..batchflow import Pipeline, D

def plot_loss(graph_lists, labels=None, ylabel='Loss', figsize=(8, 5), title=None):
    """ Plot losses.

    Parameters
    ----------
    graph_lists : sequence of arrays
        list of arrays to plot.

    labels : sequence of str
        labels for different graphs.

    ylabel : str
        y-axis label.

    figsize : tuple of int
        size of resulting figure.

    title : str
        title of resulting figure.
    """
    if not isinstance(graph_lists[0], (tuple, list)):
        graph_lists = [graph_lists]

    labels = labels or 'loss'
    labels = labels if isinstance(labels, (tuple, list)) else [labels]

    plt.figure(figsize=figsize)
    for arr, label in zip(graph_lists, labels):
        plt.plot(arr, label=label)
    plt.xlabel('Iterations', fontdict={'fontsize': 15})
    plt.ylabel(ylabel, fontdict={'fontsize': 15})
    plt.grid(True)
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.legend()
    plt.show()

def plot_batch_components(batch, *components, idx=0, overlap=True, order_axes=None, cmaps=None, alphas=None, **kwargs):
    """ Plot components of batch.

    Parameters
    ----------
    batch : Batch
        Batch to get data from.

    idx : int or None
        If int, then index of desired image in list.
        If None, then no indexing is applied.

    overlap : bool
        Whether to draw images one over the other or on separate layouts.

    order_axes : sequence of int
        Determines desired order of the axis. The first two are plotted.

    components : str or sequence of str
        Components to get from batch and draw.

    cmaps : str or sequence of str
        Color maps for showing images.

    alphas : number or sequence of numbers
        Opacity for showing images.
    """
    if idx is not None:
        print('Image from {}'.format(batch.indices[idx][:-10]))
        imgs = [getattr(batch, comp)[idx] for comp in components]
    else:
        imgs = [getattr(batch, comp) for comp in components]

    if overlap:
        plot_images_overlap(imgs, ', '.join(components), order_axes=order_axes, cmaps=cmaps, alphas=alphas, **kwargs)
    else:
        plot_images_separate(imgs, components, order_axes=order_axes, cmaps=cmaps, alphas=alphas, **kwargs)


def plot_images_separate(imgs, titles, order_axes, cmaps=None, alphas=None, **kwargs):
    """ Plot one or more images on separate layouts. """
    cmaps = cmaps or ['gray'] + ['viridis']*len(imgs)
    cmaps = cmaps if isinstance(cmaps, (tuple, list)) else [cmaps]

    alphas = alphas or 1.0
    alphas = alphas if isinstance(alphas, (tuple, list)) else [alphas**-i for i in range(len(imgs))]

    defaults = {'figsize': (8*len(imgs), 10)}
    _, ax = plt.subplots(1, len(imgs), **{**defaults, **kwargs})
    for i, (img, title, cmap, alpha) in enumerate(zip(imgs, titles, cmaps, alphas)):
        img = _to_img(img, order_axes=order_axes, convert=False)

        ax_ = ax[i] if len(imgs) > 1 else ax
        ax_.imshow(img, alpha=alpha, cmap=cmap)
        ax_.set_title(title, fontdict={'fontsize': 15})
    plt.show()


def plot_images_overlap(imgs, title, order_axes, cmaps=None, alphas=None, **kwargs):
    """ Plot one or more images with overlap. """
    cmaps = cmaps or ['gray'] + ['Reds']*len(imgs)
    alphas = alphas or [1.0 for i in range(len(imgs))]

    defaults = {'figsize': (15, 15)}
    plt.figure(**{**defaults, **kwargs})
    for i, (img, cmap, alpha) in enumerate(zip(imgs, cmaps, alphas)):
        img = _to_img(img, order_axes=order_axes, convert=(i > 0))
        plt.imshow(img, alpha=alpha, cmap=cmap)

    plt.title(title, fontdict={'fontsize': 15})
    plt.show()


def _to_img(data, order_axes=None, convert=False):
    if order_axes:
        data = np.transpose(data, order_axes)

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


def plot_slide(dataset, *components, idx=0, iline=0, overlap=True, **kwargs):
    """ Plot full slide of the given cube on the given iline. """
    cube_name = dataset.indices[idx]
    cube_shape = dataset.geometries[cube_name].cube_shape
    point = np.array([[cube_name, iline, 0, 0]], dtype=object)

    pipeline = (Pipeline()
                .load_component(src=[D('geometries'), D('labels')],
                                dst=['geometries', 'labels'])
                .crop(points=point,
                      shape=[1] + list(cube_shape)[1:])
                .load_cubes(dst='images')
                .scale(mode='normalize', src='images')
                .rotate_axes(src='images')
                )

    if 'masks' in components:
        labels_pipeline = (Pipeline()
                           .create_masks(dst='masks', width=2)
                           .rotate_axes(src='masks')
                           .add_axis(src='masks', dst='masks')
                           )
        pipeline = pipeline + labels_pipeline

    batch = (pipeline << dataset).next_batch(len(dataset), n_epochs=None)
    plot_batch_components(batch, *components, overlap=overlap, **kwargs)
    return batch


def plot_from_above(img, title, **kwargs):
    """ Plot image with a given title with predifined axis labels."""
    default_kwargs = dict(cmap='Paired')
    plt.figure(figsize=(12, 7))
    img_ = plt.imshow(img, **{**default_kwargs, **kwargs})
    plt.title(title, y=1.1, fontdict={'fontsize': 20})
    plt.colorbar(img_, fraction=0.022, pad=0.07)
    plt.xlabel('XLINES', fontdict={'fontsize': 20})
    plt.ylabel('ILINES', fontdict={'fontsize': 20})
    plt.tick_params(labeltop=True, labelright=True)
    plt.show()


def plot_from_above_rgb(img, title, **kwargs):
    """ Plot image with a given title with predifined axis labels."""
    plt.figure(figsize=(12, 7))
    plt.imshow(img, **kwargs)
    plt.title(title, y=1.1, fontdict={'fontsize': 20})
    plt.xlabel('XLINES', fontdict={'fontsize': 20})
    plt.ylabel('ILINES', fontdict={'fontsize': 20})
    plt.tick_params(labeltop=True, labelright=True)
    plt.show()


def show_labels(dataset, idx=0, hor_idx=None, src='labels', show_plot=True):
    """ Show labeled ilines/xlines from above: yellow stands for labeled regions.

    Parameters
    ----------
    idx : int
        Number of cube to show labels for.
    hor_idx : int or None
        if int, a pixel is labeled whenever it is covered by the horizon with number hor_idx.
        if None, a pixel is labeled whenever it is covered by at least one horizon.
    """
    name = dataset.indices[idx]
    geom = dataset.geometries[name]
    labels = getattr(dataset, src)[name]
    print('len ', len(labels))
    possible_coordinates = [[il, xl] for il in geom.ilines for xl in geom.xlines]

    background = np.zeros((geom.ilines_len, geom.xlines_len))
    img = labels_matrix(background, np.array(possible_coordinates), labels,
                        geom.ilines_offset, geom.xlines_offset, hor_idx)
    img[0, 0] = 0
    
    if show_plot:
        plt.figure(figsize=(12, 7))
        plt.imshow(img, cmap='Paired')
        plt.title('Known labels for cube {}'.format(name), fontdict={'fontsize': 20})
        plt.xlabel('XLINES', fontdict={'fontsize': 20})
        plt.ylabel('ILINES', fontdict={'fontsize': 20})
        plt.show()
    return img

@njit
def labels_matrix(background, possible_coordinates, labels,
                  ilines_offset, xlines_offset, hor_idx):
    """ Jit-accelerated function to check which ilines/xlines are labeled. """
    for i in range(len(possible_coordinates)):
        point = possible_coordinates[i, :]
        hor_arr = labels.get((point[0], point[1]))
        if hor_arr is not None:
            if hor_idx is None:
                background[point[0] - ilines_offset, point[1] - xlines_offset] += 1
            elif hor_arr[hor_idx] != -999:
                background[point[0] - ilines_offset, point[1] - xlines_offset] += 1
    return background


def show_sampler(dataset, idx=0, src_sampler='sampler', n=100000, eps=1):
    """ Generate a lot of points and plot their (iline, xline) positions.

    Parameters
    ----------
    idx : int
        cube-number to sample points for.
    src_sampler : str
        attribute to store the sampler.
    n : int
        size of generated sample.
    eps : int
        radius of a labeled point used for plotting.
    """
    name = dataset.indices[idx]
    geom = dataset.geometries[name]

    background = np.zeros((geom.ilines_len, geom.xlines_len))

    sampler = getattr(dataset, src_sampler)
    if not callable(sampler):
        sampler = sampler.sample

    array = sampler(n)
    array = array[array[:, 0] == name]

    if not isinstance(array[0, 1], int):
        array[:, 1:] = (array[:, 1:]*geom.cube_shape).astype(int)

    for point in array:
        background[point[1]-eps:point[1]+eps, point[2]-eps:point[2]+eps] += 1

    plt.figure(figsize=(10, 7))
    plt.imshow(background)
    plt.title('Sampled points for cube {}'.format(name), fontdict={'fontsize': 20})
    plt.xlabel('XLINES', fontdict={'fontsize': 20})
    plt.ylabel('ILINES', fontdict={'fontsize': 20})
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(array[:, -1].astype(float), bins=n//1000)
    plt.title('Height distribution of sampled points for cube {}'.format(name),
              fontdict={'fontsize': 20})
    plt.show()


def plot_stratum_predictions(cubes, targets, predictions, n_rows=None):
    """ Plot a set of stratum predictions along with cubes and targets.
    """
    n_rows = n_rows or len(cubes)
    cubes = np.squeeze(cubes)

    # transform ohe to labels
    targets, predictions = np.argmax(targets, axis=-1), np.argmax(predictions, axis=-1)

    # plot crops
    _, axes = plt.subplots(n_rows, 3, figsize=(3 * 4, n_rows * 4))
    for i in range(n_rows):
        vmin, vmax = np.min(targets[i]) - 1, np.max(targets[i]) + 1
        axes[i, 0].imshow(cubes[i].T)
        axes[i, 1].imshow(targets[i].T, vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(predictions[i].T, vmin=vmin, vmax=vmax)

        axes[i, 0].set_title('Input crop')
        axes[i, 1].set_title('True mask')
        axes[i, 2].set_title('Predicted mask')

def plot_extension_history(next_predict_pipeline, btch):
    """ Function to show single extension step. """
    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(1, 5, 1)
    plt.imshow(btch.images[0][..., 0].T)
    height, width = btch.images[0].shape[:2]
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, height, 10))
    ax.set_yticks(np.arange(0, width, 10))
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 2)
    plt.imshow(btch.masks[0][:, :, 0, 0].T)
    plt.title('true mask')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, height, 10))
    ax.set_yticks(np.arange(0, width, 10))
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 3)
    plt.imshow(btch.cut_masks[0][..., 0].T)
    plt.title('Created mask using predictions')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, height, 10))
    ax.set_yticks(np.arange(0, width, 10))
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 4)
    plt.imshow(next_predict_pipeline.get_variable('result_preds')[0][:, :, 0, 0].T)
    plt.title('predictions')
    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, height, 10))
    ax.set_yticks(np.arange(0, width, 10))
    ax.grid(color='w', linestyle='-', linewidth=.5)

    fig.add_subplot(1, 5, 5)
    plt.imshow(btch.cut_masks[0][..., 0].T + next_predict_pipeline.get_variable('result_preds')[0][:, :, 0, 0].T)
    plt.title('overlap')

    # Major ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(0, height, 10))
    ax.set_yticks(np.arange(0, width, 10))
    ax.grid(color='w', linestyle='-', linewidth=.5)
    plt.show()
    
def show_extension_results(batch, val_pipeline, cubes_numbers, ext_result='ext_result',
                           baseline_result=None, figsize=(25, 10)):
    """ Demonstrate the results of the Horizon Extension model

    Parameters
    ----------
    batch : instance of SeismicCropBatch
    val_pipeline : instance of Pipeline
        must contain Pipeline variable that stores model's input tensor, mask and predictions.
    cube_numbers : array-like of int
        Index numbers of crops in the batch to show.
    ext_result : str
        Name of pipeline variable where extension model results are saved.
    baseline_result : str, optional
        Name of pipeline variable where baseline model results are saved.
    """
    for cube in cubes_numbers:
        print(batch.indices[cube][:-10])
        iline = 0

        truth_img = val_pipeline.get_variable(ext_result)[0][cube, :, :, iline].T
        truth_labels = val_pipeline.get_variable(ext_result)[1][cube, :, :, iline, 0].T
        predicted_img = val_pipeline.get_variable(ext_result)[2][cube, :, :, iline, 0].T
        num_channels = val_pipeline.get_variable(ext_result)[0][cube, :, :, :].shape[-1]
        cut_mask = val_pipeline.get_variable(ext_result)[0][cube, :, :, iline + num_channels].T
        if baseline_result:
            predicted_simple = val_pipeline.get_variable(baseline_result)[2][cube, :, :, iline, 0].T

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

        if baseline_result:
            fig.add_subplot(1, 5, 4)
            plt.imshow(predicted_simple, cmap="Greens")
            plt.title('Baseline prediction', fontsize=20)
            fig.add_subplot(1, 5, 5)
        else:
            fig.add_subplot(1, 5, 4)
        plt.imshow(predicted_img, cmap="Blues")
        plt.imshow(truth_img, cmap="gray", alpha=0.5)
        plt.imshow(predicted_img, cmap="Blues", alpha=0.1)
        plt.title('Extension prediction', fontsize=20)
        plt.show()

