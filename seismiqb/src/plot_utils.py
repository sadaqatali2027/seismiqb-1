""" Utility functions for plotting. """
#pylint: disable=expression-not-assigned
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..batchflow import Pipeline



def plot_loss(graph_lists, labels=None, ylabel='Loss', figsize=(8, 5), title=None, savefig=False, show_plot=True):
    """ Plot losses.

    Parameters
    ----------
    graph_lists : sequence of arrays
        Arrays to plot.
    labels : sequence of str
        Labels for different graphs.
    ylabel : str
        y-axis label.
    figsize : tuple of int
        Size of the resulting figure.
    title : str
        Title of the resulting figure.
    savefig : bool or str
        If str, then path for image saving.
        If False, then image is not saved.
    show_plot: bool
        Whether to show image in output stream.
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

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
    plt.show() if show_plot else plt.close()



def plot_batch_components(batch, *components, idx=0, plot_mode='overlap', order_axes=None, meta_title=None,
                          cmaps=None, alphas=None, **kwargs):
    """ Plot components of batch.

    Parameters
    ----------
    batch : Batch
        Batch to get data from.
    idx : int or None
        If int, then index of desired image in list.
        If None, then no indexing is applied.
    plot_mode : bool
        If 'overlap', then images are drawn one over the other.
        If 'facies', then images are drawn one over the other with transparency.
        If 'separate', then images are drawn on separate layouts.
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
        imgs = [getattr(batch, comp)[idx] for comp in components]
    else:
        imgs = [getattr(batch, comp) for comp in components]

    if plot_mode in ['overlap']:
        plot_images_overlap(imgs, ', '.join(components), order_axes=order_axes, meta_title=meta_title,
                            cmaps=cmaps, alphas=alphas, **kwargs)
    elif plot_mode in ['separate']:
        plot_images_separate(imgs, components, order_axes=order_axes, meta_title=meta_title,
                             cmaps=cmaps, alphas=alphas, **kwargs)
    elif plot_mode in ['facies']:
        plot_images_transparent(imgs, ', '.join(components), order_axes=order_axes, meta_title=meta_title,
                                cmaps=cmaps, alphas=alphas, **kwargs)


def plot_images_separate(imgs, titles, order_axes, meta_title=None, savefig=False, show_plot=True,
                         cmaps=None, alphas=None, **kwargs):
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
    plt.suptitle(meta_title, y=0.93, fontsize=20)

    if savefig:
        plt.savefig(savefig)
    plt.show() if show_plot else plt.close()

def plot_images_overlap(imgs, title, order_axes, meta_title=None, savefig=False, show_plot=True,
                        cmaps=None, alphas=None, **kwargs):
    """ Plot one or more images with overlap. """
    cmaps = cmaps or ['gray'] + ['Reds']*len(imgs)
    alphas = alphas or [1.0]*len(imgs)

    defaults = {'figsize': (15, 15)}
    plt.figure(**{**defaults, **kwargs})
    for i, (img, cmap, alpha) in enumerate(zip(imgs, cmaps, alphas)):
        img = _to_img(img, order_axes=order_axes, convert=(i > 0))
        plt.imshow(img, alpha=alpha, cmap=cmap)

    plt.title('{}\n{}'.format(meta_title, title), fontdict={'fontsize': 15})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
    plt.show() if show_plot else plt.close()

def plot_images_transparent(imgs, title, order_axes, meta_title=None, savefig=False, show_plot=True,
                            cmaps=None, alphas=None, **kwargs):
    """ Plot one or more images with overlap and transparency. """
    cmaps = cmaps or ['gray'] + [None]*len(imgs)
    alphas = alphas or [1.0] + [0.25]*len(imgs)

    defaults = {'figsize': (15, 15)}
    plt.figure(**{**defaults, **kwargs})
    for i, (img, cmap, alpha) in enumerate(zip(imgs, cmaps, alphas)):
        img = _to_img(img, order_axes=order_axes, convert=False, normalize=(i > 0))
        plt.imshow(img, alpha=alpha, cmap=cmap)

    plt.title('{}\n{}'.format(meta_title, title), fontdict={'fontsize': 15})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
    plt.show() if show_plot else plt.close()

def _to_img(data, order_axes=None, convert=False, normalize=False):
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

    if normalize:
        colors = Normalize(0, np.max(data), clip=True)(data)
        colors = plt.cm.gist_rainbow(colors)
        colors[:, :, -1] = (data != 0).astype(int)
        return colors
    return data



def plot_slide(dataset, *components, idx=0, n_line=0, plot_mode='overlap', mode='iline', **kwargs):
    """ Plot full slide of the given cube on the given n_line.

    Parameters
    ----------
    components : sequence
        Names of components to plot. Usually it is either ('images',) or ('images', 'masks').
    idx : int
        Number of cube in the dataset index to use.
    mode : str
        Axis to cut along. Can be either `iline` or `xline`.
    n_line : int
        Number of line to show.
    plot_mode : str
        Way of showing results. Can be either `overlap`, `separate`, `facies`.
    """
    cube_name = dataset.indices[idx]
    geom = dataset.geometries[cube_name]
    crop_shape = np.array(geom.cube_shape)

    if mode in ['i', 'il', 'iline']:
        point = np.array([[cube_name, n_line, 0, 0]], dtype=object)
        crop_shape[0] = 1
    elif mode in ['x', 'xl', 'xline']:
        point = np.array([[cube_name, 0, n_line, 0]], dtype=object)
        crop_shape[1] = 1

    pipeline = (Pipeline()
                .crop(points=point,
                      shape=crop_shape)
                .load_cubes(dst='images')
                .scale(mode='normalize', src='images')
                .rotate_axes(src='images')
                )

    if 'masks' in components:
        horizons = kwargs.pop('horizons', -1)
        width = kwargs.pop('width', 4)
        labels_pipeline = (Pipeline()
                           .create_masks(dst='masks', width=width, horizons=horizons)
                           .rotate_axes(src='masks')
                           )
        pipeline = pipeline + labels_pipeline

    batch = (pipeline << dataset).next_batch(len(dataset), n_epochs=None)

    if mode in ['i', 'il', 'iline']:
        meta_title = 'iline {} out of {} on {}'.format(n_line, geom.ilines_len, cube_name)
        plot_batch_components(batch, *components, meta_title=meta_title, plot_mode=plot_mode, **kwargs)
    elif mode in ['x', 'xl', 'xline']:
        meta_title = 'xline {} out of {} on {}'.format(n_line, geom.xlines_len, cube_name)
        plot_batch_components(batch, *components, meta_title=meta_title, plot_mode=plot_mode,
                              order_axes=(2, 1, 0), **kwargs)
    return batch



def plot_image(img, title=None, xlabel='xlines', ylabel='ilines', rgb=False, savefig=False, show_plot=True, **kwargs):
    """ Plot image with a given title with predifined axis labels.

    Parameters
    ----------
    img : array-like
        Image to plot.
    xlabel, ylabel : str
        Labels of axis.
    title : str
        Image title.
    rgb : bool
        If False, then colorbar is added to image.
        If True, then channels of `img` are used to reflect colors.
    savefig : bool or str
        If str, then path for image saving.
        If False, then image is not saved.
    show_plot: bool
        Whether to show image in output stream.
    """
    img = np.squeeze(img)
    default_kwargs = dict(cmap='Paired') if rgb is False else {}
    plt.figure(figsize=kwargs.pop('figsize', (12, 7)))

    img_ = plt.imshow(img, **{**default_kwargs, **kwargs})

    if title:
        plt.title(title, y=1.1, fontdict={'fontsize': 20})
    if xlabel:
        plt.xlabel(xlabel, fontdict={'fontsize': 20})
    if ylabel:
        plt.ylabel(ylabel, fontdict={'fontsize': 20})
    if rgb is False:
        plt.colorbar(img_, fraction=0.022, pad=0.07)
    plt.tick_params(labeltop=True, labelright=True)

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
    plt.show() if show_plot else plt.close()


def plot_image_roll(img, title=None, xlabel='xlines', ylabel='ilines', cols=2, rgb=False,
                    savefig=False, show_plot=True, **kwargs):
    """ Plot multiple images on grid.

    Parameters
    ----------
    img : array-like
        Image(s) to plot.
    cols : int
        Number of columns in grid.
    xlabel, ylabel : str
        Labels of axis.
    title : str
        Image title.
    rgb : bool
        If False, then colorbar is added to image.
        If True, then channels of `img` are used to reflect colors.
    savefig : bool or str
        If str, then path for image saving.
        If False, then image is not saved.
    show_plot: bool
        Whether to show image in output stream.
    """
    if img.ndim == 2 or img.shape[-1] == 1:
        plot_image(img, title=title, xlabel=xlabel, ylabel=ylabel, rgb=rgb,
                   savefig=savefig, show_plot=show_plot, **kwargs)
    else:
        default_kwargs = dict(cmap='Paired') if rgb is False else {}
        n = img.shape[-1]
        rows = n // cols + 1

        if img.shape[0] > img.shape[1]:
            col_size, row_size, fraction, y_margin = 8, 8, 0.098, 0.95
        else:
            col_size, row_size, fraction, y_margin = 12, 7, 0.021, 0.91

        fig, ax = plt.subplots(rows, cols, figsize=(col_size*cols, row_size*rows))
        for i in range(rows):
            for j in range(cols):
                n_axis = i*cols + j
                if n_axis < n:
                    img_n = img[:, :, n_axis]
                    img_ = ax[i][j].imshow(img_n, **{**default_kwargs, **kwargs})

                    ax[i][j].set_xlabel(xlabel, fontdict={'fontsize': 20})
                    ax[i][j].set_ylabel(ylabel, fontdict={'fontsize': 20})
                    ax[i][j].tick_params(labeltop=True, labelright=True)
                    ax[i][j].set_title('trial {}; mean value is {:.4}'.format(n_axis+1, np.mean(img_n[img_n != 0.0])),
                                       fontdict={'fontsize': 15})
                    if rgb is False:
                        fig.colorbar(img_, ax=ax[i][j], fraction=fraction, pad=0.1)
                else:
                    fig.delaxes(ax[i][j])
        plt.suptitle(title, y=y_margin, fontsize=20)

        if savefig:
            plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
        plt.show() if show_plot else plt.close()


def show_sampler(sampler, cube_name=None, geom=None, n=100000, eps=1, show_unique=False,
                 savefig=False, show_plot=True, **kwargs):
    """ Generate a lot of points and plot their (iline, xline) positions. """
    eps = [eps, eps] if isinstance(eps, int) else eps
    background = np.zeros((geom.ilines_len, geom.xlines_len))

    if not callable(sampler):
        sampler = sampler.sample
    array = sampler(n)

    if cube_name is not None:
        array = array[array[:, 0] == cube_name]
        array = array[:, 1:]

    if not isinstance(array[0, 0], int):
        array = np.rint(array.astype(float)*geom.cube_shape).astype(int)
    for point in array:
        background[point[0]-eps[0]:point[0]+eps[0], point[1]-eps[1]:point[1]+eps[1]] += 1


    plot_image(background, title='Sampled points for cube {}'.format(cube_name),
               show_plot=show_plot, savefig=savefig, **kwargs)

    plt.hist(array[:, -1].astype(float), bins=n//1000)
    plt.title('Height distribution of sampled points for cube {}'.format(cube_name), fontdict={'fontsize': 20})

    if savefig:
        plt.savefig(savefig.split('.')[0] + '_hist.png', bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()
        if show_unique:
            uniques = np.unique(array[:, 0])
            print('Unique inlines are: {}'.format(uniques))
    else:
        plt.close()



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
        cut_mask = val_pipeline.get_variable(ext_result)[0][cube, :, :, iline + 2].T
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
