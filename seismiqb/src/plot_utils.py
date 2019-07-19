""" Utility functions for plotting. """
import numpy as np
import matplotlib.pyplot as plt



def plot_loss(*lst, title=None):
	""" Loss. """
    plt.figure(figsize=(8, 5))
    for loss_history in lst:
        plt.plot(loss_history)

    plt.grid(True)
    plt.xlabel('Iterations', fontdict={'fontsize': 15})
    plt.ylabel('Loss', fontdict={'fontsize': 15})
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.show()


def plot_batch_components(batch, idx=0, *components):
	""" Batch. """
    n_comp = len(components)
    n_crops = len(getattr(batch, components[0]))

    print('Images from {}'.format(batch.indices[idx][:-10]))
    fig, ax = plt.subplots(1, n_comp, figsize=(8*n_comp, 10))
    for i, comp in enumerate(components):
        data = getattr(batch, comp)[idx]
        data = _to_img(data)

        ax[i].imshow(data)
        ax[i].set_title(comp, fontdict={'fontsize': 15})
    plt.show()


def plot_batch_components_o(batch, idx=0, *components):
    """ Batch with overlap. """
    n_comp = len(components)
    n_crops = len(getattr(batch, components[0]))
    alphas = [4**-i for i in range(n_comp)]
    cmaps = ['gray'] + ['plasma']*n_comp

    print('Image from {}'.format(batch.indices[idx][:-10]))
    plt.figure(figsize=(15, 15))
    for i, comp in enumerate(components):
        data = getattr(batch, comp)[idx]
        data = _to_img(data)
        plt.imshow(data, alpha=alphas[i], cmap=cmaps[i])

    plt.title(comp, fontdict={'fontsize': 15})
    plt.show()


def _to_img(data):
    shape = data.shape
    if len(shape) == 2:
        data = data[:, :].T
    elif len(shape) == 3:
        data = data[:, :, 0].T
    elif len(shape) == 4:
        data = data[:, :, 0, 0].T
    return data


def plot_slide(dataset, idx=0, iline=0, components):

    cube_name = dataset.indices[idx]
    cube_shape = dataset.geometries[cube_name].cube_shape
    points = np.array([[cube_name, iline, 0, 0]], dtype=object)

    flag = int('mask_crops' in components)
    config = {'create_masks': flag,
              'load_src': [D('geometries')] + [D('labels')]*flag,
              'load_dst': ['geometries'] + ['labels']*flag}

    pipeline = (Pipeline(config=config)
                 .load_component(src=C('load_src'),
                                 dst=C('load_dst'))
                 .crop(points=point,
                       shape=[1] + cube_shape[1:])
                 .load_cubes(dst='data_crops')
                 .create_masks(dst='mask_crops', width=3, p=C('create_masks'))
                 .scale(mode='normalize', src='data_crops')
                 )

    batch = (pipeline << dataset).next_batch(len(dataset), n_epochs=None)

    plot_batch_components(batch, *components)







