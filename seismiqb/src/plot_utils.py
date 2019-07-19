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


def plot_batch_components(batch, *components, idx=None):
	""" Batch. """
    n_comp = len(components)
    n_crops = len(getattr(batch, components[0]))

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
        ax[i].set_title(comp, fontdict={'fontsize': 15})

        plt.show()