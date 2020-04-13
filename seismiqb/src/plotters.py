""" Plotters-class containing all plotting backend for seismic cubes - data.
"""
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def channelize_image(image, total_channels, n_channel=0, greyscale=False, opacity=None):
    """ Channelize an image. Can be used to make an opaque rgb or greyscale image.
    """
    # case of a partially channelized image
    if image.ndim == 3:
        if image.shape[-1] == total_channels:
            return image
        else:
            background = np.zeros((*image.shape[:-1], total_channels))
            background[:, :, :image.shape[-1]] = image

            if opacity is not None:
                background[:, :, -1] = opacity
            return background

    # case of non-channelized image
    background = np.zeros((*image.shape, total_channels))
    background[:, :, n_channel] = image

    # in case of grescale make all 3 channels equal to supplied image
    if greyscale:
        for i in range(3):
            background[:, :, i] = image

    # add opacity if needed
    if opacity is not None:
        background[:, :, -1] = opacity * (image != 0).astype(int)

    return background



def filter_kwargs(kwargs, keys):
    kwargs_ = {}
    for key in keys:
        if key in kwargs:
            kwargs_.update({key: kwargs[key]})
    return kwargs_

def plot_image(image, mode, backend, **kwargs):
    """ Overall plotter function, redirecting plotting task to one of the methods of backend-classes.
    """
    if backend in ('matplotlib', 'plt'):
        getattr(MatplotlibPlotter(), 'plot_' + mode)(image, **kwargs)
    elif backend in ('plotly', 'go'):
        getattr(PlotlyPlotter(), 'plot_' + mode)(image, **kwargs)
    else:
        raise ValueError('{} backend is not supported!'.format(backend))

class PlotlyPlotter:
    """ Plotting backend for plotly.
    """
    def plot_single(self, image, **kwargs):
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'reversescale': True,
                    'colorscale': 'viridis' ,
                    'coloraxis_colorbar': {'title': 'amplitude'},
                    'opacity' : 1.0,
                    'title': 'Depth map',
                    'max_size' : 600}
        for_update = filter_kwargs(kwargs, list(defaults.keys()) + ['opacity', 'zmin', 'zmax', 'showscale']) # TODO: more args to add in here
        updated = {**defaults, **for_update}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['reversescale', 'colorscale', 'opacity', 'showscale'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Heatmap(z=image.T, **render_kwargs) # note the usage of Heatmap here
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)
        fig.show()

    def plot_overlap(self, image, **kwargs):
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'amplitude'},
                    'opacity' : 1.0,
                    'title': 'Seismic inline',
                    'max_size' : 600}
        for_update = filter_kwargs(kwargs, list(defaults.keys()) + ['opacity', 'zmin', 'zmax']) # TODO: more args to add in here
        updated = {**defaults, **for_update}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['zmin', 'zmax'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])

        # calculate canvas sizes
        width, height = image[0].shape[1], image[0].shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # manually combine first image in greyscale and the rest ones colored differently
        combined = channelize_image(255 * image[0].T, total_channels=4, greyscale=True)
        for img, n_channel in zip(image[1:], (0, 1, 2)):
            combined += channelize_image(255 * img.T, total_channels=4, n_channel=n_channel, opacity=updated['opacity'])
        plot_data = go.Image(z=combined, **render_kwargs) # plot manually combined image

        # plot the figure
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)
        fig.show()

    def plot_rgb(self, image, **kwargs):
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'depth'},
                    'title': 'RGB amplitudes',
                    'max_size' : 600}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()))} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Image(z=image.T, **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)
        fig.show()

    def plot_separate(self, image, **kwargs):
        # defaults
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'depth'},
                    'title': 'Seismic inline',
                    'max_size' : 600}
        grid = (1, len(image))
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()))} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['title'])
        xaxis_kwargs = filter_kwargs(updated, ['xaxis'])
        yaxis_kwargs = filter_kwargs(updated, ['yaxis'])

        # make sure that the images are greyscale and put them each on separate canvas
        fig = make_subplots(rows=grid[0], cols=grid[1])
        for i in range(grid[1]):
            img = channelize_image(255 * image[i].T, total_channels=4, greyscale=True, opacity=1)
            fig.add_trace(go.Image(z=img, **render_kwargs), row=1, col=i + 1)
            fig.update_xaxes(row=1, col=i + 1, **xaxis_kwargs['xaxis'])
            fig.update_yaxes(row=1, col=i + 1, **yaxis_kwargs['yaxis'])
        fig.update_layout(**label_kwargs)
        fig.show()


class MatplotlibPlotter:
    """ Plotting backend for matplotlib.
    """
    def plot_single(self, image, **kwargs):
        # update defaults
        defaults = {'figsize': (12, 7),
                    'label': 'Depth map',
                    'cmap': 'viridis_r',
                    'colorbar': True,
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'fontsize': 20,
                    'labeltop': True,
                    'labelright': True}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        tick_params = filter_kwargs(updated, ['labeltop', 'labelright'])

        # channelize and plot the image
        image = channelize_image(image, total_channels=3)
        plt.figure(figsize=updated['figsize'])
        _= plt.imshow(image.T, **render_kwargs)

        # add titles and labels
        plt.title(y=1.1, **label_kwargs)
        plt.xlabel(**xaxis_kwargs)
        plt.ylabel(**yaxis_kwargs)
        if updated['colorbar']:
            plt.colorbar()
        plt.tick_params(**tick_params)
        plt.show()

    def plot_overlap(self, image, **kwargs):
        defaults = {'figsize': (12, 7),
                    't': 'Seismic inline',
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'cmap': 'gray',
                    'fontsize': 20,
                    'opacity': 1.0}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['vmin', 'vmax',
                                                                                'family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax'])
        label_kwargs = filter_kwargs(updated, ['t', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        # channelize images and put them on a canvas
        fig, ax = plt.subplots(figsize=updated['figsize'])
        ax.imshow(image[0].T, **render_kwargs) # note transposition in here
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        for img, n_channel in zip(image[1:], (0, 1, 2)):
            ax.imshow(channelize_image(img.T, total_channels=4, n_channel=n_channel, opacity=updated['opacity']), **render_kwargs)
        fig.suptitle(y=1.1, **label_kwargs)
        plt.show()


    def plot_rgb(self, image, **kwargs):
        # update defaults
        defaults = {'figsize': (12, 7),
                    'label': 'RGB amplitudes',
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'fontsize': 20,
                    'labeltop': True,
                    'labelright': True}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        tick_params = filter_kwargs(updated, ['labeltop', 'labelright'])


        # channelize and plot the image
        image = channelize_image(image, total_channels=3)
        plt.figure(figsize=updated['figsize'])
        _= plt.imshow(image.T, **render_kwargs)

        # add titles and labels
        plt.title(y=1.1, **label_kwargs)
        plt.xlabel(**xaxis_kwargs)
        plt.ylabel(**yaxis_kwargs)
        plt.tick_params(**tick_params)
        plt.show()

    def plot_separate(self, image, **kwargs):
        # embedded params
        defaults = {'figsize': (6 * len(image), 15),
                    't': 'Seismic inline',
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'cmap': 'gray',
                    'fontsize': 20}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['vmin', 'vmax',
                                                                                'family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax'])
        label_kwargs = filter_kwargs(updated, ['t', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        grid = (1, len(image))
        fig, ax = plt.subplots(*grid, figsize=updated['figsize'])

        # plot image
        ax[0].imshow(image[0].T, **render_kwargs)
        for i in range(1, len(image)):
            ax[i].imshow(image[i].T, **render_kwargs) # grey colorschemes are embedded here
                                                  # might be more beautiful if red-color is in here
                                                  # if so, param for colorscheme is to be added
            ax[i].set_xlabel(**xaxis_kwargs)
            ax[i].set_ylabel(**yaxis_kwargs)

        fig.suptitle(y=1.1, **label_kwargs)
        plt.show()

    def plot_histogram(self, image, **kwargs):
        # update defaults
        defaults = {'bins': 50,
                    'density': True,
                    'alpha': 0.75,
                    'facecolor': 'b',
                    'label': 'Amplitudes histogram',
                    'xlabel': 'xlines',
                    'ylabel': 'density',
                    'fontsize': 20}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['family', 'color', 'xlim', 'ylim'])} # TODO: more args to add in here

        # form different groups of kwargs
        histo_kwargs = filter_kwargs(updated, ['bins', 'density', 'alpha', 'facecolor'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xlabel_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        ylabel_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color', 'ylim'])
        xaxis_kwargs = filter_kwargs(updated, ['xlim'])
        yaxis_kwargs = filter_kwargs(updated, ['ylim'])

        _, _, _ = plt.hist(image.flatten(), **histo_kwargs)
        plt.xlabel(**xlabel_kwargs)
        plt.ylabel(**ylabel_kwargs)
        plt.title(**label_kwargs)
        plt.xlim(xaxis_kwargs.get('xlim'))  # these are positional ones
        plt.ylim(yaxis_kwargs.get('ylim'))

        plt.show()
