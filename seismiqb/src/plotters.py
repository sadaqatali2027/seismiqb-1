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
    """ Filter the dict of kwargs leaving only supplied keys.
    """
    kwargs_ = {}
    for key in keys:
        if key in kwargs:
            kwargs_.update({key: kwargs[key]})
    return kwargs_

def plot_image(image, mode, backend, **kwargs):
    """ Overall plotter function, redirecting plotting task to one of the methods of backend-classes.
    """
    if backend in ('matplotlib', 'plt'):
        getattr(MatplotlibPlotter(), mode)(image, **kwargs)
    elif backend in ('plotly', 'go'):
        getattr(PlotlyPlotter(), mode)(image, **kwargs)
    else:
        raise ValueError('{} backend is not supported!'.format(backend))

def matplotlib_dec(cls):
    """ Decorator adding savefigure-capabilities into MatplotlibPlotter.
    """
    names=['rgb', 'overlap', 'single', 'separate', 'histogram']
    # update each method
    for name in names:
        old_ = getattr(cls, name)
        def updated(self, *args, old_=old_, **kwargs):
            # pop savefig-kwarg and show-kwarg
            show = kwargs.pop('show', True)
            save = kwargs.pop('save', None)
            plt_ = old_(self, *args, **kwargs)
            # save if necessary and render
            if save is not None:
                plt_.savefig(**save)
            if show:
                plt_.show()
            else:
                plt_.close()

        # change the class-method
        setattr(cls, name, updated)

    return cls

def plotly_dec(cls):
    """ Decorator adding savefigure-capabilities into PlotlyPlotter.
    """
    names=['rgb', 'overlap', 'single', 'separate']

    # update each method
    for name in names:
        old_ = getattr(cls, name)
        def updated(self, *args, old_=old_, **kwargs):
            # pop savefig-kwarg and show-kwarg
            show = kwargs.pop('show', True)
            save = kwargs.pop('save', None)
            fig = old_(self, *args, **kwargs)
            # save if necessary and render
            if save is not None:
                fig.write_image(**save)
            if show:
                fig.show()

        # change the class-method
        setattr(cls, name, updated)

    return cls

@plotly_dec
class PlotlyPlotter:
    """ Plotting backend for plotly.
    """
    def single(self, image, **kwargs):
        """ Plot single image/heatmap using plotly.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            zmin : float
                the lowest brightness-level to be rendered.
            zmax : float
                the highest brightness-level to be rendered.
            opacity : float
                transparency-level of the rendered image
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            other
        """
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'reversescale': True,
                    'colorscale': 'viridis',
                    'coloraxis_colorbar': {'title': 'amplitude'},
                    'opacity' : 1.0,
                    'title': 'Depth map',
                    'max_size' : 600,
                    'slice': (slice(None, None), slice(None, None))}
        for_update = filter_kwargs(kwargs, list(defaults.keys()) + ['opacity', 'zmin', 'zmax', 'showscale']) # TODO: more args to add in here
        updated = {**defaults, **for_update}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['reversescale', 'colorscale', 'opacity', 'showscale'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Heatmap(z=image.T[slc], **render_kwargs) # note the usage of Heatmap here
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        return fig

    def overlap(self, images, **kwargs):
        """ Plot several images on one canvas using plotly: render the first one in greyscale
        and the rest ones in opaque 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting. Can store up to four images.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            opacity : float
                opacity of 'rgb' channels.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            other
        """
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'amplitude'},
                    'opacity' : 1.0,
                    'title': 'Seismic inline',
                    'max_size' : 600,
                    'slice': (slice(None, None), slice(None, None))}
        for_update = filter_kwargs(kwargs, list(defaults.keys()) + ['opacity', 'zmin', 'zmax']) # TODO: more args to add in here
        updated = {**defaults, **for_update}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['zmin', 'zmax'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = images[0].shape[1], images[0].shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # manually combine first image in greyscale and the rest ones colored differently
        combined = channelize_image(255 * images[0].T, total_channels=4, greyscale=True)
        for img, n_channel in zip(images[1:], (0, 1, 2)):
            combined += channelize_image(255 * img.T, total_channels=4, n_channel=n_channel, opacity=updated['opacity'])
        plot_data = go.Image(z=combined[slc], **render_kwargs) # plot manually combined image

        # plot the figure
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        return fig

    def rgb(self, image, **kwargs):
        """ Plot one image in 'rgb' using plotly.

        Parameters
        ----------
        image : np.ndarray
            3d-array containing channeled rgb-image.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of the rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            other
        """
        # update defaults to make total dict of kwargs
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'depth'},
                    'title': 'RGB amplitudes',
                    'max_size' : 600,
                    'slice': (slice(None, None), slice(None, None))}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()))} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Image(z=np.swapaxes(image, 0, 1)[slc], **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        return fig

    def separate(self, images, **kwargs):
        """ Plot several images on a row of canvases using plotly.
        TODO: add grid support.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            other
        """
        # defaults
        defaults = {'xaxis': {'title_text': 'xlines', 'titlefont': {'size': 30}},
                    'yaxis': {'title_text': 'height', 'titlefont': {'size': 30}, 'autorange': 'reversed'},
                    'coloraxis_colorbar': {'title': 'depth'},
                    'title': 'Seismic inline',
                    'max_size' : 600}
        grid = (1, len(images))
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()))} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['title'])
        xaxis_kwargs = filter_kwargs(updated, ['xaxis'])
        yaxis_kwargs = filter_kwargs(updated, ['yaxis'])
        slc = updated['slice']

        # make sure that the images are greyscale and put them each on separate canvas
        fig = make_subplots(rows=grid[0], cols=grid[1])
        for i in range(grid[1]):
            img = channelize_image(255 * images[i].T, total_channels=4, greyscale=True, opacity=1)
            fig.add_trace(go.Image(z=img[slc], **render_kwargs), row=1, col=i + 1)
            fig.update_xaxes(row=1, col=i + 1, **xaxis_kwargs['xaxis'])
            fig.update_yaxes(row=1, col=i + 1, **yaxis_kwargs['yaxis'])
        fig.update_layout(**label_kwargs)

        return fig

@matplotlib_dec
class MatplotlibPlotter:
    """ Plotting backend for matplotlib.
    """
    def single(self, image, **kwargs):
        """ Plot single image/heatmap using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            cmap : str
                colormap of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            alpha : float
                transparency-level of the rendered image
            other
        """
        # update defaults
        defaults = {'figsize': (12, 7),
                    'label': 'Depth map',
                    'cmap': 'viridis_r',
                    'colorbar': True,
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'fontsize': 20,
                    'fraction': 0.022,
                    'pad': 0.07,
                    'labeltop': True,
                    'labelright': True}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax', 'alpha'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        tick_params = filter_kwargs(updated, ['labeltop', 'labelright'])
        colorbar_kwargs = filter_kwargs(updated, ['fraction', 'pad'])

        # channelize and plot the image
        plt.figure(figsize=updated['figsize'])
        _ = plt.imshow(image.T, **render_kwargs)

        # add titles and labels
        plt.title(y=1.1, **label_kwargs)
        plt.xlabel(**xaxis_kwargs)
        plt.ylabel(**yaxis_kwargs)
        if updated['colorbar']:
            plt.colorbar(**colorbar_kwargs)
        plt.tick_params(**tick_params)

        return plt

    def overlap(self, images, **kwargs):
        """ Plot several images on one canvas using matplotlib: render the first one in greyscale
        and the rest ones in 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : tuple or list
            sequence of 2d-arrays for plotting. Supports up to 4 images.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            y : float
                height of the title
            cmap : str
                colormap to render the first image in.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            other
        """
        defaults = {'figsize': (12, 7),
                    'label': 'Seismic inline',
                    'y' : 1.1,
                    'xlabel': 'xlines',
                    'ylabel': 'ilines',
                    'cmap': 'gray',
                    'fontsize': 20,
                    'opacity': 1.0}
        updated = {**defaults, **filter_kwargs(kwargs, list(defaults.keys()) + ['vmin', 'vmax',
                                                                                'family', 'color'])} # TODO: more args to add in here

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color', 'y'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        # channelize images and put them on a canvas
        fig, ax = plt.subplots(figsize=updated['figsize'])
        ax.imshow(images[0].T, **render_kwargs) # note transposition in here
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        for img, n_channel in zip(images[1:], (0, 1, 2)):
            ax.imshow(channelize_image(img.T, total_channels=4, n_channel=n_channel, opacity=updated['opacity']),
                                       **render_kwargs)
        plt.title(**label_kwargs)

        return plt


    def rgb(self, image, **kwargs):
        """ Plot one image in 'rgb' using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            3d-array containing channeled rgb-image.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            other
        """

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
        _ = plt.imshow(np.swapaxes(image, 0, 1), **render_kwargs)

        # add titles and labels
        plt.title(y=1.1, **label_kwargs)
        plt.xlabel(**xaxis_kwargs)
        plt.ylabel(**yaxis_kwargs)
        plt.tick_params(**tick_params)

        return plt

    def separate(self, images, **kwargs):
        """ Plot several images on a row of canvases using matplotlib.
        TODO: add grid support.

        Parameters
        ----------
        images : tuple or list
            sequence of 2d-arrays for plotting. Supports up to 4 images.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            t : str
                overal title of rendered image.
            label : list or tuple
                sequence of titles for each image.
            cmap : str
                colormap to render the first image in.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            other
        """
        # embedded params
        defaults = {'figsize': (6 * len(images), 15),
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
        titles_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])

        grid = (1, len(images))
        fig, ax = plt.subplots(*grid, figsize=updated['figsize'])

        # plot image
        ax[0].imshow(images[0].T, **render_kwargs)
        for i in range(1, len(images)):
            ax[i].imshow(images[i].T, **render_kwargs) # grey colorschemes are embedded here
                                                  # might be more beautiful if red-color is in here
                                                  # if so, param for colorscheme is to be added
            ax[i].set_xlabel(**xaxis_kwargs)
            ax[i].set_ylabel(**yaxis_kwargs)
            ax[i].set_title(**dict(titles_kwargs, **{'label': titles_kwargs['label'][i]}))

        fig.suptitle(y=1.1, **label_kwargs)

        return fig

    def histogram(self, image, **kwargs):
        """ Plot histogram using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            2d-image for plotting.
        kwargs : dict
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            bins : int
                the number of bins to use.
            other
        """
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

        return plt
