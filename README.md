# Seismiqb

`seismiqb` is a framework for deep learning research on 3d-cubes of seismic data. It allows to

* `sample` and `load` crops of `SEG-Y` cubes for training neural networks
* convert `SEG-Y` cubes to `HDF5`-format for even faster `load`
* `create_masks` of different types from horizon labels for segmenting horizons, facies and other seismic bodies
* build augmentation pipelines using custom augmentations for seismic data as well as `rotate`, `noise` and `elastic_transform`
* segment horizons and interlayers using [`UNet`](https://arxiv.org/abs/1505.04597) and [`Tiramisu`](https://arxiv.org/abs/1611.09326)
* extend horizons from a couple of seismic `ilines` in spirit of classic autocorrelation tools but with deep learning
* convert predicted masks into horizons for convenient validation by geophysicists


## Installation

```
git clone -- recursive https://github.com/analysiscenter/seismiqb.git
```

## Turorials

### [Cube-preprocessing](https://github.com/analysiscenter/seismiqb/blob/master/tutorials/2.%20Batch.ipynb)
Seismic cube preprocessing: `load_cubes`, `create_masks`, `scale`, `cutout_2d`, `rotate` and others.

### [Horizon segmentations](https://github.com/analysiscenter/seismiqb/blob/master/models/Horizons_detection.ipynb)
Solving a task of binary segmentation to detect seismic horizons.

### [Horizon extension](https://github.com/analysiscenter/seismiqb/blob/master/models/Horizons_extension.ipynb)
Extending picked horizons on the area of interest given marked horizons on a couple of `ilines`/`xlines`.

### [Interlayers segmentation](https://github.com/analysiscenter/seismiqb/blob/master/models/Segmenting_interlayers.ipynb)
Performing multiclass segmentation.
