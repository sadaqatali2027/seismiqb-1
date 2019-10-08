[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.12-orange.svg)](https://tensorflow.org)
[![Run Status](https://api.shippable.com/projects/5d5fbdc7d9f40a0006391187/badge?branch=master)](https://app.shippable.com/github/gazprom-neft/seismiqb)

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
git clone --recursive https://github.com/gazprom-neft/seismiqb.git
```

## Turorials

### [Cube-preprocessing](tutorials/2.%20Batch.ipynb)
Seismic cube preprocessing: `load_cubes`, `create_masks`, `scale`, `cutout_2d`, `rotate` and others.

### [Horizon segmentations](models/Horizons_detection.ipynb)
Solving a task of binary segmentation to detect seismic horizons.

### [Horizon extension](models/Horizons_extension.ipynb)
Extending picked horizons on the area of interest given marked horizons on a couple of `ilines`/`xlines`.

### [Interlayers segmentation](models/Segmenting_interlayers.ipynb)
Performing multiclass segmentation.


## Citing seismiqb

Please cite `seismicqb` in your publications if it helps your research.

    Khudorozhkov R., Koryagin A., Tsimfer S., Mylzenova D. Seismiqb library for seismic interpretation with deep learning. 2019.

```
@misc{seismiqb_2019,
  author       = {R. Khudorozhkov and A. Koryagin and S. Tsimfer and D. Mylzenova},
  title        = {Seismiqb library for seismic interpretation with deep learning},
  year         = 2019
}
```
