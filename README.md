# Seismiqb

`seismiqb` is a framework for deep learning research on 3d-cubes of seismic data. It allows to

* `sample` and `load` crops of `SEG-Y` cubes for training neural networks
* convert `SEG-Y` cubes to `HDF5`-format for even faster `load`
* `create_masks` of different types for segmenting horizons, facies and other seismic bodies


## Installation

```
git clone -- recursive https://github.com/analysiscenter/seismiqb.git
```

## Models

### Horizon segmentation
A task of binary segmentation. You can find the model-notebook here.

### Horizon extension
A task of binary segmentation. You can find the tutorial here.

### Interlayers segmentation
Multiclass segmentation. You can find the notebook here.
