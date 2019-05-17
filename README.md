# Seismiqb

`seismiqb` is a framework for deep learning research on 3d-cubes of seismic data. It allows to

* `sample` and `load` crops of `SEG-Y` cubes for training neural networks
* convert `SEG-Y` cubes to `HDF5`-format for even faster `load`
* `create_masks` of different types for segmenting horizons, facies and other seismic bodies


## Installation

```
git clone -- recursive https://github.com/analysiscenter/seismiqb.git
```

## Turorials

### Cube-preprocessing
Performing basic preprocessing on seismic cubes. You can find the tutorial [here](https://github.com/analysiscenter/seismiqb/blob/tutorials/tutorials/2.%20Batch.ipynb).

### Horizon segmentation
Solving a task of binary segmentation. You can find the model-notebook [here](https://github.com/analysiscenter/seismiqb/blob/tutorials/tutorials/3.%20Horizonts_model.ipynb).

### Horizon extension
Solving a task of binary segmentation. You can find the tutorial [here](https://github.com/analysiscenter/seismiqb/blob/tutorials/tutorials/Horizon%20Extension.ipynb).

### Interlayers segmentation
Performing multiclass segmentation. You can find the notebook [here](https://github.com/analysiscenter/seismiqb/blob/tutorials/tutorials/4.%20Segmenting_interlayers.ipynb).
