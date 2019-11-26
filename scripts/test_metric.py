""" Metric evaluation testing. """
import os
import sys
import warnings
import argparse
from glob import glob

import segyio
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd

sys.path.append('..')
from seismiqb import SeismicCubeset
from seismiqb.batchflow import FilesIndex



def main(path_data, path_horizon, path_save):
    """ Metric evaluation. """
    if isinstance(path_save, str):
        original_umask = os.umask(0)
        os.makedirs(path_save, exist_ok=True)
        os.umask(original_umask)
        path_save = [os.path.join(path_save, item.split('/')[-1])
                     for item in glob(path_horizon)]

    dsi = FilesIndex(path=[path_data], no_ext=True)
    ds = SeismicCubeset(dsi)
    ds = ds.load(path_horizon, filter_zeros=True)
    geom = ds.geometries[ds.indices[0]]

    @njit
    def _convert(array):
        shape = array.shape
        res = np.zeros((shape[0]*shape[1], 3), dtype=np.float64)
        counter = 0

        for i in range(shape[0]):
            for j in range(shape[1]):
                res[counter, 0] = i
                res[counter, 1] = j
                res[counter, 2] = array[i, j]
                counter += 1
        return res

    for i, path in enumerate(geom.horizon_list):
        corrs = ds.compute_horizon_corrs(idx=0, horizon_idx=i, _no_plot=True, _return=True)
        converted = _convert(corrs)
        converted[:, 0] += geom.ilines_offset
        converted[:, 1] += geom.xlines_offset
        df = pd.DataFrame(converted, columns=['iline', 'xline', 'value'])
        df.sort_values(['iline', 'xline'], inplace=True)
        df.to_csv(path_save[i], sep=' ', columns=['iline', 'xline', 'value'],
                  index=False, header=False)
        print('Metric evaluation for {} is done.\nResults are saved to {}'.format(path, path_save[i]))
    return converted


if __name__ == '__main__':
    # Fetch paths from args
    parser = argparse.ArgumentParser(description="Predict horizons on a part of seismic-cube.")
    parser.add_argument("--path_cube", type=str, default='/')
    parser.add_argument("--path_horizon", type=str)
    parser.add_argument("--path_save", type=str)
    args = parser.parse_args()

    main(args.path_cube, args.path_horizon, args.path_save)
