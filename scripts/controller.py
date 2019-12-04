""" Compare each horizont in the first directory with each horizont in the second. """
#pylint: disable=import-error, wrong-import-position
import os
import sys
import argparse
import json
import logging
import zipfile
from time import time
from glob import glob

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from seismiqb import read_point_cloud, make_labels_dict, compare_horizons



class HorizonDetectionController:
    """ Amazing docstring. """
    PATH_PUBLIC = '/home/tsimfer/SEISMIC_DATA/CUBE_2/BEST_HORIZONS/*'
    PATH_PRIVATE = '/home/tsimfer/SEISMIC_DATA/CUBE_2/BEST_HORIZONS/*'


    def __init__(self):
        pass

    def calc_metric(self, path, private=False, eps=200):
        extr_dir = os.path.join(*path.split('/')[:-1], '__'+str(time()))
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extr_dir)


        list_1 = glob(extr_dir + '/*')
        list_2 = glob(self.PATH_PUBLIC if not private else self.PATH_PRIVATE)
        cross = [(item_1, item_2) for item_1 in list_1 for item_2 in list_2]

        metrics = []
        for horizont_1, horizont_2 in cross:
            error = self.matcher(horizont_1, horizont_2)
            if not (error < eps):
                # rejected
                # print('Horizons {} \n           {} \
                #          \nwere REJECTED with error of: {}\n'.format(horizont_1, horizont_2, error))
                continue

            window_metric, area_metric = self.compare(horizont_1, horizont_2)
            metrics.append((window_metric, area_metric))

        metrics = np.asarray(metrics).reshape((-1, 2))
        metric = metrics[:, 0].dot(metrics[:, 1])
        print(metric)
        return metric


    def matcher(self, horizont_1, horizont_2):
        """ Check if horizonts are close to each other.

        Parameters
        ----------
        horizont_1, horizont_2 : str
            Paths to the respective horizonts.

        eps : int, float
            Distance threshold.

        Returns
            True, if horizonts are on average closer to each other than eps.
            False otherwise
        """
        mean_1 = pd.read_csv(horizont_1, names=['iline', 'xline', 'height'],
                             sep='\s+', usecols=['height'])['height'].mean()

        mean_2 = pd.read_csv(horizont_2, names=['iline', 'xline', 'height'],
                             sep='\s+', usecols=['height'])['height'].mean()
        error = np.abs((mean_1 - mean_2))
        return error


    def compare(self, horizont_1, horizont_2):
        """ Compare two horizonts by computing multiple simple metrics.
        All of the results are logged via passed `printer`.

        Parameters
        ----------
        horizont_1, horizont_2 : str
            Path to horizont. Each line in the file must be in (iline, xline, height) format.
        """
        point_cloud_1 = read_point_cloud(horizont_1)
        labels_1 = make_labels_dict(point_cloud_1)

        point_cloud_2 = read_point_cloud(horizont_2)
        labels_2 = make_labels_dict(point_cloud_2)

        # print('First horizont:  {}'.format('/'.join(horizont_1.split('/')[-3:])))
        # print('Second horizont: {}'.format('/'.join(horizont_2.split('/')[-3:])))
        # print('Window metric is {}, \narea_metric is {}'.format(window_metric, area_metric))
        window_metric, area_metric = compare_horizons(labels_1, labels_2, printer=None, plot=False,
                                                      sample_rate=1, offset=1)
        return window_metric, area_metric
