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
    def __init__(self, public_path, private_path):
        self.public_path = public_path
        self.private_path = private_path
        self.eps = 200

    def calc_metric(self, submission_path, public=True):
        extr_dir = os.path.join(*path.split('/')[:-1], '__' + str(time()))
        with zipfile.ZipFile(submission_path, 'r') as zip_ref:
            zip_ref.extractall(extr_dir)

        list_1 = glob(extr_dir + '/*')
        list_2 = glob(self.public_path if public else self.private_path)
        cross = [(item_1, item_2) for item_1 in list_1 for item_2 in list_2]

        metrics = []
        for horizon_1, horizon_2 in cross:
            error = self.matcher(horizon_1, horizon_2)
            if not (error < self.eps):
                continue

            window_metric, area_metric = self.compare(horizon_1, horizon_2)
            metrics.append((window_metric, area_metric))

        metrics = np.asarray(metrics).reshape((-1, 2))
        metric = metrics[:, 0].dot(metrics[:, 1])
        # print(metric)
        return metric


    def matcher(self, horizon_1, horizon_2):
        """ Fetch difference between horizons' hs.

        Parameters
        ----------
        horizon_1, horizon_2 : str
            Paths to the respective horizonts.
        """
        mean_1 = pd.read_csv(horizon_1, names=['iline', 'xline', 'height'],
                             sep='\s+', usecols=['height'])['height'].mean()

        mean_2 = pd.read_csv(horizon_2, names=['iline', 'xline', 'height'],
                             sep='\s+', usecols=['height'])['height'].mean()
        error = np.abs((mean_1 - mean_2))
        return error


    def compare(self, horizon_1, horizon_2):
        """ Compare two horizons by computing multiple simple metrics.
        All of the results are logged via passed `printer`.

        Parameters
        ----------
        horizon_1, horizon_2 : str
            Path to horizon. Each line in the file must be in (iline, xline, height) format.
        """
        point_cloud_1 = read_point_cloud(horizon_1)
        labels_1 = make_labels_dict(point_cloud_1)

        point_cloud_2 = read_point_cloud(horizon_2)
        labels_2 = make_labels_dict(point_cloud_2)

        # print('First horizont:  {}'.format('/'.join(horizont_1.split('/')[-3:])))
        # print('Second horizont: {}'.format('/'.join(horizont_2.split('/')[-3:])))
        # print('Window metric is {}, \narea_metric is {}'.format(window_metric, area_metric))
        window_metric, area_metric = compare_horizons(labels_1, labels_2, printer=None, plot=False,
                                                      sample_rate=1, offset=1)
        return window_metric, area_metric
