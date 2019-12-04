""" Compare each horizont in the first directory with each horizont in the second. """
#pylint: disable=import-error, wrong-import-position
import os
import sys
import argparse
import json
import logging
from glob import glob

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from seismiqb import read_point_cloud, make_labels_dict, compare_horizons



def matcher(horizont_1, horizont_2):
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


def compare(horizont_1, horizont_2, printer):
    """ Compare two horizonts by computing multiple simple metrics.
    All of the results are logged via passed `printer`.

    Parameters
    ----------
    horizont_1, horizont_2 : str
        Path to horizont. Each line in the file must be in (iline, xline, height) format.
    printer : callable
        Function to log information with.
    """
    point_cloud_1 = read_point_cloud(horizont_1)
    labels_1 = make_labels_dict(point_cloud_1)

    point_cloud_2 = read_point_cloud(horizont_2)
    labels_2 = make_labels_dict(point_cloud_2)

    printer('First horizont:  {}'.format('/'.join(horizont_1.split('/')[-3:])))
    printer('Second horizont: {}'.format('/'.join(horizont_2.split('/')[-3:])))
    window_metric, area_metric = compare_horizons(labels_1, labels_2, printer=printer, plot=False,
                                                  sample_rate=1, offset=1)
    return window_metric, area_metric


def main(dir_1, dir_2, printer=None, eps=200):
    """ Compare each pair of horizonts in passed lists.

    Parameters
    ----------
    dir_1, dir_2 : str
        Path to directories with horizonts to compare.

    printer : callable
        Function to print with.
    """
    list_1 = glob(dir_1)
    list_2 = glob(dir_2)
    cross = [(item_1, item_2) for item_1 in list_1 for item_2 in list_2]

    metrics = []
    for horizont_1, horizont_2 in tqdm(cross):
        error = matcher(horizont_1, horizont_2)
        if not (error < eps):
            printer('Horizons {} \n           {} \
                     \nwere REJECTED with error of: {}\n'.format(horizont_1, horizont_2, error))
            continue

        window_metric, area_metric = compare(horizont_1, horizont_2, printer)
        metrics.append((window_metric, area_metric))

    metrics = np.asarray(metrics).reshape((-1, 2))
    metric = metrics[:, 0].dot(metrics[:, 1])
    printer(metric)
    return metric


if __name__ == '__main__':
    # Get arguments from passed json
    parser = argparse.ArgumentParser(description="Compare two lists of horizonts.")
    parser.add_argument("--config_path", type=str, default="./configs/compare.json")
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = json.load(file)
        args = [config.get(key) for key in ["dir_1", "dir_2"]]

    # Logging to either stdout or file
    if config.get("print"):
        printer = print
    else:
        path_log = config.get("path_log") or os.path.join(os.getcwd(), "logs/compare.log")
        handler = logging.FileHandler(path_log, mode='w')
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger('compare_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        printer = logger.info

    # Compare each pair of horizonts in two directories
    main(*args, printer=printer)
