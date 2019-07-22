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
from seismiqb import read_point_cloud, make_labels_dict



def matcher(horizont_1, horizont_2, eps=400):
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
    mean_1 = pd.read_csv(horizont_1, names=['ilines', 'xlines', 'heights'],
                         sep='\s+', usecols=['heights'])['heights'].mean()

    mean_2 = pd.read_csv(horizont_2, names=['ilines', 'xlines', 'heights'],
                         sep='\s+', usecols=['heights'])['heights'].mean()
    return np.abs((mean_1 - mean_2)) < eps


def compare(horizont_1, horizont_2, printer):
    """ Compare two horizonts by computing multiple simple metrics.

    Parameters
    ----------
    horizont_1, horizont_2 : str
        Path to horizont. Each line in the file must be in (iline, xline, height) format.

    Returns
    -------
    dict
        Computed metrics.
    """
    point_cloud_1 = read_point_cloud(horizont_1)
    labels_1 = make_labels_dict(point_cloud_1)

    point_cloud_2 = read_point_cloud(horizont_2)
    labels_2 = make_labels_dict(point_cloud_2)

    differences = []
    not_present_1, not_present_2 = 0, 0
    vals_1, vals_2 = [], []

    for key, val_1 in labels_1.items():
        if labels_2.get(key) is not None:
            val_2 = labels_2.get(key)
            diff = abs(val_2[0] - val_1[0])
            differences.append(diff)

            vals_1.append(val_1)
            vals_2.append(val_2)
        else:
            not_present_1 += 1

    for key, val_2 in labels_2.items():
        if labels_1.get(key) is None:
            not_present_2 += 1


    printer('First horizont:  {}'.format('/'.join(horizont_1.split('/')[-3:])))
    printer('Second horizont: {}'.format('/'.join(horizont_2.split('/')[-3:])))

    printer('Mean value/std of error:                  {:8.7} / {:8.7}'.format(np.mean(differences),
                                                                               np.std(differences)))
    printer('First horizont length:                    {}'.format(len(labels_1)))
    printer('Second horizont length:                   {}'.format(len(labels_2)))

    printer('Number in 5 ms window:                    {}'.format(np.sum(np.array(differences) <= 5)))
    printer('Rate in 5 ms window:                      {:8.7}'.format(np.sum(np.array(differences) <= 5) / len(differences)))

    printer('Average height of FIRST horizont:         {:8.7}'.format(np.mean(vals_1)))
    printer('Average height of SECOND horizont:        {:8.7}'.format(np.mean(vals_2)))

    printer('In the FIRST, but not in the SECOND:      {}'.format(not_present_1))
    printer('In the SECOND, but not in the FIRST:      {}'.format(not_present_2))
    printer('\n\n')



def main(dir_1, dir_2, printer=None):
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

    for horizont_1, horizont_2 in tqdm(cross):

        if not matcher(horizont_1, horizont_2):
            printer('Horizonts {} \n          {} \nwere REJECTED\n'.format(horizont_1, horizont_2))
            continue

        info = compare(horizont_1, horizont_2, printer)


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
