"""Test"""
#pylint: disable=import-error, wrong-import-position
import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

sys.path.append('..')
from seismiqb import SeismicCubeset, read_point_cloud, make_labels_dict
from seismiqb.batchflow import Pipeline, FilesIndex, B, V, L, D
from seismiqb.batchflow.models.tf import TFModel

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Script-args
LIST_1 = '/notebooks/SEISMIC_DATA/CUBE_3/LINE_HORIZONTS/prb_G_anon.txt'
LIST_2 = '/notebooks/SEISMIC_DATA/CUBE_3/PREDICTED_HORIZONTS/Hor_10-2731_0-2550_500-600_EACH_100_SAME_CUBE_1+VAL_1.txt'


def compare(horizont_1, horizont_2):
    print('\nComparing: ')
    print('           {}'.format(horizont_1))
    print('           {}'.format(horizont_2))

    point_cloud_1 = read_point_cloud(horizont_1)
    labels_1 = make_labels_dict(point_cloud_1)
    print('First horizont is processed..')

    point_cloud_2 = read_point_cloud(horizont_2)
    labels_2 = make_labels_dict(point_cloud_2)
    print('Second horizont is processed..')

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

    info = {'name_1': '/'.join(horizont_1.split('/')[-3:])
            'name_2': '/'.join(horizont_2.split('/')[-3:])
            'mean_error': np.mean(differences),
            'std_error':  np.std(differences),
            'len_1': len(labels_1),
            'len_2': len(labels_2),
            'in_window':  sum(np.array(differences) <= 5),
            'rate_in_window': sum(np.array(differences) <= 5) / len(differences),
            'mean_1': np.mean(vals_1),
            'mean_2': np.mean(vals_2),
            'not_present_1': not_present_1,
            'not_present_2': not_present_2,
            }

    return info


def main(list_1, list_2):
    list_1 = glob(list_1)
    list_2 = glob(list_2)
    cross = [(item_1, item_2) for item_1 in list_1 for item_2 in list_2]

    for horizont_1, horizont_2 in tqdm(cross):
        info = compare(horizont_1, horizont_2)

        print('First horizont:  {}'.format(info['name_1']))
        print('Second horizont: {}'.format(info['name_2']))

        print('Mean value/std of error:                  {:8.7} / {:8.7}'.format(info['mean_error'], info['std_error']))
        print('First horizont length:                    {}'.format(info['len_1']))
        print('Second horizont length:                   {}'.format(info['len_2']))

        print('Number in 5 ms window:                    {}'.format(info['in_window']))
        print('Rate in 5 ms window:                      {:8.7}'.format(info['rate_in_window']))

        print('Average height of FIRST horizont:         {:8.7}'.format(info['mean_1']))
        print('Average height of SECOND horizont:        {:8.7}'.format(info['mean_2']))

        print('In the FIRST, but not in the SECOND:      {}'.format(info['not_present_1']))
        print('In the SECOND, but not in the FIRST:      {}'.format(info['not_present_2']))
        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two lists of horizonts.")
    parser.add_argument("--config_path", type=str, default="./config_compare.json")
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = json.load(file)
        args = [config.get(key) for key in ["list_1", "list_2"]]
    main(*args)
