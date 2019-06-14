"""Test"""
#pylint: disable=import-error, wrong-import-position
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from seismiqb import SeismicCubeset, read_point_cloud, make_labels_dict
from seismiqb.batchflow import Pipeline, FilesIndex, B, V, L, D
from seismiqb.batchflow.models.tf import TFModel

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Script-args
FIRST_HORIZONT = '/notebooks/SEISMIC_DATA/CUBE_3/LINE_HORIZONTS/prb_G_anon.txt'
SECOND_HORIZONT = '/notebooks/SEISMIC_DATA/CUBE_3/PREDICTED_HORIZONTS/Hor_10-2731_0-2550_500-600_EACH_100_SAME_CUBE_1+VAL_1.txt'


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
    
    differences, not_present = [], 0
    vals_1, vals_2 = [], []

    for key, val_1 in labels_1.items():
        if labels_2.get(key) is not None:
            val_2 = labels_2.get(key)
            diff = abs(val_2[0] - val_1[0])
            differences.append(diff)
            
            vals_1.append(val_1)
            vals_2.append(val_2)
        else:
            not_present += 1
    
    info = {'mean_error': np.mean(differences),
            'std_error':  np.std(differences),
            'len_1': len(labels_1),
            'len_2': len(labels_2),
            'in_window':  sum(np.array(differences) <= 5),
            'rate_in_window': sum(np.array(differences) <= 5) / len(differences),
            'mean_1': np.mean(vals_1),
            'mean_2': np.mean(vals_2),
            'not_present': not_present,
            }
    
    return info


def main():
    info = compare(FIRST_HORIZONT, SECOND_HORIZONT)

    print('Mean value/std of error:                  {:8.7} / {:8.7}'.format(info['mean_error'], info['std_error']))
    print('First horizont length:                    {}'.format(info['len_1']))
    print('Second horizont length:                   {}'.format(info['len_2']))
    
    print('Number in 5 ms window:                    {}'.format(info['in_window']))
    print('Rate in 5 ms window:                      {:8.7}'.format(info['rate_in_window']))

    print('Average height of FIRST horizont:         {:8.7}'.format(info['mean_1']))
    print('Average height of SECOND horizont:        {:8.7}'.format(info['mean_2']))
    print('\nNumber of values that were present in the FIRST, but not in the SECOND: {}'.format(info['not_present']))
    
if __name__ == '__main__':
    main()
    