#!/usr/bin/python3.6

import os
import re
import sys
import yaml

from glob import glob
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd

from tqdm import tqdm
from metrics import F_score
from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'
NUM_FOLDS = 5
NUM_CLASSES = 1103

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} ensemble.yml')
        sys.exit()

    source_file = sys.argv[1]
    result_name = os.path.splitext(os.path.basename(source_file))[0] + '_oof.npy'

    fold_num = np.load('folds.npy')
    train_df = pd.read_csv('../input/train.csv')

    with open(source_file) as f:
        ensemble = yaml.load(f, Loader=yaml.SafeLoader)

    result = np.zeros((train_df.shape[0], NUM_CLASSES))

    for predicts in tqdm(ensemble):
        weight = predicts['weight']
        assert len(predicts['predicts']) == NUM_FOLDS

        for fold, pred in enumerate(predicts['predicts']):
            result[fold_num == fold] += np.load(pred) * weight

    result /= len(ensemble)

    dprint(result.shape)
    dprint(result)

    np.save(result_name, result)
