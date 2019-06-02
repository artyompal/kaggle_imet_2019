#!/usr/bin/python3.6

import os
import pickle
import re
import sys
import yaml

from glob import glob
from collections import OrderedDict
from typing import Any, List

import numpy as np
import pandas as pd

from scipy import optimize
from scipy.stats import describe
from tqdm import tqdm
from sklearn.metrics import fbeta_score

from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'
NUM_ATTEMPTS = 100
NUM_FOLDS = 5
NUM_CLASSES = 1103
YAML_DIR = '../yml'

def parse_labels(s: str) -> np.array:
    res = np.zeros(NUM_CLASSES)
    res[list(map(int, s.split()))] = 1
    return res

if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} predict1.npy ...')
        sys.exit()

    level2_fold = 0

    model_dir = f'../lightgbm/fold_{level2_fold}'
    os.makedirs(model_dir, exist_ok=True)

    # load data
    fold_num = np.load('folds.npy')
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')
    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))

    # build dataset
    all_predicts_list, all_thresholds = [], []
    predicts = sorted(sys.argv[1:])

    for filename in predicts:
        assert 'level1_train_' in filename
        m = re.match(r'(.*)_f(\d)_e\d+.*\.npy', filename)
        assert m
        model_path = m.group(1)

        predict = np.zeros((train_df.shape[0], NUM_CLASSES))

        for fold in range(NUM_FOLDS):
            filenames = glob(f'{model_path}_f{fold}_*.npy')
            if len(filenames) != 1:
                dprint(filenames)
                assert False # the model must be unique in this fold

            filename = filenames[0]
            print('reading', filename)

            # load data
            data = np.load(filename)

            # # read threshold
            # filename = os.path.basename(filename)
            # assert filename.startswith('level1_train_') and filename.endswith('.npy')
            #
            # with open(os.path.join(YAML_DIR, filename[13:-4] + '.yml')) as f:
            #     threshold = yaml.load(f, Loader=yaml.SafeLoader)['threshold']
            #     all_thresholds.append(threshold)
            #     data = data + threshold
            #
            # if np.min(data) < 0 or np.max(data) > 1:
            #     print('invalid range of data:', describe(data))

            predict[fold_num == fold] = data

        all_predicts_list.append(predict)

    level1_predicts = np.dstack(all_predicts_list)
    dprint(level1_predicts.shape)
    dprint(all_labels.shape)

    # gold_threshold = np.mean(all_thresholds)
    ground_truth = all_labels
    lines = []

    with open('../level2_linear_model.txt') as f:
        for line in f:
            if '[' in line:
                lines.append(line)
            else:
                lines[-1] += line

    assert len(lines) == NUM_CLASSES
    num_predicts = level1_predicts.shape[2]
    weights = []

    for class_, line in enumerate(tqdm(lines)):
        m = re.match(r'class=\d+ weights=\[((.|\n)*)\] f2=.+', line)
        assert m

        w = np.array(list(map(float, m.group(1).split())))
        assert w.size == num_predicts + 1
        weights.append(w)

    assert len(weights) == NUM_CLASSES
    level2_predicts = np.zeros((level1_predicts.shape[0], NUM_CLASSES))

    for sample in tqdm(range(level1_predicts.shape[0])):
        for class_ in range(NUM_CLASSES):
            x = level1_predicts[sample, class_]
            w = weights[class_]

            level2_predicts[sample, class_] = np.dot(w[:-1], x) + w[-1]
            # level2_predicts[sample, class_] += w[-1]

    dprint(describe(level2_predicts.flatten()))
    f2 = fbeta_score(ground_truth, level2_predicts > 0, beta=2,
                     average='samples')
    dprint(f2)
