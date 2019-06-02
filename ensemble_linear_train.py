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
THRESHOLDS_PATH = '../yml/' if not IN_KERNEL else '../input/imet-yaml/yml/'
ADD_THRESHOLD = False


def parse_labels(s: str) -> np.array:
    res = np.zeros(NUM_CLASSES)
    res[list(map(int, s.split()))] = 1
    return res

if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} result.txt predict1.npy ...')
        sys.exit()

    level2_fold = 0

    # load data
    fold_num = np.load('folds.npy')
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')
    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))

    # build dataset
    all_predicts_list, all_thresholds = [], []
    predicts = sorted(sys.argv[2:])

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

            if ADD_THRESHOLD:
                # read threshold
                filename = os.path.basename(filename)
                assert filename.startswith('level1_train_') and filename.endswith('.npy')

                with open(os.path.join(THRESHOLDS_PATH, filename[13:-4] + '.yml')) as f:
                    threshold = yaml.load(f, Loader=yaml.SafeLoader)['threshold']
                    all_thresholds.append(threshold)
                    data = data + threshold

                if np.min(data) < 0 or np.max(data) > 1:
                    print('invalid range of data:', describe(data))

            predict[fold_num == fold] = data

        all_predicts_list.append(predict)

    all_predicts = np.dstack(all_predicts_list)
    dprint(all_predicts.shape)
    dprint(all_labels.shape)

    gold_threshold = np.mean(all_thresholds) if ADD_THRESHOLD else 0

    for class_ in tqdm(range(NUM_CLASSES)):
        x_train = all_predicts[:, class_]
        y_train = all_labels[:, class_]

        dims = x_train.shape[1]

        def loss_function(weights: np.ndarray) -> float:
            y_pred = np.matmul(x_train, weights[:-1])
            y_pred += weights[-1]

            y_pred = (y_pred > gold_threshold).astype(int)

            if np.sum(y_pred) == 0:
                res = 0
            else:
                res = fbeta_score(y_train, y_pred, beta=2)

            return -res

        weights = optimize.minimize(loss_function,
                                    [1 / dims] * dims + [0],
                                    method= 'Nelder-Mead', #'SLSQP',
                                    # constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                    # bounds=[(0.0, 1.0)] * len(level1_train_predicts),
                                    options = {'ftol':1e-10},
                                    )['x']

        best_score = -loss_function(weights)
        # print('class', class_, 'weights', weights, 'f2', best_score)

        with open(sys.argv[1], 'a') as f:
            f.write(f'class={class_} weights={weights} f2={best_score}\n')
