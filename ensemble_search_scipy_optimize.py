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

from scipy import optimize
# from tqdm import tqdm
from metrics import F_score
from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'
NUM_ATTEMPTS = 100
NUM_FOLDS = 5
NUM_CLASSES = 1103

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} <ensemble_name> predict1.npy ...')
        sys.exit()

    ensemble_name, predicts = sys.argv[1], sys.argv[2:]
    level1_filenames: List[List[str]] = []
    level1_train_predicts: List[List[np.array]] = []

    # load labels
    fold_num = np.load('folds.npy')
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')

    def parse_labels(s: str) -> np.array:
        res = np.zeros(NUM_CLASSES)
        res[list(map(int, s.split()))] = 1
        return res - 0.5    # we use zero threshold instead of 0.5

    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))
    dprint(fold_num.shape)
    dprint(all_labels.shape)

    # build a list of models, for every model build a list of predicts
    for predict in predicts:
        assert 'level1_train_' in predict
        m = re.match(r'(.*)_f(\d)_e\d+.*\.npy', predict)
        assert m
        model_path = m.group(1)

        level1_fnames, level1_train = [], []
        for fold in range(NUM_FOLDS):
            filenames = glob(f'{model_path}_f{fold}_*.npy')
            if len(filenames) != 1:
                dprint(filenames)
                assert False # the model must be unique in this fold

            filename = filenames[0]

            print('found', filename)
            level1_fnames.append(filename)
            level1_train.append(np.load(filename))

        level1_filenames.append(level1_fnames)
        level1_train_predicts.append(level1_train)

    # search for the best blend weights
    def loss_function(weights: np.ndarray) -> float:
        all_predicts = np.zeros_like(all_labels)

        for lvl1_predicts, w in zip(level1_train_predicts, weights):
            model_predict = np.zeros_like(all_labels)

            for fold, lvl1_pred in enumerate(lvl1_predicts):
                predict = lvl1_pred * w
                model_predict[fold_num == fold] = predict

            all_predicts += model_predict

        score = F_score(all_predicts, all_labels, beta=2, threshold=0)
        print('score', score, 'weights', weights)

        return -score

    weights = optimize.minimize(loss_function,
                                [1 / len(level1_train_predicts)] * len(level1_train_predicts),
                                # constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'Nelder-Mead', #'SLSQP',
                                bounds=[(0.0, 1.0)] * len(level1_train_predicts),
                                options = {'ftol':1e-10},
                                )['x']
    best_score = -loss_function(weights)

    # generate an ensemble description file
    ensemble = []

    for model, weight in zip(level1_filenames, weights):
        model_filenames = [os.path.basename(f) for f in model]
        ensemble.append({'predicts': model_filenames, 'weight': weight.item()})

    filename = f'{ensemble_name}_val_{best_score:.04f}.yml'
    print('saving weights to', filename)

    with open(filename, 'w') as f:
        yaml.dump(ensemble, f)
