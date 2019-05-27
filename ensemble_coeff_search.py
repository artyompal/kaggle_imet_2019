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
    level1_test_predicts: List[List[np.array]] = []

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

        level1_fnames, level1_train, level1_test = [], [], []
        for fold in range(NUM_FOLDS):
            filenames = glob(f'{model_path}_f{fold}_*.npy')
            assert len(filenames) == 1 # the model must be unique in this fold
            filename = filenames[0]

            print('found', filename)
            level1_fnames.append(filename)
            level1_train.append(np.load(filename))
            level1_test.append(np.load(filename.replace('level1_train_', 'level1_test_')))

        level1_filenames.append(level1_fnames)
        level1_train_predicts.append(level1_train)
        level1_test_predicts.append(level1_test)

    # search for the best blend weights
    best_weights = np.ones(len(level1_train_predicts))
    best_score = 0.0

    for _ in tqdm(range(NUM_ATTEMPTS)):
        # print('-' * 50)
        weights = np.random.rand(len(level1_train_predicts))
        weights /= sum(weights)
        all_predicts = np.zeros_like(all_labels)

        for lvl1_predicts, w in zip(level1_train_predicts, weights):
            model_predict = np.zeros_like(all_labels)

            for fold, lvl1_pred in enumerate(lvl1_predicts):
                predict = lvl1_pred * w
                model_predict[fold_num == fold] = predict

            all_predicts += model_predict

        score = F_score(all_predicts, all_labels, beta=2, threshold=0)

        if score > best_score:
            best_score, best_weights = score, weights
            print('best_score', best_score, 'weights', weights)

    # generate an ensemble description file
    ensemble = []

    for model, weight in zip(level1_filenames, best_weights):
        ensemble.append({'predicts': model, 'weight': weight.item()})

    filename = f'ens_{ensemble_name}_val_{best_score:.04f}.yml'
    print('saving weights to', filename)

    with open(filename, 'w') as f:
        yaml.dump(ensemble, f)
