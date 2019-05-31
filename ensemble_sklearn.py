#!/usr/bin/python3.6

import os
import re
import sys
import yaml

from glob import glob
from collections import OrderedDict
from typing import Any, List

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import describe
from tqdm import tqdm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score

from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'
NUM_ATTEMPTS = 100
NUM_FOLDS = 5
NUM_CLASSES = 1103


def parse_labels(s: str) -> np.array:
    res = np.zeros(NUM_CLASSES)
    res[list(map(int, s.split()))] = 1
    return res

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} predict1.npy ...')
        sys.exit()

    # load data
    fold_num = np.load('folds.npy')
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')

    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))
    dprint(fold_num.shape)
    dprint(all_labels.shape)


    # build dataset
    all_predicts_list = []
    predicts = sys.argv[1:]

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

            data = np.load(filename)
            predict[fold_num == fold] = data

        all_predicts_list.append(predict)

    all_predicts = np.dstack(all_predicts_list)

    # FIXME: use real thresholds here
    all_predicts -= np.min(all_predicts, axis=1, keepdims=True)

    dprint(all_predicts.shape)
    dprint(all_labels.shape)

    for class_ in tqdm(range(NUM_CLASSES)):
        # print('-' * 80)
        # dprint(class_)

        x_train = all_predicts[fold_num != 0][:, class_]
        y_train = all_labels[fold_num != 0][:, class_]
        x_val = all_predicts[fold_num == 0][:, class_]
        y_val = all_labels[fold_num == 0][:, class_]

        # dprint(x_train.shape)
        # dprint(y_train.shape)
        # dprint(x_val.shape)
        # dprint(y_val.shape)
        #
        # dprint(describe(x_train))
        # dprint(describe(x_val))
        # dprint(describe(y_train))
        # dprint(describe(y_val))
        #
        # dprint(np.unique(y_val))


        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(x_train, y_train)
        y_pred = classif.predict(x_val)

        # FIXME: do I have to find the best threshold?
        y_pred = y_pred > 0.1

        if np.sum(y_pred) > 0:
            score = fbeta_score(y_val, y_pred, beta=2)
        else:
            score = 0

        print('class', class_, 'F2 score:', score)
