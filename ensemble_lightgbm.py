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

from scipy import optimize
from tqdm import tqdm
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

    C = 125
    for class_ in range(C, C+1): # tqdm(range(NUM_CLASSES)):
        # print('-' * 80)
        # dprint(class_)

        x_train = all_predicts[fold_num != 0][:, class_]
        y_train = all_labels[fold_num != 0][:, class_]
        x_val = all_predicts[fold_num == 0][:, class_]
        y_val = all_labels[fold_num == 0][:, class_]

        # training stage
        lgtrain = lgb.Dataset(x_train, y_train)
        lgvalid = lgb.Dataset(x_val, y_val)

        lgbm_params =  {
            'task': 'train',
            # 'eval_metric' : f2_score,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'xentropy',
            'num_leaves': 25,
            # 'max_bin': 128,
            'min_data_in_leaf': 10,
            # 'max_depth': 20,

            # 'feature_fraction': 0.5,
            # 'bagging_fraction': 0.75,
            # 'bagging_freq': 2,
            'learning_rate': 0.003,
            'verbose': 1
        }
        print("LightGBM params:", lgbm_params)

        def f2_score(y_pred, data):
            y_true = data.get_label()
            y_pred = y_pred > 0.1
            # y_pred = y_pred > 0
            return 'f2', fbeta_score(y_true, y_pred, beta=2), True

        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=15000,
            # num_boost_round=2000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=100,
            verbose_eval=10,
            feval=f2_score
            )
