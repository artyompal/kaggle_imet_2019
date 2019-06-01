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
import lightgbm as lgb

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

            # read threshold
            filename = os.path.basename(filename)
            assert filename.startswith('level1_train_') and filename.endswith('.npy')

            with open(os.path.join(YAML_DIR, filename[13:-4] + '.yml')) as f:
                threshold = yaml.load(f, Loader=yaml.SafeLoader)['threshold']
                all_thresholds.append(threshold)
                data = data + threshold

            if np.min(data) < 0 or np.max(data) > 1:
                print('invalid range of data:', describe(data))

            predict[fold_num == fold] = data

        all_predicts_list.append(predict)

    level1_predicts = np.dstack(all_predicts_list)
    dprint(level1_predicts.shape)
    dprint(all_labels.shape)

    gold_threshold = np.mean(all_thresholds)
    level2_predicts = np.zeros((level1_predicts.shape[1], NUM_CLASSES))

    # for class_ in tqdm(range(NUM_CLASSES)):
    for filename in tqdm(sorted(glob(f'{model_dir}/lightgbm_*.pkl'))):
        m = re.match(r'lightgbm_f0_c(.*)_([.0-9]+)\.pth', os.path.basename(filename))
        if not m:
            raise RuntimeError('could not parse model name')

        class_ = int(m.group(1))
        score = float(m.group(2))

        x_train = level1_predicts[fold_num != level2_fold][:, class_]
        y_train = all_labels[fold_num != level2_fold][:, class_]
        x_val = level1_predicts[fold_num == level2_fold][:, class_]
        y_val = all_labels[fold_num == level2_fold][:, class_]

        lgtrain = lgb.Dataset(x_train, y_train)
        lgvalid = lgb.Dataset(x_val, y_val)

        with open(filename, 'rb') as model_file:
            lgb_clf = pickle.load(model_file)

        val_pred = lgb_clf.predict(x_val)
        level2_predicts[:, class_] = val_pred

    f2 = fbeta_score(y_val, val_pred > gold_threshold, beta=2)
    dprint(f2)
