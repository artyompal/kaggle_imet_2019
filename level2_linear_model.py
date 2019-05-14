#!/usr/bin/python3.6

import os
import re
import sys

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

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} result.csv predict1.npy ...')
        sys.exit()

    result_name, predicts = sys.argv[1], sys.argv[2:]
    num_folds = 5
    models: 'OrderedDict[str, List[str]]' = OrderedDict()

    # get a full list of predicts and a full list of models
    for predict in predicts:
        assert 'level1_train_' in predict
        m = re.match(r'(.*)_f(\d)_e\d+.*\.npy', predict)
        assert m
        model_path = m.group(1)

        folds = []
        for fold in range(num_folds):
            names = glob(f'{model_path}_f{fold}_*.npy')
            assert len(names) == 1
            folds.append(names[0])

        models[os.path.basename(model_path)] = folds

    print('models')
    for model_name in models.keys():
        dprint(model_name)
        print(np.array(models[model_name]))

    # load labels
    fold_num = np.load('folds.npy')
    train_df = pd.read_csv('../input/train.csv')
    num_classes = pd.read_csv('../input/labels.csv').shape[0]

    def parse_labels(s: str) -> np.array:
        res = np.zeros(num_classes)
        res[list(map(int, s.split()))] = 1
        return res

    labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))
    dprint(fold_num.shape)
    dprint(labels.shape)

    # search for the best blend weights
    NUM_ATTEMPTS = 1000
    best_weights = np.array([1.5, 1, 1.2])
    best_score = 0.0

    for _ in range(NUM_ATTEMPTS):
        print('-' * 50)
        weights = np.random.rand(len(models))

        for predicts, w in zip(models.values(), best_weights):
            for fold, predict in enumerate(predicts):
                print(f'reading {predict}, weight={w}')
                if fold == 0:
                    result = np.load(predict) * w
                else:
                    result += np.load(predict) * w

        score = F_score(result, labels[fold_num == fold], beta=2, threshold=0)
        dprint(score)
        dprint(weights)

        if score > best_score:
            best_score, best_weights = score, weights

    # generate submission
    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    result = np.zeros((sub.shape[0], 1103))

    for predicts, w in zip(models.values(), best_weights):
        for predict in predicts:
            test_predict = predict.replace('level1_train_', 'level1_test_')
            print(f'reading {test_predict}, weight={w}')
            result += np.load(test_predict) * w

    dprint(result.shape)
    dprint(result)
    labels = [" ".join([str(i) for i, p in enumerate(pred) if p > 0])
              for pred in tqdm(result, disable=IN_KERNEL)]
    dprint(len(labels))
    print('labels')
    print(np.array(labels))

    sub['attribute_ids'] = labels
    weights_str = '_'.join(map(lambda w: f'{w:.3}', best_weights))
    result_name = os.path.splitext(result_name)[0]
    sub.to_csv(f'{result_name}_{weights_str}.csv', index=False)
