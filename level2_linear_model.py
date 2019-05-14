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

    # train_predicts = list(map(np.load, train_pred_fnames))

    # search for the best blend weights
    weights = np.array([1.5, 1, 1.2])
    weights /= sum(weights)

    # # get test predictions
    # test_predicts = list(map(lambda s: s.replace('level1_train_', 'level1_test_'),
    #                          train_pred_fnames))
    # print('test_predicts')
    # print(np.array(test_predicts))

    # generate submission
    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    result = np.zeros((sub.shape[0], 1103))

    for predicts, w in zip(models.values(), weights):
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
    weights_str = '_'.join(map(lambda w: f'{w:.3}', weights))
    result_name = os.path.splitext(result_name)[0]
    sub.to_csv(f'{result_name}_{weights_str}.csv', index=False)
