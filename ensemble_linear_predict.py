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
ADD_THRESHOLD = True


def parse_labels(s: str) -> np.array:
    res = np.zeros(NUM_CLASSES)
    res[list(map(int, s.split()))] = 1
    return res

def load_weights(weights_file: str) -> List[np.ndarray]:
    lines = []

    with open(weights_file) as f:
        for line in f:
            if '[' in line:
                lines.append(line)
            else:
                lines[-1] += line

    assert len(lines) == NUM_CLASSES
    num_predicts = level1_predicts.shape[2]
    weights = []

    for class_, line in enumerate(lines):
        m = re.match(r'class=\d+ weights=\[((.|\n)*)\] f2=.+', line)
        assert m

        w = np.array(list(map(float, m.group(1).split())))
        assert w.size == num_predicts + 1
        weights.append(w)

    return weights

if __name__ == '__main__':
    np.set_printoptions(linewidth=120)
    if len(sys.argv) < 5:
        print(f'usage: {sys.argv[0]} result.npy coeffs.txt predict1.npy ...')
        sys.exit()

    all_predicts_list, all_thresholds = [], []
    predicts = sorted(sys.argv[3:])
    test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')

    for filename in predicts:
        assert 'level1_test_' in filename and '_f0_' in filename
        m = re.match(r'(.*)_f(\d)_e\d+.*\.npy', filename)
        assert m
        model_path = m.group(1)

        fold_predicts = []

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
                assert filename.startswith('level1_test_') and filename.endswith('.npy')

                with open(os.path.join(THRESHOLDS_PATH, filename[12:-4] + '.yml')) as f:
                    threshold = yaml.load(f, Loader=yaml.SafeLoader)['threshold']
                    all_thresholds.append(threshold)
                    data = data + threshold

            fold_predicts.append(data)

        predict = np.mean(np.dstack(fold_predicts), axis=2)
        all_predicts_list.append(predict)

    level1_predicts = np.dstack(all_predicts_list)
    dprint(level1_predicts.shape)

    weights = load_weights(sys.argv[2])
    assert len(weights) == NUM_CLASSES
    level2_predicts = np.zeros((level1_predicts.shape[0], NUM_CLASSES))

    for sample in tqdm(range(level1_predicts.shape[0]), disable=IN_KERNEL):
        for class_ in range(NUM_CLASSES):
            x = level1_predicts[sample, class_]
            w = weights[class_]

            level2_predicts[sample, class_] = np.dot(w[:-1], x) + w[-1]

    gold_threshold = np.mean(all_thresholds) if ADD_THRESHOLD else 0
    level2_predicts -= gold_threshold
    
    np.save(sys.argv[1], level2_predicts)
