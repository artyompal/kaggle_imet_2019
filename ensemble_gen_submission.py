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
NUM_CLASSES = 1103

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} ensemble.yml')
        sys.exit()

    source_file = sys.argv[1]
    result_name = os.path.splitext(os.path.basename(source_file))[0] + '.csv' \
                  if not IN_KERNEL else 'submission.csv'

    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    result = np.zeros((sub.shape[0], NUM_CLASSES))

    with open(source_file) as f:
        ensemble = yaml.load(f, Loader=yaml.SafeLoader)

    for predicts in ensemble:
        weight = predicts['weight']

        for pred in predicts['predicts']:
            result += np.load(pred.replace('_train_', '_test_')) * weight

    dprint(result.shape)
    dprint(result)
    labels = [" ".join([str(i) for i, p in enumerate(pred) if p > 0])
              for pred in tqdm(result, disable=IN_KERNEL)]
    dprint(len(labels))
    print('labels')
    print(np.array(labels))

    sub['attribute_ids'] = labels
    sub.to_csv(result_name, index=False)
