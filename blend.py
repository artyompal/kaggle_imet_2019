#!/usr/bin/python3.6

import os, sys
from debug import dprint
import numpy as np, pandas as pd
from tqdm import tqdm

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
INPUT_PATH = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
        print(f'usage: {sys.argv[0]} result.csv submission1.npy weight1...')
        sys.exit()

    result_name = sys.argv[1]
    predicts = sys.argv[2::2]
    weights = np.array(sys.argv[3::2], dtype=float)
    weights /= np.sum(weights)
    dprint(predicts)
    dprint(weights)

    sub = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    result = np.zeros((sub.shape[0], 1103))

    for pred, w in zip(predicts, weights):
        print(f'reading {pred}, weight={w}')

        data = np.load(pred)
        result += data * w

    dprint(result.shape)
    dprint(result)
    labels = [" ".join([str(i) for i, p in enumerate(pred) if p > 0])
              for pred in tqdm(result, disable=IN_KERNEL)]
    dprint(len(labels))
    dprint(np.array(labels))

    sub['attribute_ids'] = labels
    sub.to_csv(result_name, index=False)
