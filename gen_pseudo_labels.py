#!/usr/bin/python3.6
''' Fixes labels in the train set with the most confident predictions. '''

import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import describe

from debug import dprint

NUM_CLASSES = 1103

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} <confidence_threshold> <predicts.npy>')
        sys.exit()

    # load data
    min_conf = float(sys.argv[1])
    train_df = pd.read_csv('../input/train.csv')
    all_predicts = np.load(sys.argv[2])
    assert len(train_df) == len(all_predicts)

    def parse_labels(s: str) -> np.array:
        res = np.zeros(NUM_CLASSES)
        res[list(map(int, s.split()))] = 1
        return res

    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))
    dprint(all_labels.shape)

    # get most confident predicts
    stats = []
    print('filtering labels')

    for i in tqdm(range(train_df.shape[0])):
        dataset_labels = all_labels[i]
        predicts = all_predicts[i]

        my_labels = dataset_labels.copy()
        my_labels[predicts > min_conf] = 1
        stats.append(any(dataset_labels != my_labels))

        all_labels[i] = my_labels

    print('fraction of changed labels', np.mean(stats))

    print('encoding labels')
    labels = [" ".join([str(i) for i, p in enumerate(pred) if p > 0])
              for pred in tqdm(all_labels)]
    dprint(len(labels))
    print('labels')
    print(np.array(labels))

    train_df['attribute_ids'] = labels
    train_df.to_csv(f'train_conf_{min_conf:01f}.csv', index=False)
