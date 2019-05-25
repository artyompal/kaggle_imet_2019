#!/usr/bin/python3.6
''' Visualizes some predictions. '''

import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

from debug import dprint


NUM_CLASSES = 1103
COLUMNS     = 1
ROWS        = 3

NUM_SAMPLES_PER_CLASS   = COLUMNS * ROWS
NUM_SAMPLES_TO_SHOW     = 20


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} predicts.npy')
        sys.exit()

    # load data
    train_df = pd.read_csv('../input/train.csv')
    labels_table = pd.read_csv('../input/labels.csv').attribute_name.values
    predicts = np.load(sys.argv[1])
    assert len(train_df) == len(predicts)

    def parse_labels(s: str) -> np.array:
        res = np.zeros(NUM_CLASSES)
        res[list(map(int, s.split()))] = 1
        return res

    all_labels = np.vstack(list(map(parse_labels, train_df.attribute_ids)))
    dprint(all_labels.shape)

    # plt.hist(np.amin(predicts, axis=1), bins=20)
    # plt.show()
    # plt.hist(np.amax(predicts, axis=1), bins=20)
    # plt.show()

    # analyze mistakes
    rounded_predicts = (predicts > 0).astype(int)
    ground_truths = (all_labels > 0.5).astype(int)

    dprint(rounded_predicts)
    dprint(ground_truths.shape)
    dprint(ground_truths)

    negatives = ground_truths != rounded_predicts
    masked_predicts = np.copy(predicts)
    masked_predicts[negatives == 0] = 0

    confs = np.amax(masked_predicts, axis=1)
    dprint(confs.shape)
    most_confident_mistakes = np.argsort(-confs)
    dprint(most_confident_mistakes.shape)
    dprint(most_confident_mistakes)
    dprint(confs[most_confident_mistakes])

    # plt.plot(confs[most_confident_mistakes])
    # plt.show()

    # visualize mistakes
    for j, sample in enumerate(most_confident_mistakes[:NUM_SAMPLES_TO_SHOW]):
        print('-' * 80)
        print(f'index {j} sample {sample} image {train_df.id[sample]}')

        print('predicts', [i for i, p in enumerate(predicts[sample]) if p > 0])
        print(f'ground truth [{train_df.attribute_ids[sample]}]')

        conf = predicts[sample]
        predict_str = " ".join(f'{labels_table[i]} ({c:.02f})'
                               for i, c in enumerate(conf) if c > 0)
        labels = " ".join(labels_table[i] for i, L in enumerate(ground_truths[sample]) if L)
        dprint(predict_str)
        dprint(labels)

        fig = plt.figure(figsize=(12, 12))
        plt.suptitle(f'predict:     {predict_str}\nshould be:     {labels}')

        img = plt.imread(f'../input/train/{train_df.id[sample]}.png')
        plt.imshow(img)
        plt.show()
