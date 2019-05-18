#!/usr/bin/python3.6
""" Shows images of some class. """

import os
import random
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

COLUMNS     = 4
ROWS        = 3
NUM_TESTS   = COLUMNS * ROWS

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} class')
        sys.exit()

    key = sys.argv[1]
    labels = pd.read_csv('../input/labels.csv').attribute_name.values

    train = pd.read_csv('../input/train.csv')
    train['classes'] = train.attribute_ids.apply(lambda s: s.split())

    all_images = train.id.loc[train.classes.apply(lambda s: key in s)]
    num_tests = min(NUM_TESTS, len(all_images))
    images = np.random.choice(all_images, size=num_tests, replace=False)

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'class {key}, label {labels[int(key)]}, total={len(all_images)}')

    for i in range(num_tests):
        subplot = fig.add_subplot(ROWS, COLUMNS, i + 1)
        subplot.set_title(images[i])

        img = plt.imread(f'../input/train/{images[i]}.png')
        plt.imshow(img)

    plt.show()
    fig.savefig(f'../debug/samples_{labels[int(key)]}_{random.randrange(1000)}.png')
    plt.close(fig)
