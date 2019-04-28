#!/usr/bin/python3.6

import os, pickle, sys
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt
from debug import dprint


directory = sys.argv[1]
cache_path = f'../cache/sizes_{directory}.pkl'
os.makedirs(os.path.basename(cache_path), exist_ok=True)

if os.path.exists(cache_path):
    sizes = pickle.load(open(cache_path, 'rb'))
else:
    sizes = [np.array(Image.open(path)).shape for path in tqdm(glob(f'../input/{directory}/*.png'))]
    with open(cache_path, 'wb') as f:
        pickle.dump(sizes, f)

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
ratios = [s[0] / s[1] for s in sizes]

dprint(describe(widths))
dprint(describe(heights))

for sz in [288, 320, 352, 384]:
    print('num images greater than', sz, 'is', len([s for s in sizes if s[0] > sz and s[1] > sz]))

# plt.plot(ratios); plt.show()
# plt.plot(widths); plt.show()
# plt.plot(heights); plt.show()
