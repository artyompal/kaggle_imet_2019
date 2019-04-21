#!/usr/bin/python3.6

from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt
from debug import dprint


# sizes = [np.array(Image.open(path)).shape for path in tqdm(glob('../input/train/*.png'))]
sizes = [np.array(Image.open(path)).shape for path in tqdm(glob('../input/test/*.png'))]

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
ratios = [s[0] / s[1] for s in sizes]

dprint(describe(widths))
dprint(describe(heights))

plt.plot(ratios); plt.show()
plt.plot(widths); plt.show()
plt.plot(heights); plt.show()

