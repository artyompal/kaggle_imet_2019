#!/usr/bin/python3.6

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from debug import dprint


df = pd.read_csv('../input/train.csv')
cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)

labels = pd.read_csv('../input/labels.csv')
labels = labels.to_dict()['attribute_name']
counts = {labels[int(id)]:count for (id, count) in cls_counts.items()}
# dprint(counts)

# plt.plot(sorted(counts.values()))
# plt.show()

dprint(len([cnt for id,cnt in counts.items() if cnt < 100]))
dprint(len([cnt for id,cnt in counts.items() if cnt < 20]))
dprint(len([cnt for id,cnt in counts.items() if cnt < 10]))
dprint(len([cnt for id,cnt in counts.items() if cnt < 5]))
dprint(len([cnt for id,cnt in counts.items() if cnt < 3]))

# plt.plot(sorted([cnt for id,cnt in counts.items() if cnt < 100])); plt.show()

