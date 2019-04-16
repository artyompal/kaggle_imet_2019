#!/usr/bin/python3.6

import os, sys
from typing import List

def run(command: List[str]) -> None:
    res = os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + ' '.join(command))
    if res != 0:
        sys.exit()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'usage: {sys.argv[0]} model.py [parameters...]')
        sys.exit()

    start_fold = 1
    num_folds = 5

    for fold in range(start_fold, num_folds):
        cmd = ['python3.6'] + sys.argv[1:] + ['--fold', str(fold)]
        print('running', cmd)
        run(cmd)
