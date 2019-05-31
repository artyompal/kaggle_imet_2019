#!/usr/bin/python3.6

import os
import re
import sys
import yaml

from glob import glob
from typing import List

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MODEL_PATH = '../input/' if IN_KERNEL else '../best_models/'


def run(command: List[str]) -> None:
    print('running', command)
    res = os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + ' '.join(command))
    if res != 0:
        sys.exit()

def generate_predict(filename: str) -> None:
    filename = os.path.basename(filename)
    m = re.match(r'level1_test_(.*).npy', filename)

    if not m:
        print('could not parse filename', filename)
        assert m

    weights = m.group(1) + '.pth'
    cmd = ['python3.6', 'train.py', '--num_ttas=2', '--predict_test', '--weights', model2path[weights]]
    run(cmd)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} ensemble.yml')
        sys.exit()

    model2path = {os.path.basename(path): path for path in glob(MODEL_PATH + '**/*.pth')}
    print('models found', model2path.keys())
    ensemble_file = sys.argv[1]

    with open(ensemble_file) as f:
        ensemble = yaml.load(f, Loader=yaml.SafeLoader)

    for predicts in ensemble:
        for pred in predicts['predicts']:
            predict_filename = pred.replace('_train_', '_test_')
            if not os.path.exists(predict_filename):
                generate_predict(predict_filename)

    run(['python3.6', 'ensemble_gen_submission.py', ensemble_file])
