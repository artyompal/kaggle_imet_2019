#!/usr/bin/python3.6

import os, re, sys
from typing import List
from glob import glob
from debug import dprint


def run(command: List[str]) -> None:
    res = os.system(' '.join(command))
    if res != 0:
        sys.exit()

def usage() -> None:
    print(f'usage: {sys.argv[0]} [--start_fold NUM] [--num_folds NUM] model.py [parameters...]')
    sys.exit()

def get_saved_model_score(model: str, script: str) -> float:
    model = os.path.basename(model)
    m = re.match(r'(.*)_f(\d)_e\d+.*_(0\.\d+)\.pth', model)
    if not m:
        print('could not parse filename', model)
        return 0

    script_name = f'train_{m.group(1)}.py'
    if script != script_name:
        print(f'detected script name {script_name} does not match {script}')
        return 0

    return float(m.group(3))

def find_best_model(script: str, fold: int) -> List[str]:
    script = os.path.basename(script)
    model = script[6:-3]

    scores = {get_saved_model_score(path, script): path
             for path in glob(f'../models/{model}/fold_{fold}/*.pth')}
    if not scores:
        return []

    return ['--weights', scores[max(scores.keys())]]

if __name__ == '__main__':
    if len(sys.argv) == 1:
        usage()

    params = sys.argv[1:]
    options = {'--start_fold': 0, '--num_folds': 5}

    while params[0] in options.keys():
        if len(params) < 3:
            usage()

        options[params[0]] = int(params[1])
        params = params[2:]

    for fold in range(options['--start_fold'], options['--num_folds']):
        start_weights = find_best_model(params[0], fold)

        cmd = ['python3.6'] + params + ['--fold', str(fold)] + start_weights
        print('running', cmd)
        run(cmd)
