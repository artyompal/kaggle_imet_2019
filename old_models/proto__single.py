#!/usr/bin/python3.6

import itertools, os, re, sys
from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MODEL_PATH = '../input/seresnext50-resnet50/' if IN_KERNEL else ''

def run(command):
    res = os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + ' '.join(command))
    if res != 0:
        sys.exit()

models = {
    '2a_resnet50_f0_e17_0.5846.pth': 1
}

for model in models.keys():
    m = re.match(r'(.*)_f\d_e\d+.*\.pth', os.path.basename(model))
    if m:
        script_name = f'train_{m.group(1)}.py'

    cmd = ['python3.6', script_name, '--predict', '--weights', MODEL_PATH + model]
    print('running', cmd)
    res = run(cmd)

cmd = ['python3.6', 'blend.py', 'submission.csv']

for model, weight in models.items():
    name = os.path.splitext(os.path.basename(model))[0]
    predict = f'pred_level1_{name}.npz'
    cmd.extend([predict, str(weight)])

print('running', cmd)
run(cmd)
