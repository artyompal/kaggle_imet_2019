#!/usr/bin/python3.6

import itertools, os, re, sys
from debug import dprint
from typing import List

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MODEL_PATH = '../input/seresnext50-resnet50/' if IN_KERNEL else ''

def run(command: List[str]) -> None:
    res = os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + ' '.join(command))
    if res != 0:
        sys.exit()

coeff = 4
models = {
    '2a_resnet50_f0_e17_0.5846.pth': 1,
    '2a_resnet50_f1_e16_0.5688.pth': 1,
    '2a_resnet50_f2_e34_0.5651.pth': 1,
    '2a_resnet50_f3_e11_0.5645.pth': 1,
    '2a_resnet50_f4_e14_0.5669.pth': 1,
    '2b_se_resnext50_f0_e09_0.5972.pth': coeff,
    '2b_se_resnext50_f1_e19_0.6006.pth': coeff,
    '2b_se_resnext50_f2_e20_0.5973.pth': coeff,
    '2b_se_resnext50_f3_e21_0.5982.pth': coeff,
    '2b_se_resnext50_f4_e21_0.5978.pth': coeff
}

for model in models.keys():
    m = re.match(r'(.*)_f(\d)_e\d+.*\.pth', os.path.basename(model))
    assert m

    script_name = f'train_{m.group(1)}.py'
    fold = m.group(2)

    cmd = ['python3.6', script_name, '--predict', '--weights', MODEL_PATH + model,
           '--fold', fold]
    print('running', cmd)
    run(cmd)

cmd = ['python3.6', 'blend.py', 'submission.csv']

for model, weight in models.items():
    name = os.path.splitext(os.path.basename(model))[0]
    predict = f'pred_level1_{name}.npz'
    cmd.extend([predict, str(weight)])

print('running', cmd)
run(cmd)
