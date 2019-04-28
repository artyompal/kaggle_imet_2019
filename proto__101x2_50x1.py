#!/usr/bin/python3.6

import itertools, os, re, sys
from glob import glob
from typing import List
from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MODEL_PATH = '../input/' if IN_KERNEL else '../best_models/'

def run(command: List[str]) -> None:
    res = os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + ' '.join(command))
    if res != 0:
        sys.exit()

coeff1, coeff2, coeff3 = 3, 2, 1
models = {
    '4b_se_resnext101_352x352_f0_e16_0.6050.pth': coeff1,
    '4b_se_resnext101_352x352_f1_e29_0.6079.pth': coeff1,
    '4b_se_resnext101_352x352_f2_e24_0.6022.pth': coeff1,
    '4b_se_resnext101_352x352_f3_e21_0.6047.pth': coeff1,
    '4b_se_resnext101_352x352_f4_e24_0.6050.pth': coeff1,
    '2o_se_resnext101_def_aug_f0_e30_0.5994.pth': coeff2,
    '2o_se_resnext101_def_aug_f1_e30_0.6051.pth': coeff2,
    '2o_se_resnext101_def_aug_f2_e25_0.6022.pth': coeff2,
    '2o_se_resnext101_def_aug_f3_e26_0.6032.pth': coeff2,
    '2o_se_resnext101_def_aug_f4_e21_0.6032.pth': coeff2,
    '2b_se_resnext50_f0_e14_0.5975.pth': coeff3,
    '2b_se_resnext50_f1_e19_0.6006.pth': coeff3,
    '2b_se_resnext50_f2_e20_0.5973.pth': coeff3,
    '2b_se_resnext50_f3_e21_0.5982.pth': coeff3,
    '2b_se_resnext50_f4_e21_0.5978.pth': coeff3,
    }

model2path = {os.path.basename(path): path for path in glob(MODEL_PATH + '**/*.pth')}
for model in models.keys():
    assert os.path.exists(model2path[model])

for model in models.keys():
    m = re.match(r'(.*)_f(\d)_e\d+.*\.pth', os.path.basename(model))
    assert m

    script_name = f'train_{m.group(1)}.py'
    fold = m.group(2)

    cmd = ['python3.6', script_name, '--predict', '--weights', model2path[model],
           '--fold', fold, '--num_tta', '1']
    print('running', cmd)
    run(cmd)

cmd = ['python3.6', 'blend.py', 'submission.csv']

for model, weight in models.items():
    name = os.path.splitext(os.path.basename(model))[0]
    predict = f'pred_level1_{name}.npz'
    cmd.extend([predict, str(weight)])

print('running', cmd)
run(cmd)
