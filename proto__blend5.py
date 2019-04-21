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

coeff = 3
models = {
    '2b_se_resnext50_f0_e14_0.5975.pth': coeff,
    '2b_se_resnext50_f1_e19_0.6006.pth': coeff,
    '2b_se_resnext50_f2_e20_0.5973.pth': coeff,
    '2b_se_resnext50_f3_e21_0.5982.pth': coeff,
    '2b_se_resnext50_f4_e21_0.5978.pth': coeff,
    '2i_se_resnext101_auto_aug_f0_e27_0.5991.pth': coeff,
    '2i_se_resnext101_auto_aug_f1_e23_0.6011.pth': coeff,
    '2i_se_resnext101_auto_aug_f2_e27_0.5986.pth': coeff,
    '2i_se_resnext101_auto_aug_f3_e27_0.5986.pth': coeff,
    '2i_se_resnext101_auto_aug_f4_e22_0.6007.pth': coeff,
    '2j_cbam_resnet50_auto_aug_f0_e27_0.5929.pth': 1,
    '2j_cbam_resnet50_auto_aug_f1_e38_0.5953.pth': 1,
    '2j_cbam_resnet50_auto_aug_f2_e35_0.5923.pth': 1,
    '2j_cbam_resnet50_auto_aug_f3_e28_0.5931.pth': 1,
    '2j_cbam_resnet50_auto_aug_f4_e27_0.5936.pth': 1,
    '2k_pnasnet_auto_aug_f0_e08_0.5949.pth': 1,
    '2k_pnasnet_auto_aug_f1_e15_0.5933.pth': 1,
    '2k_pnasnet_auto_aug_f2_e11_0.5929.pth': 1,
    '2k_pnasnet_auto_aug_f3_e13_0.5941.pth': 1,
    '2k_pnasnet_auto_aug_f4_e11_0.5936.pth': 1,
    '2o_se_resnext101_def_aug_f0_e30_0.5994.pth': coeff,
    '2o_se_resnext101_def_aug_f1_e30_0.6051.pth': coeff,
    '2o_se_resnext101_def_aug_f2_e25_0.6022.pth': coeff,
    '2o_se_resnext101_def_aug_f3_e26_0.6032.pth': coeff,
    '2o_se_resnext101_def_aug_f4_e21_0.6032.pth': coeff,
    }

model2path = {os.path.basename(path): path for path in glob(MODEL_PATH + '**/*.pth')}
for model in models.keys():
    assert os.path.exists(model2path[model])

for model in models.keys():
    m = re.match(r'(.*)_f(\d)_e\d+.*\.pth', os.path.basename(model))
    assert m

    script_name = f'train_{m.group(1)}.py'
    fold = m.group(2)

    cmd = ['python3.6', script_name, '--predict', '--weights', MODEL_PATH + model,
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
