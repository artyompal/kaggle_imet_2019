#!/usr/bin/python3.6
''' Averages checkpoints using SWA. '''

import argparse
import os
import re

from glob import glob
from typing import Any, Tuple

import numpy as np
import pandas as pd
import albumentations as albu

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

import swa_impl

from data_loader import ImageDataset
from parse_config import load_config
from model import create_model


def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = np.load(config.train.folds_file)
    assert folds.shape[0] == df.shape[0]
    return df.loc[folds != fold], df.loc[folds == fold]

def load_data() -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system') # type: ignore
    cudnn.benchmark = True # type: ignore

    fold = 0
    full_df = pd.read_csv('../input/train.csv')
    print('full_df', full_df.shape)
    train_df, val_df = train_val_split(full_df, fold)
    print('train_df', train_df.shape)

    transform_test = albu.Compose([
        albu.PadIfNeeded(config.model.input_size, config.model.input_size),
        albu.CenterCrop(height=config.model.input_size, width=config.model.input_size),
    ])

    # train_dataset = ImageDataset(train_df, mode='train', config=config,
    #                              augmentor=transform_test)

    val_dataset = ImageDataset(val_df, mode='val', config=config,
                               num_ttas=1, augmentor=transform_test)

    data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=True)

    return data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--coeff', help='moving average coefficient', type=float, required=True)
    parser.add_argument('path', help='models path', type=str)
    args = parser.parse_args()

    files = glob(os.path.join(args.path, '*.pth'))
    assert files
    m = re.match(r'(.*)_f(\d)_e(\d+)_([.0-9]+)\.pth', os.path.basename(files[0]))
    assert m

    model_name = m.group(1)
    print('model is', model_name)

    config = load_config(f'config/{model_name}.yml', 0)

    avg_model = create_model(config, pretrained=False)
    cur_model = create_model(config, pretrained=False)

    print('averaging models')
    weights = torch.load(files[0], map_location='cpu')
    avg_model.load_state_dict(weights['state_dict'])

    for i, path in enumerate(tqdm(files[1:])):
        path = torch.load(files[0], map_location='cpu')
        cur_model.load_state_dict(weights['state_dict'])
        swa_impl.moving_average(avg_model, cur_model, 1.0 / (i + 2))

    with torch.no_grad():
        data_loader = load_data()
        swa_impl.bn_update(data_loader, avg_model)
