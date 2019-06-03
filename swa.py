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
import torch.nn as nn
import torch.backends.cudnn as cudnn

import swa_impl

from data_loader import ImageDataset
from parse_config import load_config
from model import create_model
from metrics import F_score


def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = np.load(config.train.folds_file)
    assert folds.shape[0] == df.shape[0]
    return df.loc[folds != fold], df.loc[folds == fold]

def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system') # type: ignore
    cudnn.benchmark = True # type: ignore

    full_df = pd.read_csv('../input/train.csv')
    print('full_df', full_df.shape)
    train_df, val_df = train_val_split(full_df, fold)
    print('train_df', train_df.shape)

    num_ttas = 1

    if num_ttas > 1:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            albu.RandomCrop(height=config.model.input_size, width=config.model.input_size),
            # horizontal flip is done by the data loader
        ])
    else:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            albu.CenterCrop(height=config.model.input_size, width=config.model.input_size),
        ])

    val_dataset = ImageDataset(val_df, mode='val', config=config,
                               num_ttas=num_ttas, augmentor=transform_test)

    data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=True)

    return data_loader

def validate(data_loader: Any, model: Any) -> float:
    ''' Performs validation, returns validation score. '''
    model.eval()

    sigmoid = nn.Sigmoid()
    predicts_list, targets_list = [], []

    with torch.no_grad():
        for input_data in tqdm(data_loader):
            if data_loader.dataset.mode != 'test':
                input_, target = input_data
            else:
                input_, target = input_data, None

            if data_loader.dataset.num_ttas != 1:
                bs, ncrops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w)

                output = model(input_)
                output = sigmoid(output)

                if config.test.tta_combine_func == 'max':
                    output = output.view(bs, ncrops, -1).max(1)[0]
                elif config.test.tta_combine_func == 'mean':
                    output = output.view(bs, ncrops, -1).mean(1)
                else:
                    assert False
            else:
                output = model(input_.cuda())
                output = sigmoid(output)

            predicts_list.append(output.detach().cpu())
            targets_list.append(target)

    predicts, targets = torch.cat(predicts_list), torch.cat(targets_list)
    best_score, best_thresh = 0.0, 0.0

    for threshold in tqdm(np.linspace(0.05, 0.25, 100)):
        score = F_score(predicts, targets, beta=2, threshold=threshold)
        if score > best_score:
            best_score, best_thresh = score, threshold.item()

    print(f'F2 {best_score:.4f} threshold {best_thresh:.4f}')
    return best_score

def parse_model_name(path: str) -> Tuple[str, int, int, float]:
    m = re.match(r'(.*)_f(\d)_e(\d+)_([.0-9]+)\.pth', os.path.basename(path))
    assert m

    model_name = m.group(1)
    fold = int(m.group(2))
    epoch = int(m.group(3))
    score = float(m.group(4))

    return model_name, fold, epoch, score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='models path', type=str)
    args = parser.parse_args()

    files = glob(os.path.join(args.path, '*.pth'))
    files.sort(key=lambda path: parse_model_name(path)[2])
    print(np.array(files))
    assert files

    model_name, fold, _, __ = parse_model_name(files[0])
    print(f'model {model_name}, fold {fold}')

    config = load_config(f'config/{model_name}.yml', 0)

    avg_model = create_model(config, pretrained=False)
    cur_model = create_model(config, pretrained=False)

    data_loader = load_data(fold)

    print('averaging models')
    weights = torch.load(files[0], map_location='cpu')
    avg_model.load_state_dict(weights['state_dict'])

    for i, path in enumerate(tqdm(files[1:])):
        weights = torch.load(path, map_location='cpu')
        cur_model.load_state_dict(weights['state_dict'])

        # swa_impl.moving_average(avg_model, cur_model, 1.0 / (i + 2))
        swa_impl.moving_average(avg_model, cur_model, 0.5)

    with torch.no_grad():
        print('updating batchnorm')
        swa_impl.bn_update(data_loader, avg_model)

    print('predicting on validation set')
    score = validate(data_loader, avg_model)

    data_to_save = {
        'arch': config.model.arch,
        'state_dict': avg_model.state_dict(),
        'score': score,
        'config': config
    }

    torch.save(data_to_save, f'{model_name}_f{fold}_e99_{score:.04f}.pth')
