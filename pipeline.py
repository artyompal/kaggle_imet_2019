#!/usr/bin/python3.6
''' Trains a model or infers predictions. '''

import argparse
import math
import multiprocessing as mp
import os
import pickle
import pprint
import random
import sys
import time

from typing import *
from collections import defaultdict, Counter
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from easydict import EasyDict as edict
from PIL import Image

import albumentations as albu
import parse_config

from data_loader import ImageDataset
from utils import create_logger, AverageMeter
from debug import dprint

from losses import get_loss
from schedulers import get_scheduler, is_scheduler_continuous, get_warmup_scheduler
from optimizers import get_optimizer, get_lr, set_lr
from metrics import F_score
from random_rect_crop import RandomRectCrop
from random_erase import RandomErase
from random_aligned_crop import RandomAlignedCrop
from models import create_model, freeze_layers, unfreeze_layers

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

if not IN_KERNEL:
    import torchcontrib


def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)
    fold_cls_counts = defaultdict(int) # type: ignore
    folds = [-1] * len(df)

    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                     total=len(df), disable=IN_KERNEL):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(config.model.num_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1

    return np.array(folds, dtype=np.uint8)

def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(config.train.folds_file):
        folds = make_folds(df)
        np.save(config.train.folds_file, folds)
    else:
        folds = np.load(config.train.folds_file)

    assert folds.shape[0] == df.shape[0]
    return df.loc[folds != fold].copy(), df.loc[folds == fold].copy()

def get_size(path: str) -> Tuple[int, int]:
    return np.array(Image.open(path)).shape

def load_image_sizes(df: pd.DataFrame, path: str) -> pd.DataFrame:
    ''' Reads image size for every image in dataframe. '''
    cache_path = f'sizes_{os.path.basename(path)}.pkl'
    print('image sizes cache:', cache_path)

    if os.path.exists(cache_path):
        sizes = pickle.load(open(cache_path, 'rb'))
    else:
        with mp.Pool() as pool:
            files = sorted(glob(os.path.join(path, '*.png')))
            sizes = list(tqdm(pool.imap(get_size, files), total=len(files), disable=IN_KERNEL))
        if not IN_KERNEL:
            with open(cache_path, 'wb') as f:
                pickle.dump(sizes, f)

    df['h'] = [s[0] for s in sizes]
    df['w'] = [s[1] for s in sizes]
    return df

def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system') # type: ignore
    cudnn.benchmark = True # type: ignore

    logger.info('config:')
    logger.info(pprint.pformat(config))

    full_df = pd.read_csv(config.train.csv)
    full_df = load_image_sizes(full_df, '../input/train')
    print('full_df', full_df.shape)
    train_df, val_df = train_val_split(full_df, fold)
    print('train_df', train_df.shape, 'val_df', val_df.shape)

    test_df = pd.read_csv(config.test.csv)
    test_df = load_image_sizes(test_df, '../input/test')

    augs: List[Union[albu.BasicTransform, albu.OneOf]] = []

    if config.augmentations.hflip:
        augs.append(albu.HorizontalFlip(.5))
    if config.augmentations.vflip:
        augs.append(albu.VerticalFlip(.5))
    if config.augmentations.rotate90:
        augs.append(albu.RandomRotate90())

    if config.augmentations.affine == 'soft':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=10, p=.75))
    elif config.augmentations.affine == 'medium':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2))
    elif config.augmentations.affine == 'hard':
        augs.append(albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75))

    if not config.train.use_arbitrary_sizes and config.augmentations.rect_crop.enable:
        augs.append(RandomRectCrop(rect_min_area=config.augmentations.rect_crop.rect_min_area,
                                   rect_min_ratio=config.augmentations.rect_crop.rect_min_ratio,
                                   image_size=config.model.image_size,
                                   input_size=config.model.input_size))

    if config.augmentations.noise != 0:
        augs.append(albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=config.augmentations.noise))

    if config.augmentations.blur != 0:
        augs.append(albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=config.augmentations.blur))

    if config.augmentations.distortion != 0:
        augs.append(albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=config.augmentations.distortion))

    if config.augmentations.color != 0:
        augs.append(albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(),
        ], p=config.augmentations.color))

    if config.augmentations.erase.prob != 0:
        augs.append(RandomErase(min_area=config.augmentations.erase.min_area,
                                max_area=config.augmentations.erase.max_area,
                                min_ratio=config.augmentations.erase.min_ratio,
                                max_ratio=config.augmentations.erase.max_ratio,
                                input_size=config.model.input_size,
                                p=config.augmentations.erase.prob))

    if not config.train.use_arbitrary_sizes:
        random_crop = albu.RandomCrop(height=config.model.input_size,
                                      width=config.model.input_size)
        center_crop = albu.CenterCrop(height=config.model.input_size,
                                      width=config.model.input_size)
    else:
        random_crop = RandomAlignedCrop(config.train.max_image_size)
        center_crop = RandomAlignedCrop(config.train.max_image_size, center=True)

    transform_train = albu.Compose([
        albu.PadIfNeeded(config.model.input_size, config.model.input_size),
        random_crop,
        albu.Compose(augs, p=config.augmentations.global_prob),
        ])

    if config.test.num_ttas > 1:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            random_crop,
            albu.HorizontalFlip(),
        ])
    else:
        transform_test = albu.Compose([
            albu.PadIfNeeded(config.model.input_size, config.model.input_size),
            center_crop
        ])


    train_dataset = ImageDataset(train_df, mode='train', config=config,
                                 augmentor=transform_train)

    val_dataset = ImageDataset(val_df, mode='val', config=config,
                               augmentor=transform_test)

    test_dataset = ImageDataset(test_df, mode='test', config=config,
                                augmentor=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=not config.train.use_arbitrary_sizes,
        num_workers=config.num_workers, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.num_workers)

    return train_loader, val_loader, test_loader

def lr_finder(train_loader: Any, model: Any, criterion: Any, optimizer: Any) -> None:
    ''' Finds the optimal LR range and sets up first optimizer parameters. '''
    logger.info('lr_finder called')

    batch_time = AverageMeter()
    num_steps = min(len(train_loader), config.train.lr_finder.num_steps)
    logger.info(f'total batches: {num_steps}')
    end = time.time()
    lr_str = ''
    model.train()

    init_value = config.train.lr_finder.init_value
    final_value = config.train.lr_finder.final_value
    beta = config.train.lr_finder.beta

    mult = (final_value / init_value) ** (1 / (num_steps - 1))
    lr = init_value

    avg_loss = best_loss = 0.0
    losses = np.zeros(num_steps)
    logs = np.zeros(num_steps)

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        set_lr(optimizer, lr)

        output = model(input_.cuda())
        loss = criterion(output, target.cuda())
        loss_val = loss.data.item()

        predict = (output.detach() > 0.1).type(torch.FloatTensor)
        f2 = F_score(predict, target, beta=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_str = f'\tlr {lr:.08f}'

        # compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        # stop if the loss is exploding
        if i > 0 and smoothed_loss > 4 * best_loss:
            break

        # record the best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss

        # store the values
        losses[i] = smoothed_loss
        logs[i] = math.log10(lr)

        # update the lr for the next step
        lr *= mult

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'lr_finder [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {loss:.4f} ({smoothed_loss:.4f})\t'
                        f'F2 {f2:.4f} {lr_str}')

    np.savez(os.path.join(config.experiment_dir, f'lr_finder_{config.version}'),
             logs=logs, losses=losses)

    d1 = np.zeros_like(losses); d1[1:] = losses[1:] - losses[:-1]
    first, last = np.argmin(d1), np.argmin(losses)

    MAGIC_COEFF = 4

    highest_lr = 10 ** logs[last]
    best_high_lr = highest_lr / MAGIC_COEFF
    best_low_lr = 10 ** logs[first]
    logger.info(f'best_low_lr={best_low_lr} best_high_lr={best_high_lr} '
                f'highest_lr={highest_lr}')

    def find_nearest(array: np.array, value: float) -> int:
        return (np.abs(array - value)).argmin()

    last = find_nearest(logs, math.log10(best_high_lr))
    logger.info(f'first={first} last={last}')

    import matplotlib.pyplot as plt
    plt.plot(logs, losses, '-D', markevery=[first, last])
    plt.savefig(os.path.join(config.experiment_dir, 'lr_finder_plot.png'))

def mixup(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Performs mixup: https://arxiv.org/pdf/1710.09412.pdf '''
    coeff = np.random.beta(config.train.mixup.beta_a, config.train.mixup.beta_a)
    indices = np.roll(np.arange(x.shape[0]), np.random.randint(1, x.shape[0]))
    indices = torch.tensor(indices).cuda()

    x = x * coeff + x[indices] * (1 - coeff)
    y = y * coeff + y[indices] * (1 - coeff)
    return x, y

def train_epoch(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
                epoch: int, lr_scheduler: Any, lr_scheduler2: Any,
                max_steps: Optional[int]) -> None:
    logger.info(f'epoch: {epoch}')
    logger.info(f'learning rate: {get_lr(optimizer)}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    if max_steps is not None:
        num_steps = min(max_steps, num_steps)
    num_steps -= num_steps % config.train.accum_batches_num

    logger.info(f'total batches: {num_steps}')
    end = time.time()
    lr_str = ''

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        input_ = input_.cuda()

        if config.train.mixup.enable:
            input_, target = mixup(input_, target)

        output = model(input_)
        loss = criterion(output, target.cuda())

        predict = (output.detach() > 0.1).type(torch.FloatTensor)
        avg_score.update(F_score(predict, target, beta=2))

        losses.update(loss.data.item(), input_.size(0))
        loss.backward()

        if (i + 1) % config.train.accum_batches_num == 0:
            optimizer.step()
            optimizer.zero_grad()

        if is_scheduler_continuous(lr_scheduler):
            lr_scheduler.step()
            lr_str = f'\tlr {get_lr(optimizer):.08f}'

        if is_scheduler_continuous(lr_scheduler2):
            lr_scheduler2.step()
            lr_str = f'\tlr {get_lr(optimizer):.08f}'

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'F2 {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

    if config.train.swa.enable and epoch > 0 and epoch % config.train.swa.period == 0:
        optimizer.update_swa()

    logger.info(f' * average F2 on train {avg_score.avg:.4f}')

def inference(data_loader: Any, model: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    ''' Returns predictions and targets, if any. '''
    model.eval()

    sigmoid = nn.Sigmoid()
    predicts_list, targets_list = [], []

    with torch.no_grad():
        for i, (input_, target) in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if config.test.num_ttas != 1 and data_loader.dataset.mode == 'test':
                bs, ncrops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w) # fuse batch size and ncrops

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

            predicts_list.append(output.detach().cpu().numpy())
            if target is not None:
                targets_list.append(target)

    predicts = np.concatenate(predicts_list)
    targets = np.concatenate(targets_list)
    return predicts, targets

def validate(val_loader: Any, model: Any, epoch: int) -> Tuple[float, float, np.ndarray]:
    ''' Calculates validation score.
    1. Infers predictions
    2. Finds optimal threshold
    3. Returns the best score and a threshold. '''
    logger.info('validate()')

    predicts, targets = inference(val_loader, model)
    predicts, targets = torch.tensor(predicts), torch.tensor(targets)
    best_score, best_thresh = 0.0, 0.0

    for threshold in tqdm(np.linspace(0.05, 0.15, 33), disable=IN_KERNEL):
        score = F_score(predicts, targets, beta=2, threshold=threshold)
        if score > best_score:
            best_score, best_thresh = score, threshold

    logger.info(f'{epoch} F2 {best_score:.4f} threshold {best_thresh:.4f}')
    logger.info(f' * F2 on validation {best_score:.4f}')
    return best_score, best_thresh, predicts.numpy()

def gen_prediction(val_loader: Any, test_loader: Any, model: Any, epoch: int,
                   model_path: str) -> np.ndarray:
    # calculate the threshold first
    score, threshold, predicts = validate(val_loader, model, epoch)

    if args.dataset == 'test':
        predicts, _ = inference(test_loader, model)

    predicts -= threshold

    name = 'test' if args.dataset == 'test' else 'train'
    filename = f'level1_{name}_{os.path.splitext(os.path.basename(model_path))[0]}'
    np.save(filename, predicts)

def run() -> float:
    np.random.seed(0)
    model_dir = config.experiment_dir

    logger.info('=' * 50)

    train_loader, val_loader, test_loader = load_data(args.fold)
    model = create_model(config, logger, args)
    criterion = get_loss(config)

    if args.lr_finder:
        optimizer = get_optimizer(config, model.parameters())
        lr_finder(train_loader, model, criterion, optimizer)
        sys.exit()

    if args.weights is None and config.train.head_only_warmup:
        logger.info('-' * 50)
        logger.info(f'doing warmup for {config.train.warmup.steps} steps')
        logger.info(f'max_lr will be {config.train.warmup.max_lr}')

        optimizer = get_optimizer(config, model.parameters())
        warmup_scheduler = get_warmup_scheduler(config, optimizer)

        freeze_layers(model)
        train_epoch(train_loader, model, criterion, optimizer, 0,
                    warmup_scheduler, None, config.train.warmup.steps)
        unfreeze_layers(model)

    if args.weights is None and config.train.enable_warmup:
        logger.info('-' * 50)
        logger.info(f'doing warmup for {config.train.warmup.steps} steps')
        logger.info(f'max_lr will be {config.train.warmup.max_lr}')

        optimizer = get_optimizer(config, model.parameters())
        warmup_scheduler = get_warmup_scheduler(config, optimizer)
        train_epoch(train_loader, model, criterion, optimizer, 0,
                    warmup_scheduler, None, config.train.warmup.steps)

    optimizer = get_optimizer(config, model.parameters())

    if config.train.swa.enable:
        optimizer = torchcontrib.optim.SWA(optimizer)

    if args.weights is None:
        last_epoch = -1
    else:
        last_checkpoint = torch.load(args.weights)
        assert last_checkpoint['arch'] == config.model.arch
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint loaded: {args.weights}')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_epoch}')

        if args.lr_override != 0:
            set_lr(optimizer, float(args.lr_override))
        elif 'lr' in config.optimizer.params:
            set_lr(optimizer, config.optimizer.params.lr)

    lr_scheduler = get_scheduler(config.scheduler, optimizer, last_epoch=
                                 (last_epoch if config.scheduler.name != 'cosine' else -1))
    lr_scheduler2 = get_scheduler(config.scheduler2, optimizer, last_epoch=last_epoch) \
                    if config.scheduler2.name else None

    if args.gen_predict:
        print('inference mode')
        assert args.weights is not None
        gen_prediction(val_loader, test_loader, model, last_epoch, args.weights)
        sys.exit()

    logger.info(f'training will start from epoch {last_epoch + 1}')

    best_score = 0.0
    best_epoch = 0

    last_lr = get_lr(optimizer)
    best_model_path = args.weights

    for epoch in range(last_epoch + 1, config.train.num_epochs):
        logger.info('-' * 50)

        if not is_scheduler_continuous(lr_scheduler) and lr_scheduler2 is None:
            # if we have just reduced LR, reload the best saved model
            lr = get_lr(optimizer)

            if lr < last_lr - 1e-10 and best_model_path is not None:
                logger.info(f'learning rate dropped: {lr}, reloading')
                last_checkpoint = torch.load(best_model_path)

                assert(last_checkpoint['arch']==config.model.arch)
                model.load_state_dict(last_checkpoint['state_dict'])
                optimizer.load_state_dict(last_checkpoint['optimizer'])
                logger.info(f'checkpoint loaded: {best_model_path}')
                set_lr(optimizer, lr)
                last_lr = lr

        train_epoch(train_loader, model, criterion, optimizer, epoch,
                    lr_scheduler, lr_scheduler2, config.train.max_steps_per_epoch)
        score, _, _ = validate(val_loader, model, epoch)

        if config.scheduler.name == 'reduce_lr_on_plateau':
            lr_scheduler.step(metrics=score)
        elif not is_scheduler_continuous(lr_scheduler):
            lr_scheduler.step()

        if config.scheduler2.name == 'reduce_lr_on_plateau':
            lr_scheduler2.step(metrics=score)
        elif lr_scheduler2 and not is_scheduler_continuous(lr_scheduler2):
            lr_scheduler2.step()

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        if is_best:
            best_model_path = os.path.join(model_dir,
                f'{config.version}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth')

            data_to_save = {
                'epoch': epoch,
                'arch': config.model.arch,
                'state_dict': model.state_dict(),
                'score': score,
                'optimizer': optimizer.state_dict(),
                'config': config
            }

            torch.save(data_to_save, best_model_path)
            logger.info(f'a snapshot was saved to {best_model_path}')

    logger.info(f'best score: {best_score:.04f}')
    return -best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model configuration file (YAML)', type=str, required=True)
    parser.add_argument('--lr_finder', help='invoke LR finder and exit', action='store_true')
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--dataset', help='dataset for prediction, train/test',
                        type=str, default='test')
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--gen_predict', help='make predictions for the testset and return', action='store_true')
    parser.add_argument('--summary', help='show model summary', action='store_true')
    parser.add_argument('--lr_override', help='override learning rate', type=float, default=0)
    args = parser.parse_args()

    config = parse_config.load(args.config, args)

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)

    log_filename = 'log_training.txt' if not args.gen_predict else 'log_predict.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))
    run()
