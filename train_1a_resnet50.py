#!/usr/bin/python3.6
''' Trains a model. '''

import argparse, hashlib, logging, math, os, pprint, sys, time
from typing import *
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from easydict import EasyDict as edict
from sklearn.model_selection import KFold
import torchsummary
import pretrainedmodels
import PIL

from data_loader_v1_single import Dataset
from utils import create_logger, AverageMeter, F_score
from debug import dprint, assert_eq, assert_ne
from cosine_scheduler import CosineLRWithRestarts
from tqdm import tqdm


opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
# opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224 # crop size
opt.MODEL.VERSION = os.path.splitext(os.path.basename(__file__))[0][6:]
opt.MODEL.DROPOUT = 0.5
opt.MODEL.NUM_CLASSES = 1103
opt.MODEL.POOL_SIZE = 1 # 7

opt.MODELS_DIR = 'models'

opt.EXPERIMENT = edict()
opt.EXPERIMENT.DIR = f'models/{opt.MODEL.VERSION}'

opt.LOG = edict()
opt.LOG.LOG_FILE = os.path.join(opt.EXPERIMENT.DIR, f'log_training.txt')

opt.TRAIN = edict()
opt.TRAIN.NUM_FOLDS = 5
opt.TRAIN.BATCH_SIZE = 32 * torch.cuda.device_count()
opt.TRAIN.LOSS = 'BCE'
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 12
opt.TRAIN.PRINT_FREQ = 100
opt.TRAIN.LEARNING_RATE = 0.001
opt.TRAIN.PATIENCE = 100 # 8
opt.TRAIN.SAMPLES_PER_CLASS = 10
opt.TRAIN.LR_REDUCE_FACTOR = 0.2
opt.TRAIN.MIN_LR = 1e-7
opt.TRAIN.EPOCHS = 1000
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.STEPS_PER_EPOCH = 30000
opt.TRAIN.PATH = 'input/train'
opt.TRAIN.CSV = 'input/train.csv'
opt.TRAIN.OPTIMIZER = 'Adam'
opt.TRAIN.MIN_IMPROVEMENT = 0.001
opt.TRAIN.RESUME = None if len(sys.argv) == 1 else sys.argv[-1]

opt.TRAIN.COSINE = edict()
opt.TRAIN.COSINE.ENABLE = False
opt.TRAIN.COSINE.LR = 1e-4
opt.TRAIN.COSINE.PERIOD = 10
opt.TRAIN.COSINE.COEFF = 1.2

opt.TEST = edict()
opt.TEST.PATH = f'input/test'
opt.TEST.CSV = f'input/sample_submission.csv'


def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=opt.TRAIN.NUM_FOLDS, shuffle=True, random_state=0)
    train_idx, val_idx = list(kf.split(df))[fold]
    return df.iloc[train_idx], df.iloc[val_idx]

def load_data(fold: int) -> Any:
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    logger.info('Options:')
    logger.info(pprint.pformat(opt))

    full_df = pd.read_csv(opt.TRAIN.CSV)
    print('full_df', full_df.shape)
    train_df, val_df = train_val_split(full_df, fold)
    print('train_df', train_df.shape, 'val_df', val_df.shape)
    test_df = pd.read_csv(opt.TEST.CSV)

    transform_train = transforms.Compose([
        # transforms.Resize((opt.MODEL.IMAGE_SIZE)), # smaller edge
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), shear=10, resample=PIL.Image.BILINEAR),
        # transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
    ])

    # transform_val = transforms.Compose([
    #     transforms.Resize((opt.MODEL.IMAGE_SIZE)),
    #     transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    # ])

    train_dataset = Dataset(train_df, path=opt.TRAIN.PATH, mode='train',
                            num_classes=opt.MODEL.NUM_CLASSES,
                            image_size=opt.MODEL.INPUT_SIZE,
                            augmentor=transform_train)

    val_dataset = Dataset(val_df, path=opt.TRAIN.PATH, mode='val',
                          image_size=opt.MODEL.INPUT_SIZE,
                          num_classes=opt.MODEL.NUM_CLASSES)
    test_dataset = Dataset(test_df, path=opt.TEST.PATH, mode='test',
                           image_size=opt.MODEL.INPUT_SIZE,
                           num_classes=opt.MODEL.NUM_CLASSES)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=opt.TRAIN.WORKERS)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    return train_loader, val_loader, test_loader

def create_model() -> Any:
    logger.info(f'creating a model {opt.MODEL.ARCH}')

    logger.info("using model '{}'".format(opt.MODEL.ARCH ))
    model = models.__dict__[opt.MODEL.ARCH](pretrained=True)

    # assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AdaptiveAvgPool2d(opt.MODEL.POOL_SIZE)
    model.fc = nn.Linear(model.fc.in_features, opt.MODEL.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    # if torch.cuda.device_count() == 1:
    #     torchsummary.summary(model, (3, 224, 224))

    return model

def save_checkpoint(state: Dict[str, Any], filename: str) -> None:
    torch.save(state, os.path.join(opt.EXPERIMENT.DIR, filename))
    logger.info(f'A snapshot was saved to {filename}')

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any) -> None:
    logger.info(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    print('total batches:', len(train_loader))
    num_steps = min(len(train_loader), opt.TRAIN.STEPS_PER_EPOCH)

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        if i >= opt.TRAIN.STEPS_PER_EPOCH:
            break

        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())

        # get metric
        predict = (output.detach() > 0.5).type(torch.FloatTensor)
        avg_score.update(F_score(predict, target))

        # compute gradient and do SGD step
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(lr_scheduler, 'batch_step'):
            lr_scheduler.batch_step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'F2 {avg_score.val:.4f} ({avg_score.avg:.4f})')

    logger.info(f' * average accuracy on train {avg_score.avg:.4f}')

def inference(data_loader: Any, model: Any) -> Tuple[torch.tensor, torch.tensor]:
    ''' Returns predictions and targets, if any. '''
    model.eval()

    sigmoid = nn.Sigmoid()
    predicts_list, targets_list = [], []

    with torch.no_grad():
        for i, (input_, target) in enumerate(tqdm(data_loader)):
            output = model(input_.cuda())
            output = sigmoid(output)

            predicts_list.append(output.detach().cpu().numpy())
            if target is not None:
                targets_list.append(target)

    predicts = np.concatenate(predicts_list)
    targets = np.concatenate(targets_list)
    return predicts, targets

def validate(val_loader: Any, model: Any, epoch: int) -> Tuple[float, float]:
    ''' Calculates validation score.
    1. Infers predictions
    2. Finds optimal threshold
    3. Returns the best score and a threshold. '''
    logger.info('validate()')

    predicts, targets = inference(val_loader, model)
    predicts, targets = torch.tensor(predicts), torch.tensor(targets)
    best_score, best_thresh = 0.0, 0.0

    for threshold in tqdm(np.linspace(0.40, 0.60, 40)):
        score = F_score(predicts, targets, threshold=threshold)

        if score > best_score:
            best_score, best_thresh = score, threshold

    logger.info(f'{epoch} F2 {best_score:.4f} threshold {best_thresh:.4f}')
    logger.info(f' * F2 on validation {best_score:.4f}')
    return best_score, best_thresh

def generate_submission(val_loader: Any, test_loader: Any, model: Any,
                        epoch: int, model_path: Any) -> np.ndarray:
    score, threshold = validate(val_loader, model, epoch)
    predicts, _ = inference(test_loader, model)

    dprint(predicts.shape)
    labels = [" ".join([str(i) for i, p in enumerate(pred) if p > threshold])
              for pred in tqdm(predicts)]
    dprint(len(labels))
    dprint(np.array(labels))

    sub = test_loader.dataset.df
    sub['attribute_ids'] = labels
    sub_name = f'submissions_{os.path.basename(model_path)[:-4]}.csv'
    sub.to_csv(sub_name, index=False)

def set_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr
       param_group['initial_lr'] = lr

def read_lr(optimizer: Any) -> float:
    for param_group in optimizer.param_groups:
       lr = float(param_group['lr'])
       logger.info(f'learning rate: {lr}')
       return lr

    assert False

if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", help="model to resume training", type=str)
    parser.add_argument("--predict", help="model to resume training", action='store_true')
    parser.add_argument("--fold", help="fold number", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(opt.EXPERIMENT.DIR):
        os.makedirs(opt.EXPERIMENT.DIR)

    logger = create_logger(opt.LOG.LOG_FILE)
    logger.info('=' * 50)

    print("available models:", pretrainedmodels.model_names)
    train_loader, val_loader, test_loader = load_data(args.fold)
    model = create_model()

    if opt.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), opt.TRAIN.LEARNING_RATE)
    elif opt.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), opt.TRAIN.LEARNING_RATE,
                              momentum=0.9, nesterov=True)
    else:
        assert False

    if opt.TRAIN.COSINE.ENABLE:
        set_lr(optimizer, opt.TRAIN.COSINE.LR)
        lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
            opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
            restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                           patience=opt.TRAIN.PATIENCE, factor=opt.TRAIN.LR_REDUCE_FACTOR,
                           verbose=True, min_lr=opt.TRAIN.MIN_LR,
                           threshold=opt.TRAIN.MIN_IMPROVEMENT, threshold_mode='abs')

    if args.pretrained is None:
        last_epoch = 0
        logger.info(f'training will start from epoch {last_epoch+1}')
    else:
        last_checkpoint = torch.load(args.pretrained)
        assert(last_checkpoint['arch']==opt.MODEL.ARCH)
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint {args.pretrained} was loaded.')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_checkpoint["epoch"]}')


    if args.predict:
        print('inference mode')
        generate_submission(val_loader, test_loader, model, last_epoch, args.pretrained)
        sys.exit(0)

    if opt.TRAIN.LOSS == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise RuntimeError('unknown loss specified')

    best_score = 0.0
    best_epoch = 0

    for epoch in range(last_epoch + 1, opt.TRAIN.EPOCHS + 1):
        logger.info('-' * 50)

        if not opt.TRAIN.COSINE.ENABLE:
            lr = read_lr(optimizer)

            if lr < opt.TRAIN.MIN_LR * 1.01:
                logger.info(f'lr={lr}, start cosine annealing!')
                set_lr(optimizer, opt.TRAIN.COSINE.LR)
                opt.TRAIN.COSINE.ENABLE = True

                lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
                    opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
                    restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)

        if opt.TRAIN.COSINE.ENABLE:
            lr_scheduler.step()

        read_lr(optimizer)

        train(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        score, _ = validate(val_loader, model, epoch)

        if not opt.TRAIN.COSINE.ENABLE:
            lr_scheduler.step(score)    # type: ignore

        is_best = score > best_score
        best_score = max(score, best_score)
        if is_best:
            best_epoch = epoch

        data_to_save = {
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'score': score,
            'optimizer': optimizer.state_dict(),
            'options': opt
        }

        filename = opt.MODEL.VERSION
        if is_best:
            save_checkpoint(data_to_save, f'{filename}_e{epoch:03d}_{score:.04f}.pth')

    logger.info(f'best score: {best_score:.04f}')
