#!/usr/bin/python3.6
''' Trains a model. '''

import argparse, hashlib, logging, math, os, pprint, random, sys, time
import multiprocessing
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

from sklearn.model_selection import KFold
import PIL

from data_loader_v1_single import Dataset
from utils import create_logger, AverageMeter, F_score
from debug import dprint, assert_eq, assert_ne
from cosine_scheduler import CosineLRWithRestarts
from tqdm import tqdm

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

if not IN_KERNEL:
    import torchsummary
    from pytorchcv.model_provider import get_model
    from hyperopt import hp, tpe, fmin
else:
    import senet

from easydict import EasyDict as edict # type: ignore

opt = edict()
opt.INPUT = '../input/imet-2019-fgvc6/' if IN_KERNEL else '../input/'

opt.MODEL = edict()
opt.MODEL.ARCH = 'seresnext101_32x4d'
# opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 288 # crop size
opt.MODEL.VERSION = os.path.splitext(os.path.basename(__file__))[0][6:]
opt.MODEL.DROPOUT = 0.5
opt.MODEL.NUM_CLASSES = 1103

opt.EXPERIMENT_DIR = f'../models/{opt.MODEL.VERSION}'

opt.TRAIN = edict()
opt.TRAIN.NUM_FOLDS = 5
opt.TRAIN.BATCH_SIZE = 20 * torch.cuda.device_count()
opt.TRAIN.LOSS = 'BCE'
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = min(12, multiprocessing.cpu_count())
opt.TRAIN.PRINT_FREQ = 100
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.PATIENCE = 4
opt.TRAIN.LR_REDUCE_FACTOR = 0.2
opt.TRAIN.MIN_LR = 1e-7
opt.TRAIN.EPOCHS = 30
opt.TRAIN.STEPS_PER_EPOCH = 30000
opt.TRAIN.PATH = opt.INPUT + 'train'
opt.TRAIN.FOLDS_FILE = 'folds.npy'
opt.TRAIN.CSV = opt.INPUT + 'train.csv'
opt.TRAIN.OPTIMIZER = 'Adam'
opt.TRAIN.MIN_IMPROVEMENT = 0.001

opt.TRAIN.COSINE = edict()
opt.TRAIN.COSINE.ENABLE = False
opt.TRAIN.COSINE.LR = 1e-4
opt.TRAIN.COSINE.PERIOD = 10
opt.TRAIN.COSINE.COEFF = 1.2

opt.TEST = edict()
opt.TEST.PATH = opt.INPUT + 'test'
opt.TEST.CSV = opt.INPUT + 'sample_submission.csv'
opt.TEST.NUM_TTAS = 4
opt.TEST.TTA_COMBINE_FUNC = 'mean'


def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)
    fold_cls_counts = defaultdict(int) # type: ignore
    folds = [-1] * len(df)

    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df), disable=IN_KERNEL):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(opt.TRAIN.NUM_FOLDS)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1

    return np.array(folds, dtype=np.uint8)

def train_val_split(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(opt.TRAIN.FOLDS_FILE):
        folds = make_folds(df)
        np.save(opt.TRAIN.FOLDS_FILE, folds)
    else:
        folds = np.load(opt.TRAIN.FOLDS_FILE)

    assert folds.shape[0] == df.shape[0]
    return df.loc[folds != fold], df.loc[folds == fold]

def load_data(fold: int, params: Dict[str, Any]) -> Any:
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
        transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), shear=10, resample=PIL.Image.BILINEAR),
        # transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((opt.MODEL.IMAGE_SIZE)),
        # transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
        transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
    ])


    train_dataset = Dataset(train_df, path=opt.TRAIN.PATH, mode='train',
                            num_classes=opt.MODEL.NUM_CLASSES, resize=False,
                            augmentor=transform_train)

    val_dataset = Dataset(val_df, path=opt.TRAIN.PATH, mode='val',
                          # image_size=opt.MODEL.INPUT_SIZE,
                          num_classes=opt.MODEL.NUM_CLASSES, resize=False,
                          num_tta=1, # opt.TEST.NUM_TTAS,
                          augmentor=transform_test)
    test_dataset = Dataset(test_df, path=opt.TEST.PATH, mode='test',
                           # image_size=opt.MODEL.INPUT_SIZE,
                           num_classes=opt.MODEL.NUM_CLASSES, resize=False,
                           num_tta=opt.TEST.NUM_TTAS,
                           augmentor=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=opt.TRAIN.WORKERS)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)

    return train_loader, val_loader, test_loader

def create_model(predict_only: bool, dropout: float) -> Any:
    logger.info(f'creating a model {opt.MODEL.ARCH}')

    model = get_model(opt.MODEL.ARCH, pretrained=not predict_only)

    model.features[-1] = nn.AdaptiveAvgPool2d(1)

    if opt.MODEL.ARCH == 'pnasnet5large':
        if dropout < 0.1:
            model.output = nn.Linear(model.output[-1].in_features, opt.MODEL.NUM_CLASSES)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output[-1].in_features, opt.MODEL.NUM_CLASSES))
    else:
        if dropout < 0.1:
            model.output = nn.Linear(model.output.in_features, opt.MODEL.NUM_CLASSES)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output.in_features, opt.MODEL.NUM_CLASSES))

    model = torch.nn.DataParallel(model).cuda()
    model.cuda()
    return model

def save_checkpoint(state: Dict[str, Any], filename: str, model_dir: str) -> None:
    torch.save(state, os.path.join(model_dir, filename))
    logger.info(f'A snapshot was saved to {filename}')

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any) -> None:
    logger.info(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    num_steps = min(len(train_loader), opt.TRAIN.STEPS_PER_EPOCH)
    print('total batches:', len(train_loader))

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        if i >= opt.TRAIN.STEPS_PER_EPOCH:
            break

        # compute output
        output = model(input_.cuda())
        loss = criterion(output, target.cuda())

        # get metric
        predict = (output.detach() > 0.5).type(torch.FloatTensor)
        avg_score.update(F_score(predict, target).item())

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
        for i, (input_, target) in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if opt.TEST.NUM_TTAS != 1 and data_loader.dataset.mode == 'test':
                bs, ncrops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w) # fuse batch size and ncrops

                output = model(input_)
                output = sigmoid(output)

                if opt.TEST.TTA_COMBINE_FUNC == 'max':
                    output = output.view(bs, ncrops, -1).max(1)[0]
                elif opt.TEST.TTA_COMBINE_FUNC == 'mean':
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

def validate(val_loader: Any, model: Any, epoch: int) -> Tuple[float, float]:
    ''' Calculates validation score.
    1. Infers predictions
    2. Finds optimal threshold
    3. Returns the best score and a threshold. '''
    logger.info('validate()')

    predicts, targets = inference(val_loader, model)
    predicts, targets = torch.tensor(predicts), torch.tensor(targets)
    best_score, best_thresh = 0.0, 0.0

    for threshold in tqdm(np.linspace(0.05, 0.15, 33), disable=IN_KERNEL):
        score = F_score(predicts, targets, threshold=threshold).item()
        if score > best_score:
            best_score, best_thresh = score, threshold

    logger.info(f'{epoch} F2 {best_score:.4f} threshold {best_thresh:.4f}')
    logger.info(f' * F2 on validation {best_score:.4f}')
    return best_score, best_thresh

def generate_submission(val_loader: Any, test_loader: Any, model: Any,
                        epoch: int, model_path: Any) -> np.ndarray:
    score, threshold = validate(val_loader, model, epoch)
    predicts, _ = inference(test_loader, model)

    filename = f'pred_level1_{os.path.splitext(os.path.basename(model_path))[0]}'
    np.savez(filename, predicts=predicts, threshold=threshold)

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

def freeze_layers(model: Any) -> None:
    ''' Freezes all layers but the last one. '''
    m = model.module
    for layer in m.children():
        if layer not in [m.layer4, m.fc]:
            for param in layer.parameters():
                param.requires_grad = False

    # for layer in [m.fc, m.layer4[0][2].conv3, m.layer4[0][2].bn3]:
        # for param in layer.parameters():
            # param.requires_grad = True

def unfreeze_layers(model: Any) -> None:
    for layer in model.module.children():
        for param in layer.parameters():
            param.requires_grad = True


def train_model(params: Dict[str, Any]) -> float:
    np.random.seed(0)
    model_dir = opt.EXPERIMENT_DIR

    logger.info('=' * 50)
    logger.info(f'hyperparameters: {params}')

    train_loader, val_loader, test_loader = load_data(args.fold, params)
    model = create_model(args.predict, float(params['dropout']))
    # freeze_layers(model)

    # if torch.cuda.device_count() == 1:
    #     torchsummary.summary(model, (3, 224, 224))

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

    if args.weights is None:
        last_epoch = 0
        logger.info(f'training will start from epoch {last_epoch+1}')
    else:
        last_checkpoint = torch.load(args.weights)
        assert(last_checkpoint['arch']==opt.MODEL.ARCH)
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint {args.weights} was loaded.')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_epoch}')


    if args.predict:
        print('inference mode')
        generate_submission(val_loader, test_loader, model, last_epoch, args.weights)
        sys.exit(0)

    if opt.TRAIN.LOSS == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise RuntimeError('unknown loss specified')

    best_score = 0.0
    best_epoch = 0

    last_lr = read_lr(optimizer)
    best_model_path = None

    for epoch in range(last_epoch + 1, opt.TRAIN.EPOCHS + 1):
        logger.info('-' * 50)

        if not opt.TRAIN.COSINE.ENABLE:
            lr = read_lr(optimizer)
            if lr < last_lr - 1e-10 and best_model_path is not None:
                # reload the best model
                last_checkpoint = torch.load(os.path.join(model_dir, best_model_path))
                assert(last_checkpoint['arch']==opt.MODEL.ARCH)
                model.load_state_dict(last_checkpoint['state_dict'])
                optimizer.load_state_dict(last_checkpoint['optimizer'])
                logger.info(f'checkpoint {best_model_path} was loaded.')
                last_lr = read_lr(optimizer)

            if lr < opt.TRAIN.MIN_LR * 1.01:
                logger.info('reached minimum LR, stopping')
                break

                # logger.info(f'lr={lr}, start cosine annealing!')
                # set_lr(optimizer, opt.TRAIN.COSINE.LR)
                # opt.TRAIN.COSINE.ENABLE = True
                #
                # lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
                #     opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
                #     restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)

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
            best_model_path = f'{filename}_f{args.fold}_e{epoch:02d}_{score:.04f}.pth'
            save_checkpoint(data_to_save, best_model_path, model_dir)

    logger.info(f'best score: {best_score:.04f}')
    return -best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--predict', help='model to resume training', action='store_true')
    args = parser.parse_args()

    params = {'dropout': 0.3}

    opt.EXPERIMENT_DIR = os.path.join(opt.EXPERIMENT_DIR, f'fold_{args.fold}')

    if not os.path.exists(opt.EXPERIMENT_DIR):
        os.makedirs(opt.EXPERIMENT_DIR)

    logger = create_logger(os.path.join(opt.EXPERIMENT_DIR, 'log_training.txt'))
    train_model(params)
