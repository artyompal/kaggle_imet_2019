''' Reads config file and merges settings with default ones. '''

import multiprocessing
import os
import re
import yaml

import torch

from typing import Any
from easydict import EasyDict as edict

from debug import dprint


def _get_default_config(filename: str, args: Any) -> edict:
    cfg = edict()
    cfg.in_kernel = False
    cfg.version = os.path.splitext(os.path.basename(filename))[0]
    cfg.experiment_dir = f'../models/{cfg.version}/fold_{args.fold}/'
    cfg.num_workers = min(12, multiprocessing.cpu_count())

    cfg.data = edict()
    cfg.data.train_dir = '../input/train/'
    cfg.data.test_dir = '../input/test/'
    cfg.data.train_filename = None
    cfg.data.params = edict()

    cfg.model = edict()
    cfg.model.arch = 'resnet50'
    cfg.model.image_size = 0
    cfg.model.input_size = 0
    cfg.model.num_classes = None
    cfg.model.num_folds = 5
    cfg.model.bottleneck_fc = None
    cfg.model.dropout = 0

    cfg.train = edict()
    cfg.train.batch_size = 32 * torch.cuda.device_count()
    cfg.train.num_epochs = 1000
    cfg.train.shuffle = True
    cfg.train.images_per_class = None
    cfg.train.max_steps_per_epoch = None
    cfg.train.log_freq = 100
    cfg.train.min_lr = 3e-7
    cfg.train.lr_scheduler = None
    cfg.train.use_balancing_sampler = False

    cfg.val = edict()
    cfg.val.images_per_class = None

    cfg.test = edict()
    cfg.test.batch_size = 64 * torch.cuda.device_count()
    cfg.test.num_ttas = 1
    cfg.test.num_predicts = 5

    cfg.optimizer = edict()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.params = edict()

    cfg.scheduler = edict()
    cfg.scheduler.name = ''
    cfg.scheduler.params = edict()

    cfg.scheduler2 = edict()
    cfg.scheduler2.name = ''
    cfg.scheduler2.params = edict()

    cfg.loss = edict()
    cfg.loss.name = 'none'

    cfg.augmentations = edict()
    cfg.augmentations.global_prob = 1.0

    cfg.augmentations.hflip = False
    cfg.augmentations.vflip = False
    cfg.augmentations.rotate90 = False
    cfg.augmentations.affine = 'none'

    cfg.augmentations.rect_crop = edict()
    cfg.augmentations.rect_crop.enable = False
    cfg.augmentations.rect_crop.rect_min_area = 0.1
    cfg.augmentations.rect_crop.rect_min_ratio = 0.75

    cfg.augmentations.noise = 0
    cfg.augmentations.blur = 0
    cfg.augmentations.distortion = 0
    cfg.augmentations.color = 0

    return cfg

def _merge_config(src: edict, dst: edict) -> edict:
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v

def load(config_path: str, args: Any) -> edict:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_path) as f:
        yaml_config = edict(yaml.load(f, Loader=loader))

    config = _get_default_config(config_path, args)
    _merge_config(yaml_config, config)

    return config
