''' Learning rate schedulers. '''

import json
import torch.optim.lr_scheduler as lr_sched

from typing import Any

from cosine_scheduler import CosineLRWithRestarts


def step(optimizer, last_epoch, step_size=10, gamma=0.1, **_) -> Any:
    return lr_sched.StepLR(optimizer, step_size=step_size, gamma=gamma,
                           last_epoch=last_epoch)

def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_) -> Any:
    if isinstance(milestones, str):
        milestones = json.loads(milestones)

    return lr_sched.MultiStepLR(optimizer, milestones=milestones, gamma=gamma,
                                last_epoch=last_epoch)

def exponential(optimizer, last_epoch, gamma=0.995, **_) -> Any:
    return lr_sched.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def none(optimizer, last_epoch, **_) -> Any:
    return lr_sched.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)

def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1,
                         patience=10, threshold=0.0001, threshold_mode='rel',
                         cooldown=0, min_lr=0, **_) -> Any:
    return lr_sched.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                      patience=patience, threshold=threshold,
                                      threshold_mode=threshold_mode,
                                      cooldown=cooldown, min_lr=min_lr)

def cyclic_lr(optimizer, last_epoch, base_lr=0.001, max_lr=0.01,
              step_size_up=2000, step_size_down=None, mode='triangular',
              gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True,
              base_momentum=0.8, max_momentum=0.9, coeff=1, **_) -> Any:
    return lr_sched.CyclicLR(optimizer, base_lr=base_lr*coeff, max_lr=max_lr*coeff,
                             step_size_up=step_size_up, step_size_down=
                             step_size_down, mode=mode, gamma=gamma,
                             scale_mode=scale_mode, cycle_momentum=
                             cycle_momentum, base_momentum=base_momentum,
                             max_momentum=max_momentum, last_epoch=last_epoch)

def get_scheduler(config, optimizer, last_epoch=-1, coeff=1):
    func = globals().get(config.name)
    return func(optimizer, last_epoch, coeff=coeff, **config.params)

def is_scheduler_continuous(scheduler) -> bool:
    return type(scheduler) in [lr_sched.ExponentialLR,
                               lr_sched.CosineAnnealingLR,
                               lr_sched.CyclicLR,
                               CosineLRWithRestarts]

def get_warmup_scheduler(config, optimizer) -> Any:
    return lr_sched.CyclicLR(optimizer, base_lr=0, max_lr=config.train.warmup.max_lr,
                             step_size_up=config.train.warmup.steps, step_size_down=0,
                             cycle_momentum=False, mode='triangular')
