''' Model implementations should be placed here. '''

import os
import torch
import torch.nn as nn

from typing import Any, Dict

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

if not IN_KERNEL:
    import torchsummary
    from pytorchcv.model_provider import get_model
else:
    from model_provider import get_model


def create_model(config: Any, logger: Any, args: Any) -> Any:
    logger.info(f'creating a model {config.model.arch}')
    dropout = config.model.dropout

    model = get_model(config.model.arch, pretrained=args.weights is None)

    if config.model.arch == 'xception':
        model.features[-1].pool = nn.AdaptiveAvgPool2d(1)
    else:
        model.features[-1] = nn.AdaptiveAvgPool2d(1)

    if config.model.arch == 'pnasnet5large':
        if dropout == 0.0:
            model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output[-1].in_features, config.model.num_classes))
    elif config.model.arch == 'xception':
        if dropout < 0.1:
            model.output = nn.Linear(2048, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(2048, config.model.num_classes))
    elif config.model.arch.startswith('inception'):
        if dropout < 0.1:
            model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output[-1].in_features, config.model.num_classes))
    else:
        if dropout < 0.1:
            model.output = nn.Linear(model.output.in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output.in_features, config.model.num_classes))

    model = torch.nn.DataParallel(model).cuda()

    if args.summary:
        torchsummary.summary(model, (3, config.model.input_size, config.model.input_size))

    return model

def freeze_layers(model: Any) -> None:
    ''' Freezes all layers but the last one. '''
    m = model.module
    for layer in m.children():
        for param in layer.parameters():
            param.requires_grad = False

    for layer in model.module.output.children():
        for param in layer.parameters():
            param.requires_grad = True

def unfreeze_layers(model: Any) -> None:
    for layer in model.module.children():
        for param in layer.parameters():
            param.requires_grad = True
