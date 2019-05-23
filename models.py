''' Model implementations should be placed here. '''

import os
import torch
import torch.nn as nn

from typing import Any, Dict
from debug import dprint

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

if not IN_KERNEL:
    import torchsummary
    from pytorchcv.model_provider import get_model
else:
    from model_provider import get_model


class Model(nn.Module):
    ''' Network that supports arbitrary resolutions by iterating over tiles. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()

        self.model = get_model(config.model.arch, pretrained=pretrained)
        self.model.features[-1] = nn.AdaptiveAvgPool2d(1)

        self.tile_size = config.model.input_size
        self.pool = nn.AdaptiveAvgPool1d(1)

        if config.model.dropout == 0:
            self.output = nn.Linear(self.model.output.in_features, config.model.num_classes)
        else:
            self.output = nn.Sequential(
                 nn.Dropout(config.model.dropout),
                 nn.Linear(self.model.output.in_features, config.model.num_classes))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        ''' Iterate through all tiles of the fixed size, then pool,
        then invoke the head. '''
        h, w = input_.shape[-2:]
        size = self.tile_size

        assert h >= size and w >= size
        outputs = []

        for y in range(0, h, size):
            for x in range(0, w, size):
                y2, x2 = min(h, y + size), min(w, x + size)
                y1, x1 = y2 - size, x2 - size
                assert y1 >= 0 and x1 >= 0

                data = input_[:, :, y1:y2, x1:x2]
                features = self.model.features(data)
                outputs.append(features.view(features.size(0), features.size(1), 1))

        features = torch.cat(outputs, dim=2)
        features = self.pool(features)
        features = features.view(features.size(0), -1)

        return self.output(features)

def create_model(config: Any, logger: Any, args: Any) -> Any:
    logger.info(f'creating a model {config.model.arch}')
    model = Model(config, pretrained=args.weights is None)
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
