''' Add all the necessary metrics here. '''

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict


def cross_entropy() -> Any:
    return torch.nn.CrossEntropyLoss()

def binary_cross_entropy() -> Any:
    return torch.nn.BCEWithLogitsLoss()

def mse_loss() -> Any:
    return torch.nn.MSELoss()

def l1_loss() -> Any:
    return torch.nn.L1Loss()

def smooth_l1_loss() -> Any:
    return torch.nn.SmoothL1Loss()

class FocalLoss(nn.Module):
    def __init__(self, gamma : int = 2) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

class FScoreLoss(nn.Module):
    def __init__(self, beta: int) -> None:
        super().__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.size() == labels.size()

        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss

def focal_loss() -> Any:
    return FocalLoss()

def f2_loss() -> Any:
    return FScoreLoss(beta = 2)

def get_loss(config: edict) -> Any:
    f = globals().get(config.loss.name)
    return f()
