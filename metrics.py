''' Add all the necessary metrics here. '''

import torch
from debug import dprint

def F_score(predict: torch.Tensor, labels: torch.Tensor, beta: int,
            threshold: float = 0.5) -> float:
    if not isinstance(predict, torch.Tensor):
        predict = torch.tensor(predict)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if predict.shape != labels.shape:
        dprint(predict.shape)
        dprint(labels.shape)
        assert False

    predict = predict > threshold
    labels = labels > threshold

    TP = (predict & labels).sum(1).float()
    TN = ((~predict) & (~labels)).sum(1).float()
    FP = (predict & (~labels)).sum(1).float()
    FN = ((~predict) & labels).sum(1).float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0).item()

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Computes GAP@1 '''
    if len(predicts.shape) != 1:
        dprint(predicts.shape)
        assert False

    if len(confs.shape) != 1:
        dprint(confs.shape)
        assert False

    if len(targets.shape) != 1:
        dprint(targets.shape)
        assert False

    assert predicts.shape == confs.shape and confs.shape == targets.shape

    sorted_confs, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res
