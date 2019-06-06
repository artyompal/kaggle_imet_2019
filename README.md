# Solution for the iMet Collection 2019 Kaggle challenge
## Usage
Training:<br>
`./train.py --config <config.yml>`

Out-of-fold prediction:<br>
`./train.py --predict_oof --weights <model.pth>` or `./predict_all.sh` to use all pth files in the current directory.

Searching for blend coefficients:<br>
`./ensemble_search_scipy_optimize.py <ensemble_name_here> <prediction1.npy> <prediction2.npy> ...` (one fold per model, other folds will be found automatically in the same directory).

This will generate `ensemble_name_here.yml` like this: https://github.com/artyompal/imet/blob/master/best_ensemble_val_0.6397_lb_651.yml

Predicting on the test set and generating submission file: <br>
`./ensemble_inference.py <ensemble.yml>`

Generating a Kaggle kernels with submission (you should add all .pth and .yml files for every model into datasets):<br>
`./deploy_kernel.py ensemble_inference.py <ensemble.yml>`

## Description
I decided that a big number of TTAs means essentially a blend of the same model, so I chose to use many different models. I even ran out of 20 Gb space of private datasets, so I added encrypted models in public datasets (they are not being used in the best ensemble, though).

My best ensemble is 7 models with TTA x2. It's supposed to finish Stage 2 prediction in 8 hrs 40 mins:
* SE-ResNext50 at 288x288;
* SE-ResNext101 at 288x288;
* CBAM-ResNet50 at 288x288;
* PNASNet5 Large at 288x288;
* another SE-ResNext101 at 288x288 with fewer augmentations and dropout 0.3;
* two more SE-ResNext101 at 352x352 with different augmentations.

The batch size was 16. This worked better than 32 or 64, and batch accumulation 4x or 8x didn't improve the score.

I used cross-entropy loss. Focal loss and F2 loss weren't better. After I realized there's a lot of missing labels, I came up with this loss:
```
class ForgivingLoss(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.bce_loss = binary_cross_entropy()
        self.weight = weight
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(logits, labels) + self.bce_loss(logits * labels, labels) * self.weight
```

Interestingly, it worked a little better with Inception-like models, namely Xception and InceptionResNetV2, but not with ResNext-like models.

I trained 5 folds of each model. Then I used `scipy.optimize.minimize` to find the best blend coefficients.

I used Kostia's method of deployment, which packs everything into a single py-file. Also, I wrote a code which automatically searches for all available models in `../input/` and decrypts them.

What didn't work: I tried pseudo-labeling, it improved a single fold score from 611 to 622, but worked worse on LB. Probably, I generated too many labels. I would be possible to achieve gold otherwise.
