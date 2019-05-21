''' Data loaders for training & validation. '''

import os
import random
import albumentations as albu
import numpy as np

from PIL import Image


class RandomAlignedCrop(albu.DualTransform):
    def __init__(self, always_apply=True, p=1.0) -> None:
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        h, w, _ = img.shape
        align = 32
        new_h = min(h - h % align, 1024)
        new_w = min(w - w % align, 1024)
        gap_h, gap_w = h - new_h, w - new_w
        y = random.randrange(gap_h) if gap_h else 0
        x = random.randrange(gap_w) if gap_w else 0

        img = img[y : y + h - gap_h, x : x + w - gap_w]
        assert img.shape[0] % align == 0 and img.shape[1] % align == 0
        # print('cropped image from', (h, w, 3), 'to', img.shape)
        return img

    def get_params(self):
        return {'x': 0}

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint
