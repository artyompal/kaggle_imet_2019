''' Data loaders for training & validation. '''

import math
import os
import pickle

from collections import defaultdict
from glob import glob
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image

SAVE_DEBUG_IMAGES = False
VERSION = os.path.splitext(os.path.basename(__file__))[0]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, path: str, mode: str,
                 num_classes: int, image_size: int = 0, resize: bool = True,
                 augmentor: Any = None, aug_type: str = 'albu',
                 num_tta: int = 1, inception: bool = False) -> None:
        print(f'creating {VERSION} in mode={mode}')
        assert mode in ['train', 'val', 'test']
        assert mode != 'train' or num_tta == 1

        self.df = dataframe
        self.path = path
        self.mode = mode
        self.num_classes = num_classes
        self.image_size = image_size
        self.resize = resize
        self.augmentor = augmentor
        self.aug_type = aug_type
        self.num_tta = num_tta

        if not inception:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
            ])

    def _transform_image(self, image: Image, index: int) -> torch.Tensor:
        image = np.array(image)

        if self.augmentor is not None:
            if self.aug_type == 'albu':
                image = self.augmentor(image=image)['image']
            elif self.aug_type == 'imgaug':
                image = self.augmentor.augment_image(image)

        if SAVE_DEBUG_IMAGES:
            os.makedirs(f'../debug_images_{VERSION}/', exist_ok=True)
            Image.fromarray(image).save(f'../debug_images_{VERSION}/{index}.png')

        return self.transforms(image)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        filename = self.df.iloc[index, 0]
        image = Image.open(os.path.join(self.path, filename + '.png'))
        assert image.mode == 'RGB'

        if self.resize:
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.num_tta == 1:
            image = self._transform_image(image, index)
        else:
            image = torch.stack([self._transform_image(image, index) for _ in range(self.num_tta)])

        if self.mode != 'test':
            targets = np.zeros(self.num_classes, dtype=np.float32)
            labels = list(map(int, self.df.iloc[index, 1].split()))
            targets[labels] = 1
            return image, targets
        else:
            return image

    def __len__(self) -> int:
        return self.df.shape[0]
