''' Data loaders for training & validation. '''

import math, os, pickle
from collections import defaultdict
from glob import glob
from typing import *

import numpy as np, pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from debug import dprint

SAVE_DEBUG_IMAGES = False
VERSION = os.path.basename(__file__)[12:-3]


class Dataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, path: str, mode: str, num_classes: int,
                 image_size: int = 0, oversample: int = 1, augmentor: Any = None,
                 resize: bool = True, num_tta: int = 1, inception: bool = False) -> None:
        print(f'creating data loader {VERSION} in mode={mode}')
        assert mode in ['train', 'val', 'test']
        assert mode != 'train' or num_tta == 1

        self.df = dataframe
        self.path = path
        self.mode = mode
        self.num_classes = num_classes
        self.image_size = image_size
        self.augmentor = augmentor
        self.resize = resize
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

    def _transform_image(self, image: Image, index: int) -> torch.tensor:
        image = np.array(image)

        if self.augmentor is not None:
            image = self.augmentor(image=image)['image']

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

        # print(image.size())
        targets = np.zeros(self.num_classes, dtype=np.float32)

        if self.mode != 'test':
            labels = list(map(int, self.df.iloc[index, 1].split()))
            targets[labels] = 1

        return image, targets

    def __len__(self) -> int:
        count = self.df.shape[0]

        if self.mode == 'train':
            count -= count % 32

        return count
