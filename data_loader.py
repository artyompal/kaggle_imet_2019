''' Data loaders for training & validation. '''

import math
import os
import pickle
import random

from collections import defaultdict
from glob import glob
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image


DEBUG_SAVE_IMAGES = False
DEBUG_SAVE_DATASET = False


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str, config: Any,
                 augmentor: Any = None, aug_type: str = 'albu') -> None:
        print(f'creating data_loader for {config.version} in mode={mode}')
        assert mode in ['train', 'val', 'test']
        assert config.test.batch_size % config.test.num_ttas == 0

        self.df = dataframe
        self.df['extra'] = False

        self.mode = mode
        self.augmentor = augmentor
        self.aug_type = aug_type

        self.version = config.version
        self.path = '../input/train/' if mode != 'test' else '../input/test/'
        self.num_classes = config.model.num_classes
        self.input_size = config.model.input_size
        self.rect_crop = config.data.rect_crop
        self.num_tta = config.test.num_ttas if mode == 'test' else 1
        self.batch_size = config.train.batch_size if mode != 'test' else \
                          config.test.batch_size // config.test.num_ttas

        if config.train.use_arbitrary_sizes:
            align = 32
            self.df['rounded_w'] = self.df.w - self.df.w % align
            self.df['rounded_h'] = self.df.h - self.df.h % align
            self.df['size'] = self.df.rounded_w * 1000000 + self.df.rounded_h

            self.create_batches()
        else:
            self.batches = self.df

        if 'ception' in config.model.arch:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])

    def create_batches(self) -> None:
        ''' Groups samples into batches of the predefined size. '''
        dfs = []

        for size, group in self.df.groupby('size'):
            gap = (self.batch_size - len(group) % self.batch_size) % self.batch_size

            if gap:
                extra = group.sample(gap, replace = len(group) < gap)
                extra['extra'] = True
                group = pd.concat([group, extra])

            dfs.append(group)

        random.shuffle(dfs)
        self.batches = pd.concat(dfs)

        if DEBUG_SAVE_DATASET:
            self.batches.to_csv(f'dataset_{self.mode}.csv')

    def _transform_image(self, image: Image, index: int) -> torch.Tensor:
        ''' Augments the image. '''
        image = np.array(image)

        if self.rect_crop.enable:
            dims = image.shape[:2]
            biggest_size, smallest_size = max(dims), min(dims)

            assert self.rect_crop.min_ratio <= self.rect_crop.max_ratio
            assert self.rect_crop.max_ratio <= 1.0

            ratio = random.uniform(self.rect_crop.min_ratio, self.rect_crop.max_ratio)
            crop_big_size = min(int(self.input_size / ratio), biggest_size)
            crop_small_size = min(self.input_size, smallest_size)

            crop_size = (crop_big_size, crop_small_size) if biggest_size == dims[0] \
                        else (crop_small_size, crop_big_size)

            y = int(random.uniform(0, dims[0] - crop_size[0]))
            x = int(random.uniform(0, dims[1] - crop_size[1]))
            image = image[y : y + crop_size[0], x : x + crop_size[1]]

            if self.rect_crop.scale_both_dims:
                new_size = (self.input_size, self.input_size)
            else:
                new_size = (min(crop_size[0], self.input_size),
                            min(crop_size[1], self.input_size))

            # print(f'dims were {dims[0]}x{dims[1]}, crop {crop_size[0]}' +
            #       f'x{crop_size[1]}, scaled into {new_size[0]}x{new_size[1]}')

            image = np.array(Image.fromarray(image).resize(new_size, Image.BICUBIC))

        if self.augmentor is not None:
            if self.aug_type == 'albu':
                image = self.augmentor(image=image)['image']
            elif self.aug_type == 'imgaug':
                image = self.augmentor.augment_image(image)

        if DEBUG_SAVE_IMAGES:
            os.makedirs(f'../debug_images_{self.version}/', exist_ok=True)
            Image.fromarray(image).save(f'../debug_images_{self.version}/{index}.png')

        return self.transforms(image)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) or just sample. '''
        filename = self.batches.iloc[index, 0]
        image = Image.open(os.path.join(self.path, filename + '.png'))
        assert image.mode == 'RGB'

        if self.num_tta == 1:
            image = self._transform_image(image, index)
        else:
            image = torch.stack([self._transform_image(image, index) for _ in range(self.num_tta)])

        if self.mode != 'test':
            targets = np.zeros(self.num_classes, dtype=np.float32)
            labels = list(map(int, self.batches.iloc[index, 1].split()))
            targets[labels] = 1
            return image, targets
        else:
            return image

    def __len__(self) -> int:
        return self.batches.shape[0]
