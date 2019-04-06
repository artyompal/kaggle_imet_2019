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

SAVE_DEBUG_IMAGES = False
VERSION = os.path.basename(__file__)[12:-3]

class Dataset(data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, path: str, mode: str,
                 image_size: int = 0, oversample: int = 1, augmentor: Any = None,
                 resize: bool = True) -> None:
        print('creating data loader', VERSION)
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.path = path
        self.mode = mode
        self.image_size = image_size
        self.augmentor = augmentor
        self.resize = resize

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        filename = self.df.iloc[index, 0]
        image = Image.open(os.path.join(self.path, filename + '.png'))
        assert image.mode == 'RGB'

        if self.resize:
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        # else:
        #     old_size = sample.size
        #     assert old_size[0] <= self.image_size and old_size[1] <= self.image_size
        #     new_im = Image.new('RGB', (self.image_size, self.image_size))
        #     new_im.paste(sample, ((self.image_size - old_size[0]) // 2,
        #                           (self.image_size - old_size[1]) // 2))

        # image = np.array(sample)
        # assert image.dtype == np.uint8
        # assert image.shape == (self.image_size, self.image_size, 3)

        if self.mode == 'train' and self.augmentor is not None:
            image = self.augmentor(image)

        if SAVE_DEBUG_IMAGES:
            os.makedirs(f'../debug_images_{VERSION}/', exist_ok=True)
            Image.fromarray(image).save(f'../debug_images_{VERSION}/{index}.png')

        image = self.transforms(image)

        if self.mode == 'test':
            return image, ''
        else:
            return image, self.df.iloc[index, 1]

    def __len__(self) -> int:
        count = self.df.shape[0]

        if self.mode == 'train':
            count -= count % 32

        return count
