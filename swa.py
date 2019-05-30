#!/usr/bin/python3.6
''' Averages checkpoints using SWA. '''

import argparse

from glob import glob
from impl_swa import moving_average, bn_update

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--coeff', help='moving average coefficient', type=float, required=True)
    parser.add_argument('path', help='models path', type=str)
    args = parser.parse_args()

    files = glob(os.path.join(args.path, '*.pth'))
