#!/usr/bin/python3.6

import base64
import gzip
import os
import re
import sys
import yaml

from pathlib import Path
from glob import glob
from debug import dprint


def encode_file(path: str) -> str:
    if path == 'easydict.py':   # I need this hack to fool mypy linter
        path = 'easydict__.py'

    compressed = gzip.compress(Path(path).read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')

if __name__ == '__main__':
    if len(sys.argv) < 2 or not sys.argv[1].endswith('.py'):
        print(f'usage: {sys.argv[0]} <script.py> [arguments...]')
        sys.exit()

    script_py = sys.argv[1]

    to_encode = [
        'cosine_scheduler.py',
        'data_loader.py',
        'debug.py',
        'easydict.py',
        'folds.npy',
        'losses.py',
        'metrics.py',
        'model_provider.py',
        'model.py',
        'optimizers.py',
        'parse_config.py',
        'random_erase.py',
        'random_rect_crop.py',
        'schedulers.py',
        'senet.py',
        'train.py',
        'utils.py',

        'ensemble_gen_submission.py',
        script_py,
        ]

    to_encode.extend(glob('models/*.py'))
    to_encode.extend(glob('albumentations/*.py'))
    to_encode.extend(glob('albumentations/**/*.py'))
    to_encode.extend(arg for arg in sys.argv[2:] if os.path.exists(arg))

    # parse ensemble YAML file to find necessary model configs
    for arg in sys.argv[2:]:
        if arg.endswith('.yml') and os.path.exists(arg):
            with open(arg) as f:
                ensemble = yaml.load(f, Loader=yaml.SafeLoader)

            for predicts in ensemble:
                for pred in predicts['predicts']:
                    m = re.match(r'level1_train_(.*)_f(\d)_e(\d+)_([.0-9]+)\.npy',
                                 os.path.basename(pred))
                    if not m:
                        print('could not parse filename', pred)
                        assert m

                    config = f'config/{m.group(1)}.yml'
                    to_encode.append(config)

    # print('encoding files', to_encode)

    file_data = {path: encode_file(path) for path in to_encode}
    printed_data = ',\n'.join([f'"{filename}": "{content}"' for filename, content in
                              file_data.items()])

    with open('script_template.py') as f:
        template = f.read()
        template = template.replace('{file_data}', '{' + printed_data + '}')

    cmd_line = ' '.join(sys.argv[1:])
    dest_name = re.sub(r'[/ .]', '_', cmd_line)
    dest_name = f'kernel_{dest_name}.py'

    with open(dest_name, 'w') as f:
        f.write(template)
        f.write('\nos.system(\'export PYTHONPATH=${PYTHONPATH}:/kaggle/working && python ' + cmd_line + '\')\n\n')
