#!/usr/bin/python3.6
import base64
import gzip
import os
import re
import sys

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
        'train.py',
        'utils.py',

        script_py,
        ]

    to_encode.extend(glob('models/*.py'))
    to_encode.extend(glob('albumentations/*.py'))
    to_encode.extend(glob('albumentations/**/*.py'))
    to_encode.extend(arg for arg in sys.argv[2:] if os.path.exists(arg))

    file_data = {path: encode_file(path) for path in to_encode}
    printed_data = ',\n'.join([f'"{filename}": "{content}"' for filename, content in
                              file_data.items()])

    with open('script_template.py') as f:
        template = f.read()
        template = template.replace('{file_data}', '{' + printed_data + '}')

    cmd_line = ' '.join(sys.argv[1:])
    dest_name = re.sub(r'[/ ]', '_', cmd_line)
    dest_name = f'kernel_{dest_name}.py'

    with open(dest_name, 'w') as f:
        f.write(template)
        f.write('\nos.system(\'export PYTHONPATH=${PYTHONPATH}:/kaggle/working && python ' + cmd_line + '\')\n\n')
