#!/usr/bin/python3.6
import base64, gzip, os, sys
from pathlib import Path
from glob import glob


def encode_file(path: str) -> str:
    if path == 'easydict.py':   # I need this hack to fool mypy linter
        path = 'easydict__.py'

    compressed = gzip.compress(Path(path).read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')

if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.py'):
        print(f'usage: {sys.argv[0]} <script.py>')
        sys.exit()

    script_py = sys.argv[1]

    # TODO: infer this by parsing the file
    to_encode = [
        # common files
        'easydict.py',
        'data_loader_v1_single.py',
        'data_loader_v2_albu.py',
        'utils.py',
        'debug.py',
        'cosine_scheduler.py',
        'folds.npy',

        # models
        'senet.py',
        'model_provider.py',

        # prediction scripts
#         'train_2b_se_resnext50.py',
#         'train_2i_se_resnext101_auto_aug.py',
#         'train_2j_cbam_resnet50_auto_aug.py',
#         'train_2k_pnasnet_auto_aug.py',
        'train_2o_se_resnext101_def_aug.py',
        'train_4b_se_resnext101_352x352.py',
        'train_4f_se_resnext101_352x352_aug2.py',

        # ensemble scripts
        'blend.py',
        script_py,
        ]

    to_encode.extend(list(glob('**/*.py')))
    to_encode.extend(list(glob('albumentations/**/*.py')))

    file_data = {path: encode_file(path) for path in to_encode}
    printed_data = ',\n'.join([f'"{filename}": "{content}"' for filename, content in
                              file_data.items()])

    with open('script_template.py') as f:
        template = f.read()
        template = template.replace('{file_data}', '{' + printed_data + '}')

    dest_name = 'sub_' + os.path.basename(script_py)
    with open(dest_name, 'w') as f:
        f.write(template)
        f.write('\nos.system(\'export PYTHONPATH=${PYTHONPATH}:/kaggle/working && python ' + script_py + '\')\n\n')
