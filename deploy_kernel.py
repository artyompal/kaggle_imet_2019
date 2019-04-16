#!/usr/bin/python3.6
import base64, gzip, os, sys
from pathlib import Path

def encode_file(path: str) -> str:
    if path == 'easydict.py':
        path = 'easydict__.py'

    compressed = gzip.compress(Path(path).read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')

if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].endswith('.py'):
        print(f'usage: {sys.argv[0]} <script.py>')
        sys.exit()

    script_py = sys.argv[1]

    # TODO: infer this by parsing the file
    to_encode = ['easydict.py', 'data_loader_v1_single.py', 'utils.py', 'debug.py',
                 'cosine_scheduler.py', 'senet.py', 'folds.npy', 'blend.py',
                 'train_2a_resnet50.py', 'train_2b_se_resnext50.py', script_py]

    file_data = {path: encode_file(path) for path in to_encode}
    printed_data = ',\n'.join([f'"{filename}": "{content}"' for filename, content in
                              file_data.items()])

    with open('script_template.py') as f:
        template = f.read()
        template = template.replace('{file_data}', '{' + printed_data + '}')

    dest_name = 'sub_' + os.path.basename(script_py)
    with open(dest_name, 'w') as f:
        f.write(template)
        f.write('\nos.system(\'export PYTHONPATH=${PYTHONPATH}:/kaggle/working && python ' + script_py + '\')')
