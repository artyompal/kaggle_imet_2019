#!/usr/bin/python3.6
import gzip
import base64
import os
from pathlib import Path
from typing import Dict
from glob import glob

# this is base64 encoded source code
file_data: Dict = {file_data}

for path, encoded in file_data.items():
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    # print('unpacking', path)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))

print('file list after extraction')
print(list(glob('**/*.py', recursive=True)))
