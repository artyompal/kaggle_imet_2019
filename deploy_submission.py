#!/usr/bin/python3.6
import base64, gzip, os, sys
from pathlib import Path


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')

if __name__ == '__main__':
    assert len(sys.argv) == 2
    to_encode = [Path(sys.argv[1])]
    file_data = {'submission.csv': encode_file(path) for path in to_encode}
    template = Path('script_template.py').read_text('utf8')
    Path(os.path.basename(sys.argv[1]) + '.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')

