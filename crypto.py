#!/usr/bin/python3.6
''' This module implements some simple functions to encode/decode files. '''

import argparse
import hashlib
import os
import struct

from base64 import b32encode, b32decode
from Crypto.Cipher import AES
from debug import dprint

password = 'JirOlm5knaw'
key = hashlib.sha256(password.encode()).digest()

IV = 16 * '\x00'           # Initialization vector: discussed later
mode = AES.MODE_CBC


def _pad(s: bytes) -> bytes:
    return s + (' ' * (AES.block_size - len(s) % AES.block_size)).encode()

# def encrypt_string(message: str) -> str:
#     msg = _pad(message.encode())
#     cipher = AES.new(key, mode, IV=IV)
#     enc = cipher.encrypt(msg)
#     return b32encode(enc).decode()
#
# def decrypt_string(ciphertext: str) -> str:
#     enc = b32decode(ciphertext.encode())
#     cipher = AES.new(key, mode, IV=IV)
#     plaintext = cipher.decrypt(enc)
#     return plaintext.rstrip()

def encrypt(message):
    message = _pad(message)
    cipher = AES.new(key, mode, IV=IV)
    return cipher.encrypt(message)

def decrypt(ciphertext):
    cipher = AES.new(key, mode, IV=IV)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext.rstrip(b"\0")

def encrypt_file(file_name, out_name=None):
    if out_name == None:
        out_name = file_name + '.enc'

    with open(file_name, 'rb') as fo:
        plaintext = fo.read()

    enc = encrypt(plaintext)

    with open(out_name, 'wb') as fo:
        filesize = os.path.getsize(file_name)
        fo.write(struct.pack('<Q', filesize))
        fo.write(enc)

def decrypt_file(file_name, out_name=None):
    if out_name == None:
        assert file_name.endswith('.enc')
        out_name = file_name[:-4]

    with open(file_name, 'rb') as fo:
        origsize = struct.unpack('<Q', fo.read(struct.calcsize('Q')))[0]
        ciphertext = fo.read()

    dec = decrypt(ciphertext)

    with open(out_name, 'wb') as fo:
        fo.write(dec)
        fo.truncate(origsize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='file to encode or decode', type=str)
    args = parser.parse_args()

    if not args.filename.endswith('.enc'):
        encrypt_file(args.filename)
    else:
        decrypt_file(args.filename)
