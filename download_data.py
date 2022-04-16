"""
Download and unpack video stabilization dataset
"""

__author__ = 'Ian Randman'

import os
from tqdm import tqdm
import requests
import shutil
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ARCHIVE_PATH = os.path.join(DATA_DIR, 'DeepStab.zip')
DATASET_DIR = os.path.join(DATA_DIR, 'DeepStab')

URL = 'https://cg.cs.tsinghua.edu.cn/download/DeepStab.zip'


def download_and_unpack_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    # if the zip exists, try to unpack it
    retry = False
    if os.path.exists(ARCHIVE_PATH):
        try:
            shutil.unpack_archive(ARCHIVE_PATH, DATA_DIR)
        except (EOFError, shutil.ReadError):
            print('Cannot unpack. Archive is corrupt. Attempting to retry...')
            retry = True

    # if the zip does not exist or is corrupt (unpacking failed), download it, then unpack it
    if not os.path.exists(ARCHIVE_PATH) or retry:
        print(f'Downloading from {URL}...')

        # download file as stream
        response = requests.get(URL, stream=True)
        with open(ARCHIVE_PATH, 'wb') as handle:
            progress_bar = tqdm(unit="B", total=int(response.headers['Content-Length']), unit_scale=True, unit_divisor=1024)
            for data in response.iter_content(chunk_size=8192):
                progress_bar.update(len(data))
                handle.write(data)
            progress_bar.close()

        # try to unpack zip
        print(f'Attempting to unpack {ARCHIVE_PATH}...')
        try:
            shutil.unpack_archive(ARCHIVE_PATH, DATA_DIR)
        except EOFError:
            # give up if unpacking failed
            print('Archive cannot be unpacked')
            sys.exit(1)

    print('Successfully downloaded and unpacked model')
    os.remove(ARCHIVE_PATH)
    print('Deleted archive file')


if __name__ == '__main__':
    download_and_unpack_data()
