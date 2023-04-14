# -*- coding: utf-8 -*-

import sys
import os
from os import path
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
import logging

def find_ngrams(vocab: list, text: str, n: int) -> list:
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = ''.join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi


def get_app_file_path(app_name: str, filename: str) -> str:
    user_dir = path.expanduser('~')
    app_data_dir = path.join(user_dir, '.' + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    file_path = path.join(app_data_dir, filename)
    return file_path

def download_file(url: str, target: str) -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        logging.error(f"ERROR: Failed to download file from {url}. Status code: {response.status_code}")
        return False

    total_size = int(response.headers.get("Content-Length", 0))
    with open(Path(target).expanduser(), "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))

    logging.info(f"Successfully downloaded file from {url} to {target}")

    return True
