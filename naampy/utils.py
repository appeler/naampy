# -*- coding: utf-8 -*-

import sys
import os
from os import path
import requests
from tqdm import tqdm


def isstring(s):
    # if we use Python 3
    if (sys.version_info[0] >= 3):
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def column_exists(df, col):
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        print("The specify column `{0!s}` not found in the input file"
              .format(col))
        return False
    else:
        return True


def fixup_columns(cols):
    """Replace index location column to name with `col` prefix

    Args:
        cols (list): List of original columns

    Returns:
        list: List of column names

    """
    out_cols = []
    for col in cols:
        if type(col) == int:
            out_cols.append('col{:d}'.format(col))
        else:
            out_cols.append(col)
    return out_cols


def find_ngrams(vocab, text, n):
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

    if not isstring(text):
        return wi

    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = ''.join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi


def get_app_file_path(app_name, filename):
    user_dir = path.expanduser('~')
    app_data_dir = path.join(user_dir, '.' + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    file_path = path.join(app_data_dir, filename)
    return file_path


def download_file(url, target):

    headers = {}

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, headers=headers)

    if r.status_code == 200:
        chunk_size = (64 * 1024)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0)) / chunk_size

        total_size += 1

        with open(target, 'wb') as f:
            for data in tqdm(r.iter_content(chunk_size), total=round(total_size, 1), unit_scale=chunk_size/1024, unit='KB'):
                f.write(data)
        return True
    else:
        print("ERROR: status_code={0:d}".format(r.status_code))
        return False
