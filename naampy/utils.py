import logging
import os
from os import path
from pathlib import Path

import requests
from tqdm import tqdm


def find_ngrams(vocab: list, text: str, n: int) -> list:
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab: Vocabulary list.
        text: Input text
        n: N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    a = zip(*[text[i:] for i in range(n)], strict=False)
    for i in a:
        w = "".join(i)
        try:
            idx = vocab.index(w)
        except ValueError:
            idx = 0
        wi.append(idx)
    return wi


def get_app_file_path(app_name: str, filename: str) -> str:
    """Get the file path for app data storage in the user's home directory.

    Creates application data directory if it doesn't exist and returns the
    full path to the specified filename within that directory.

    Args:
        app_name: Name of the application (used to create .app_name directory)
        filename: Name of the file to store in the app directory

    Returns:
        str: Full path to the file in the application data directory
    """
    user_dir = path.expanduser("~")
    app_data_dir = path.join(user_dir, "." + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    file_path = path.join(app_data_dir, filename)
    return file_path


def download_file(url: str, target: str) -> bool:
    """Download a file from a URL with progress tracking.

    Downloads a file from the given URL to the target location with a progress bar.
    Handles HTTP errors and provides logging for success/failure.

    Args:
        url: URL to download the file from
        target: Local file path where the downloaded file should be saved

    Returns:
        bool: True if download was successful, False otherwise
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        logging.error(
            f"ERROR: Failed to download file from {url}. Status code: {response.status_code}"
        )
        return False

    total_size = int(response.headers.get("Content-Length", 0))
    with open(Path(target).expanduser(), "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))

    logging.info(f"Successfully downloaded file from {url} to {target}")

    return True
