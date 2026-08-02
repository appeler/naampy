"""Shared helpers: app data paths and file downloads."""

import logging
import os
import tempfile
from os import path

import requests
from tqdm import tqdm


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

    Downloads to a temporary file alongside the target and renames it into place
    only once the transfer completes, so an interrupted download can never leave a
    truncated file that later runs would mistake for a valid cache entry.

    Args:
        url: URL to download the file from
        target: Local file path where the downloaded file should be saved

    Returns:
        bool: True if download was successful, False otherwise
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    target = path.expanduser(target)
    tmp_path = None
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        fd, tmp_path = tempfile.mkstemp(dir=path.dirname(target) or ".", suffix=".part")
        written = 0
        with (
            os.fdopen(fd, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                written += len(data)
                pbar.update(len(data))

        if total_size and written != total_size:
            logging.error(
                f"ERROR: Truncated download from {url}: got {written} bytes, "
                f"expected {total_size}"
            )
            return False

        os.replace(tmp_path, target)
        tmp_path = None
    except (requests.RequestException, OSError) as exc:
        logging.error(f"ERROR: Failed to download file from {url}: {exc}")
        return False
    finally:
        if tmp_path is not None and path.exists(tmp_path):
            os.remove(tmp_path)

    logging.info(f"Successfully downloaded file from {url} to {target}")

    return True
