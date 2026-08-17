"""Shared helpers: app data paths and file downloads."""

import logging
import os
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


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
    app_data_dir = Path.home() / f".{app_name}"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    return str(app_data_dir / filename)


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

    target_path = Path(target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, suffix=".part")
        tmp_path = Path(tmp_name)
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
            LOGGER.error(
                "Truncated download from %s: got %s bytes, expected %s",
                url,
                written,
                total_size,
            )
            return False

        tmp_path.replace(target_path)
        tmp_path = None
    except (requests.RequestException, OSError) as exc:
        LOGGER.error("Failed to download file from %s: %s", url, exc)
        return False
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    LOGGER.info("Successfully downloaded file from %s to %s", url, target_path)

    return True
