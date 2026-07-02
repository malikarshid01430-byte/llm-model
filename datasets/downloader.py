from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List

from core.logging import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    """A simple downloader utility for educational datasets."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_text(self, url: str, filename: str) -> Path:
        destination = self.output_dir / filename
        logger.info("Downloading %s -> %s", url, destination)
        urllib.request.urlretrieve(url, destination)
        return destination

    def load_local_texts(self, pattern: str = "*.txt") -> List[str]:
        return [
            path.read_text(encoding="utf-8") for path in self.output_dir.glob(pattern)
        ]
