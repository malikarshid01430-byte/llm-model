from __future__ import annotations

from pathlib import Path
from typing import List

from preprocessing.text_processor import TextProcessor


class DocumentLoader:
    """Load plain text documents from disk."""

    def __init__(self) -> None:
        self.processor = TextProcessor()

    def load(self, file_path: str | Path) -> str:
        return self.processor.clean(Path(file_path).read_text(encoding="utf-8"))

    def load_many(self, paths: List[str | Path]) -> List[str]:
        return [self.load(path) for path in paths]
