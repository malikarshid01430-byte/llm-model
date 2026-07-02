from __future__ import annotations

import re
from typing import List

from core.logging import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """Reusable preprocessing pipeline for text corpora."""

    def __init__(self) -> None:
        self.logger = logger

    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", self.normalize(text))
        return [sentence for sentence in sentences if sentence]

    def clean(self, text: str) -> str:
        text = re.sub(r"[^\w\s.,!?;:'\-()]+", " ", text)
        return self.normalize(text)
