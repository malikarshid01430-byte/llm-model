from __future__ import annotations

import re
from typing import List


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunk = normalized[start:end]
        chunks.append(chunk)
        start = end
    return chunks
