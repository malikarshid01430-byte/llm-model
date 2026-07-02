from __future__ import annotations

from typing import Dict, List

from core.exceptions import TokenizerError


class SentencePieceStyleTokenizer:
    """A lightweight SentencePiece-style tokenizer scaffold."""

    def __init__(self, vocab: Dict[str, int] | None = None) -> None:
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, texts: List[str]) -> None:
        tokens = sorted({token for text in texts for token in text.split()})
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise TokenizerError("Tokenizer has not been fitted")
        return [self.vocab.get(token, 0) for token in text.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.inverse_vocab.get(idx, "") for idx in ids)
