from __future__ import annotations

from typing import Dict, List

from core.exceptions import TokenizerError


class CharacterTokenizer:
    """A simple character-level tokenizer for educational use."""

    def __init__(self, vocab: Dict[str, int] | None = None) -> None:
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, texts: List[str]) -> None:
        chars = sorted({char for text in texts for char in text})
        self.vocab = {char: idx for idx, char in enumerate(chars)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise TokenizerError("Tokenizer has not been fitted")
        return [self.vocab.get(char, self.vocab.get("<unk>", 0)) for char in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inverse_vocab.get(idx, "") for idx in ids)
