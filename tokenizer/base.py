from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List


class BPETokenizer:
    """A lightweight byte-pair-encoding tokenizer implemented for education and experimentation."""

    def __init__(
        self,
        vocab_size: int = 2000,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        eos_token: str = "</s>",
    ) -> None:
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = [unk_token, pad_token, eos_token]
        self._initialized = False

    def fit(self, texts: List[str]) -> None:
        token_counter: Counter[str] = Counter()
        for text in texts:
            for token in self._initial_tokens(text):
                token_counter[token] += 1

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        symbols = list(self.vocab.keys())
        current_vocab = {token: [token] for token in symbols}
        for token in self._initial_tokens(" ".join(texts)):
            if token not in current_vocab:
                current_vocab[token] = [token]

        while len(self.vocab) < self.vocab_size:
            pairs: Counter[tuple[str, str]] = Counter()
            for token, pieces in current_vocab.items():
                if len(pieces) < 2:
                    continue
                for i in range(len(pieces) - 1):
                    pair = (pieces[i], pieces[i + 1])
                    pairs[pair] += 1
            if not pairs:
                break
            best_pair, _ = pairs.most_common(1)[0]
            new_token = f"{best_pair[0]}{best_pair[1]}"
            if new_token in self.vocab:
                break
            for token, pieces in list(current_vocab.items()):
                if len(pieces) < 2:
                    continue
                new_pieces = []
                i = 0
                while i < len(pieces):
                    if (
                        i < len(pieces) - 1
                        and pieces[i] == best_pair[0]
                        and pieces[i + 1] == best_pair[1]
                    ):
                        new_pieces.append(new_token)
                        i += 2
                    else:
                        new_pieces.append(pieces[i])
                        i += 1
                current_vocab[token] = new_pieces
            self.vocab[new_token] = len(self.vocab)
            self.inverse_vocab[self.vocab[new_token]] = new_token

        self._initialized = True

    def encode(self, text: str) -> List[int]:
        if not self._initialized:
            raise ValueError("Tokenizer must be fitted before encoding")
        tokens = self._initial_tokens(text)
        ids: List[int] = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
        return ids

    def decode(self, ids: List[int]) -> str:
        if not self._initialized:
            raise ValueError("Tokenizer must be fitted before decoding")
        return " ".join(self.inverse_vocab.get(i, self.unk_token) for i in ids)

    def save(self, path: str | os.PathLike[str]) -> None:
        if not self._initialized:
            raise ValueError("Tokenizer must be fitted before saving")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "vocab": self.vocab,
                    "inverse_vocab": self.inverse_vocab,
                    "special_tokens": self.special_tokens,
                    "vocab_size": self.vocab_size,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def load(self, path: str | os.PathLike[str]) -> None:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.vocab = payload["vocab"]
        self.inverse_vocab = payload["inverse_vocab"]
        self.special_tokens = payload.get("special_tokens", self.special_tokens)
        self.vocab_size = payload.get("vocab_size", self.vocab_size)
        self._initialized = True

    def _initial_tokens(self, text: str) -> List[str]:
        text = text.replace("\n", " ")
        return [char for char in text] if text else []
