from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from core.exceptions import TokenizerError


class BPETokenizer:
    """A simple byte-pair encoding tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 2000) -> None:
        self.vocab_size = max(2, vocab_size)
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._initialized = False

    def fit(self, texts: List[str]) -> None:
        if not texts:
            raise TokenizerError(
                "At least one text example is required to fit the tokenizer"
            )

        tokenized_texts = [list(text) for text in texts]
        symbols: Counter[str] = Counter()
        for tokens in tokenized_texts:
            for token in tokens:
                symbols[token] += 1

        self.vocab = {char: idx for idx, char in enumerate(sorted(symbols.keys()))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for tokens in tokenized_texts:
                for left, right in zip(tokens, tokens[1:]):
                    left_token = self.inverse_vocab[self.vocab[left]]
                    right_token = self.inverse_vocab[self.vocab[right]]
                    pair_counts[(left_token, right_token)] += 1

            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]
            merged = f"{best_pair[0]}{best_pair[1]}"
            if merged in self.vocab:
                break

            new_id = len(self.vocab)
            self.vocab[merged] = new_id
            self.inverse_vocab[new_id] = merged
            self.merges.append(best_pair)

            updated_texts: List[List[str]] = []
            for tokens in tokenized_texts:
                updated_tokens: List[str] = []
                index = 0
                while index < len(tokens):
                    if index + 1 < len(tokens):
                        left = tokens[index]
                        right = tokens[index + 1]
                        left_token = self.inverse_vocab[self.vocab[left]]
                        right_token = self.inverse_vocab[self.vocab[right]]
                        if (left_token, right_token) == best_pair:
                            updated_tokens.append(merged)
                            index += 2
                            continue
                    updated_tokens.append(tokens[index])
                    index += 1
                updated_texts.append(updated_tokens)
            tokenized_texts = updated_texts

        self._initialized = True

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        merged_tokens = list(tokens)
        for left, right in self.merges:
            merged = f"{left}{right}"
            updated: List[str] = []
            index = 0
            while index < len(merged_tokens):
                if (
                    index + 1 < len(merged_tokens)
                    and merged_tokens[index] == left
                    and merged_tokens[index + 1] == right
                ):
                    updated.append(merged)
                    index += 2
                else:
                    updated.append(merged_tokens[index])
                    index += 1
            merged_tokens = updated
        return merged_tokens

    def encode(self, text: str) -> List[int]:
        if not self._initialized:
            raise TokenizerError("Tokenizer has not been fitted")

        tokens = self._apply_merges(list(text))
        ids: List[int] = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab.get(token, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        if not self._initialized:
            raise TokenizerError("Tokenizer has not been fitted")
        return "".join(self.inverse_vocab.get(idx, "") for idx in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "inverse_vocab": {
                str(key): value for key, value in self.inverse_vocab.items()
            },
            "merges": [list(pair) for pair in self.merges],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.vocab = payload["vocab"]
        self.inverse_vocab = {
            int(key): value for key, value in payload["inverse_vocab"].items()
        }
        self.merges = [tuple(pair) for pair in payload.get("merges", [])]
        self._initialized = True
