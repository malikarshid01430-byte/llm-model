from __future__ import annotations

from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_collate(batch: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad variable-length token sequences into a single batch tensor."""
    return pad_sequence(batch, batch_first=True, padding_value=pad_value)


class TextDataset(Dataset):
    """Dataset that prepares tokenized sequences for autoregressive training."""

    def __init__(self, texts: List[str], tokenizer, max_seq_len: int = 256) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[torch.Tensor] = []
        self._prepare()

    def _prepare(self) -> None:
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            tokens = [self.tokenizer.vocab[self.tokenizer.eos_token]] + tokens
            for offset in range(0, len(tokens), self.max_seq_len):
                chunk = tokens[offset : offset + self.max_seq_len]
                if len(chunk) < 2:
                    continue
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
