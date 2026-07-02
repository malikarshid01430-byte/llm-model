from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor | None = None


class SequencePacker:
    def __init__(self, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len

    def pack(self, sequences: List[torch.Tensor]) -> PackedBatch:
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        if padded.size(1) > self.max_seq_len:
            padded = padded[:, : self.max_seq_len]
        input_ids = padded[:, :-1]
        targets = padded[:, 1:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return PackedBatch(
            input_ids=input_ids, targets=targets, attention_mask=attention_mask
        )


class DataCollator:
    def __init__(self, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len

    def collate(self, batch: List[torch.Tensor]) -> dict[str, torch.Tensor | None]:
        packer = SequencePacker(self.max_seq_len)
        packed = packer.pack(batch)
        return {
            "input_ids": packed.input_ids,
            "targets": packed.targets,
            "attention_mask": packed.attention_mask,
        }


class TextDataset(Dataset):
    def __init__(
        self, texts: list[str], tokenizer: Any, max_seq_len: int = 256
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[torch.Tensor] = []
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
