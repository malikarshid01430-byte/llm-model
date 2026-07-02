from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class BatchConfig:
    """Configuration for dynamic batching."""
    max_tokens: int = 1024
    max_batch_size: int = 32
    pad_value: int = 0
    sort_by_length: bool = True


class DynamicBatchSampler:
    """Samples batches dynamically based on sequence length to minimize padding."""

    def __init__(
        self,
        dataset: Dataset,
        batch_config: BatchConfig,
        collate_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = batch_config
        self.collate_fn = collate_fn
        self._build_batches()

    def _build_batches(self) -> None:
        """Group samples into batches based on length."""
        # Get all samples with their lengths
        samples = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                length = sample.get("input_ids", sample.get("targets", torch.tensor([]))).size(0)
            else:
                length = sample.size(0)
            samples.append((idx, length))
        
        # Sort by length if configured
        if self.config.sort_by_length:
            samples.sort(key=lambda x: x[1])
        
        # Group into batches
        self.batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, length in samples:
            # Check if adding this sample would exceed limits
            if (len(current_batch) >= self.config.max_batch_size or
                current_tokens + length > self.config.max_tokens):
                if current_batch:
                    self.batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
            else:
                current_batch.append(idx)
                current_tokens += length
        
        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self) -> list[int]:
        """Iterate over batches."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)


class DynamicDataLoader:
    """DataLoader with dynamic batching support."""

    def __init__(
        self,
        dataset: Dataset,
        batch_config: BatchConfig,
        collate_fn: Callable[..., Any] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self.dataset = dataset
        self.config = batch_config
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampler = DynamicBatchSampler(dataset, batch_config, collate_fn)

    def __iter__(self) -> Any:
        """Iterate over batches."""
        for batch_indices in self.sampler:
            batch = [self.dataset[idx] for idx in batch_indices]
            if self.collate_fn is not None:
                yield self.collate_fn(batch)
            else:
                yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.sampler)


class SequencePacker:
    """Packs sequences efficiently with minimal padding."""

    def __init__(self, max_seq_len: int, pad_value: int = 0) -> None:
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

    def pack(self, sequences: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Pack a list of sequences into a batch with padding.
        
        Args:
            sequences: List of token tensors
            
        Returns:
            Dictionary with input_ids, targets, and attention_mask
        """
        if not sequences:
            raise ValueError("Cannot pack empty sequence list")
        
        # Find max length in batch (capped at max_seq_len)
        max_len = min(max(seq.size(0) for seq in sequences), self.max_seq_len)
        
        # Pad all sequences to max_len
        padded = []
        for seq in sequences:
            if seq.size(0) > max_len:
                seq = seq[:max_len]
            elif seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), self.pad_value, dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            padded.append(seq)
        
        # Stack into batch
        input_ids = torch.stack(padded)
        
        # Create targets (shifted by 1)
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = self.pad_value
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            if seq.size(0) < max_len:
                attention_mask[i, seq.size(0):] = False
        
        return {
            "input_ids": input_ids,
            "targets": targets,
            "attention_mask": attention_mask,
        }


class DataCollator:
    """Collates batches with sequence packing."""

    def __init__(self, max_seq_len: int, pad_value: int = 0) -> None:
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.packer = SequencePacker(max_seq_len, pad_value)

    def collate(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of sequences.
        
        Args:
            batch: List of token tensors
            
        Returns:
            Dictionary with input_ids, targets, and attention_mask
        """
        return self.packer.pack(batch)

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Make collator callable."""
        return self.collate(batch)


class CurriculumSampler:
    """Samples batches with curriculum learning - starts with shorter sequences."""

    def __init__(
        self,
        dataset: Dataset,
        batch_config: BatchConfig,
        collate_fn: Callable[..., Any] | None = None,
        curriculum_epochs: int = 3,
    ) -> None:
        self.dataset = dataset
        self.config = batch_config
        self.collate_fn = collate_fn
        self.curriculum_epochs = curriculum_epochs
        self.current_epoch = 0
        self._build_batches()

    def _build_batches(self) -> None:
        """Build batches with curriculum learning."""
        samples = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                length = sample.get("input_ids", sample.get("targets", torch.tensor([]))).size(0)
            else:
                length = sample.size(0)
            samples.append((idx, length))
        
        # Sort by length
        samples.sort(key=lambda x: x[1])
        
        # Group into batches
        self.all_batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, length in samples:
            if (len(current_batch) >= self.config.max_batch_size or
                current_tokens + length > self.config.max_tokens):
                if current_batch:
                    self.all_batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length
            else:
                current_batch.append(idx)
                current_tokens += length
        
        if current_batch:
            self.all_batches.append(current_batch)
        
        self.batches = self.all_batches.copy()

    def set_epoch(self, epoch: int) -> None:
        """Update curriculum based on epoch."""
        self.current_epoch = epoch
        
        # Gradually increase max sequence length
        progress = min(epoch / self.curriculum_epochs, 1.0)
        max_seq_len = int(self.config.max_tokens * progress)
        max_seq_len = max(max_seq_len, 32)  # Minimum sequence length
        
        # Filter batches that fit within current max_seq_len
        self.batches = []
        for batch in self.all_batches:
            total_length = sum(
                self.dataset[idx].size(0) if not isinstance(self.dataset[idx], dict)
                else self.dataset[idx].get("input_ids", torch.tensor([])).size(0)
                for idx in batch
            )
            if total_length <= max_seq_len * len(batch):
                self.batches.append(batch)

    def __iter__(self) -> list[int]:
        """Iterate over batches."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)