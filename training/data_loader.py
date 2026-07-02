from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    max_seq_len: int = 256
    stride: int = 128
    min_length: int = 2
    streaming: bool = False
    buffer_size: int = 10000


class TextDataset(Dataset):
    """Dataset that prepares tokenized sequences for autoregressive training."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_seq_len: int = 256,
        stride: int = 128,
        min_length: int = 2,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.min_length = min_length
        self.samples: list[torch.Tensor] = []
        self._prepare()

    def _prepare(self) -> None:
        """Tokenize and chunk texts into training samples."""
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            
            # Add EOS token
            tokens = [self.tokenizer.vocab[self.tokenizer.eos_token]] + tokens
            
            # Create overlapping chunks with stride
            if len(tokens) <= self.max_seq_len:
                if len(tokens) >= self.min_length:
                    self.samples.append(torch.tensor(tokens, dtype=torch.long))
            else:
                for offset in range(0, len(tokens) - self.max_seq_len + 1, self.stride):
                    chunk = tokens[offset : offset + self.max_seq_len]
                    if len(chunk) >= self.min_length:
                        self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large corpora that don't fit in memory."""

    def __init__(
        self,
        file_paths: list[Path],
        tokenizer: Any,
        max_seq_len: int = 256,
        stride: int = 128,
        min_length: int = 2,
        buffer_size: int = 10000,
    ) -> None:
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.min_length = min_length
        self.buffer_size = buffer_size

    def _read_file(self, path: Path) -> Generator[str, None, None]:
        """Read text from various file formats."""
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            yield item
                        elif isinstance(item, dict) and "text" in item:
                            yield item["text"]
                elif isinstance(data, dict) and "text" in data:
                    yield data["text"]
        else:
            # Plain text, markdown, etc.
            yield path.read_text(encoding="utf-8")

    def _tokenize_stream(self) -> Generator[torch.Tensor, None, None]:
        """Tokenize texts and yield individual sequences."""
        for file_path in self.file_paths:
            try:
                for text in self._read_file(file_path):
                    tokens = self.tokenizer.encode(text)
                    if not tokens:
                        continue
                    
                    tokens = [self.tokenizer.vocab[self.tokenizer.eos_token]] + tokens
                    
                    if len(tokens) <= self.max_seq_len:
                        if len(tokens) >= self.min_length:
                            yield torch.tensor(tokens, dtype=torch.long)
                    else:
                        for offset in range(0, len(tokens) - self.max_seq_len + 1, self.stride):
                            chunk = tokens[offset : offset + self.max_seq_len]
                            if len(chunk) >= self.min_length:
                                yield torch.tensor(chunk, dtype=torch.long)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over tokenized sequences with buffering."""
        buffer = []
        for sample in self._tokenize_stream():
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                yield from buffer
                buffer = []
        if buffer:
            yield from buffer


class DatasetLoader:
    """Utility class for loading datasets from various sources."""

    @staticmethod
    def load_from_directory(
        directory: str | Path,
        extensions: set[str] | None = None,
        recursive: bool = True,
    ) -> list[str]:
        """Load all text files from a directory."""
        if extensions is None:
            extensions = {".txt", ".md", ".json", ".csv"}
        
        directory = Path(directory)
        texts = []
        
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    texts.append(file_path.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return texts

    @staticmethod
    def load_from_files(
        file_paths: list[str] | list[Path],
    ) -> list[str]:
        """Load texts from specific files."""
        texts = []
        for path in file_paths:
            try:
                texts.append(Path(path).read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error reading {path}: {e}")
        return texts

    @staticmethod
    def load_from_json(
        file_path: str | Path,
        text_key: str = "text",
    ) -> list[str]:
        """Load texts from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and text_key in item:
                    texts.append(item[text_key])
        elif isinstance(data, dict) and text_key in data:
            texts.append(data[text_key])
        
        return texts