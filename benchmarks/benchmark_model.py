from __future__ import annotations

import time

import torch

from config import ModelConfig
from model.gpt_model import GPTModel


def benchmark() -> None:
    config = ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_hidden_dim=128,
        max_seq_len=32,
    )
    model = GPTModel(config)
    model.eval()
    batch = torch.randint(0, 128, (2, 16))
    start = time.time()
    with torch.no_grad():
        model(batch)
    elapsed = time.time() - start
    print(f"Benchmark complete in {elapsed:.4f}s")


if __name__ == "__main__":
    benchmark()
