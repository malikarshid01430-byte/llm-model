# GPT Decoder-Only Transformer

This module implements a compact GPT-style decoder-only transformer manually in PyTorch without using Hugging Face transformer implementations.

## Components

- Token embeddings
- Learnable positional embeddings
- LayerNorm
- Multi-head self-attention with causal masking
- Feed-forward network using GELU and SwiGLU
- Residual connections and decoder blocks
- KV cache support for autoregressive decoding
- Model save/load helpers

## Usage

```python
from config import ModelConfig
from model.gpt_model import GPTModel

config = ModelConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4)
model = GPTModel(config)
```
