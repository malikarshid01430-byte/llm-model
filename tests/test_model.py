import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from model.gpt_model import GPTModel
from model.kv_cache import KVCache


def test_model_forward_shape() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        ff_hidden_dim=64,
        max_seq_len=16,
    )
    model = GPTModel(config)
    token_ids = torch.randint(0, 32, (2, 8))
    logits = model(token_ids)
    assert logits.shape == (2, 8, 32)


def test_model_supports_kv_cache_and_loss() -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        ff_hidden_dim=64,
        max_seq_len=16,
    )
    model = GPTModel(config)
    token_ids = torch.randint(0, config.vocab_size, (1, 6))
    targets = torch.randint(0, config.vocab_size, (1, 6))

    logits, cache = model(token_ids, use_cache=True)
    assert isinstance(cache, KVCache)
    assert len(cache.layers) == config.n_layers
    assert logits.shape == (1, 6, config.vocab_size)

    next_logits, next_cache = model(
        token_ids[:, -1:], past_key_values=cache, use_cache=True
    )
    assert next_logits.shape == (1, 1, config.vocab_size)
    assert isinstance(next_cache, KVCache)

    loss = model.compute_loss(token_ids, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_save_and_load_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=48,
        d_model=24,
        n_layers=1,
        n_heads=2,
        ff_hidden_dim=48,
        max_seq_len=12,
    )
    model = GPTModel(config)
    save_path = tmp_path / "gpt.pt"
    model.save_pretrained(save_path)

    reloaded = GPTModel(config)
    reloaded.load_pretrained(save_path)

    assert torch.allclose(
        reloaded.token_embedding.embedding.weight,
        model.token_embedding.embedding.weight,
    )
