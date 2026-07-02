"""Tests for model components: SwiGLU, GPTModel advanced features."""

from __future__ import annotations

import torch

from config import ModelConfig
from model.gpt_model import GPTModel
from model.swiglu import SwiGLU


class TestSwiGLU:
    def test_output_shape(self) -> None:
        swiglu = SwiGLU(d_model=32, hidden_dim=64)
        x = torch.randn(2, 4, 32)
        out = swiglu(x)
        assert out.shape == (2, 4, 32)

    def test_gradient_flow(self) -> None:
        swiglu = SwiGLU(d_model=16, hidden_dim=32)
        x = torch.randn(1, 2, 16, requires_grad=True)
        out = swiglu(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_dropout(self) -> None:
        swiglu = SwiGLU(d_model=16, hidden_dim=32, dropout=0.5)
        x = torch.randn(4, 8, 16)
        swiglu.train()
        out_train = swiglu(x)
        swiglu.eval()
        out_eval = swiglu(x)
        assert out_train.shape == out_eval.shape


class TestGPTModelAdvanced:
    def _make_model(self) -> GPTModel:
        config = ModelConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ff_hidden_dim=64,
            max_seq_len=16,
        )
        return GPTModel(config)

    def test_model_param_count(self) -> None:
        model = self._make_model()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_model_different_seq_lengths(self) -> None:
        model = self._make_model()
        model.eval()
        for seq_len in [1, 4, 8, 16]:
            x = torch.randint(0, 64, (1, seq_len))
            out = model(x)
            assert out.shape == (1, seq_len, 64)

    def test_model_batch_sizes(self) -> None:
        model = self._make_model()
        model.eval()
        for batch_size in [1, 2, 4]:
            x = torch.randint(0, 64, (batch_size, 4))
            out = model(x)
            assert out.shape == (batch_size, 4, 64)

    def test_model_train_mode(self) -> None:
        model = self._make_model()
        model.train()
        x = torch.randint(0, 64, (2, 4))
        out = model(x)
        assert out.shape == (2, 4, 64)
        loss = out.sum()
        loss.backward()

    def test_model_state_dict_roundtrip(self, tmp_path) -> None:
        model1 = self._make_model()
        path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), path)
        model2 = self._make_model()
        model2.load_state_dict(torch.load(path, weights_only=True))
        x = torch.randint(0, 64, (1, 4))
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)
