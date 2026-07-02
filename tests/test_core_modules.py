"""Tests for core modules: auth, kv_cache, normalization, config."""

from __future__ import annotations

import warnings

import torch

from auth.jwt_handler import JWTHandler
from config import (
    AppConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)
from model.kv_cache import KVCache
from utils.normalization import chunk_text, normalize_text


class TestJWTHandler:
    def test_create_with_explicit_secret(self) -> None:
        handler = JWTHandler(secret_key="test-secret-123")
        assert handler.secret_key == "test-secret-123"

    def test_create_without_secret_warns(self, monkeypatch) -> None:
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            handler = JWTHandler()
            assert handler.secret_key == "dev-secret"
            assert len(w) == 1
            assert "JWT_SECRET" in str(w[0].message)

    def test_create_token_returns_hex(self) -> None:
        handler = JWTHandler(secret_key="test-secret")
        token = handler.create_token({"user": "alice", "role": "admin"})
        assert isinstance(token, str)
        assert len(token) == 64  # sha256 hex length

    def test_create_token_deterministic(self) -> None:
        handler = JWTHandler(secret_key="key1")
        t1 = handler.create_token({"user": "bob"})
        t2 = handler.create_token({"user": "bob"})
        assert t1 == t2

    def test_different_secrets_produce_different_tokens(self) -> None:
        h1 = JWTHandler(secret_key="key1")
        h2 = JWTHandler(secret_key="key2")
        t1 = h1.create_token({"user": "alice"})
        t2 = h2.create_token({"user": "alice"})
        assert t1 != t2

    def test_verify_token_returns_true_for_valid(self) -> None:
        handler = JWTHandler(secret_key="test")
        token = handler.create_token({"user": "test"})
        assert handler.verify_token(token) is True

    def test_verify_token_returns_false_for_empty(self) -> None:
        handler = JWTHandler(secret_key="test")
        assert handler.verify_token("") is False


class TestKVCache:
    def test_empty_cache(self) -> None:
        cache = KVCache()
        assert len(cache) == 0

    def test_append_and_length(self) -> None:
        cache = KVCache()
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        cache.append(k, v)
        assert len(cache) == 1

    def test_get_layer_valid(self) -> None:
        cache = KVCache()
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        cache.append(k, v)
        result = cache.get_layer(0)
        assert result is not None
        assert torch.equal(result[0], k)
        assert torch.equal(result[1], v)

    def test_get_layer_invalid_index(self) -> None:
        cache = KVCache()
        assert cache.get_layer(0) is None
        assert cache.get_layer(-1) is None

    def test_clear(self) -> None:
        cache = KVCache()
        cache.append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        cache.append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_multiple_layers(self) -> None:
        cache = KVCache()
        for i in range(5):
            cache.append(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
        assert len(cache) == 5
        assert cache.get_layer(4) is not None
        assert cache.get_layer(5) is None


class TestNormalization:
    def test_normalize_whitespace(self) -> None:
        assert normalize_text("hello   world") == "hello world"

    def test_normalize_newlines(self) -> None:
        assert normalize_text("hello\r\nworld") == "hello world"

    def test_normalize_mixed(self) -> None:
        assert normalize_text("  hello  \n  world  ") == "hello world"

    def test_chunk_text_small(self) -> None:
        text = "hello world"
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_chunk_text_large(self) -> None:
        text = "a" * 100
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) == 4  # 100/30 = 3.33 -> 4 chunks
        assert all(len(c) <= 30 for c in chunks)

    def test_chunk_text_exact_boundary(self) -> None:
        text = "a" * 60
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 30
        assert chunks[1] == "a" * 30


class TestConfigs:
    def test_training_config_defaults(self) -> None:
        config = TrainingConfig()
        assert config.batch_size == 8
        assert config.device == "cpu"
        assert config.gradient_clip == 1.0

    def test_model_config_defaults(self) -> None:
        config = ModelConfig()
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.tie_weights is True

    def test_inference_config_defaults(self) -> None:
        config = InferenceConfig()
        assert config.stop_tokens == []
        assert config.temperature == 0.8

    def test_app_config_defaults(self) -> None:
        config = AppConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000

    def test_get_default_config(self) -> None:
        config = get_default_config()
        assert "tokenizer" in config
        assert "model" in config
        assert "training" in config
        assert "inference" in config
        assert "app" in config
        assert isinstance(config["model"]["d_model"], int)
