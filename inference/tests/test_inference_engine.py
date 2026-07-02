from __future__ import annotations

# Add parent directory to path for imports
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ModelConfig
from inference.engine import ConversationEngine, ConversationMemory, InferenceEngine
from model.gpt_model import GPTModel


class TinyDataset(Dataset):
    """Minimal dataset for testing."""

    def __init__(self, size: int = 8, seq_len: int = 6, vocab_size: int = 32) -> None:
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100) -> None:
        self.vocab_size = vocab_size
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.vocab["<pad>"] = 0
        self.vocab["</s>"] = vocab_size - 1
        self.vocab["<unk>"] = vocab_size - 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"

    def encode(self, text: str) -> list[int]:
        """Simple encoding for testing."""
        tokens = []
        for char in text[:100]:
            token_id = hash(char) % (self.vocab_size - 3) + 1
            tokens.append(token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Simple decoding for testing."""
        return "".join(chr(t % 128) for t in tokens)


def create_test_model_and_tokenizer(vocab_size: int = 100, seq_len: int = 256):
    """Create test model and tokenizer."""
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        ff_hidden_dim=64,
        max_seq_len=seq_len,
    )
    model = GPTModel(config)
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    return model, tokenizer


# ==================== Inference Engine Tests ====================


class TestInferenceEngine:
    """Test inference engine functionality."""

    def test_generate_basic(self) -> None:
        """Test basic text generation."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", max_new_tokens=10)

        assert isinstance(generated, str)
        assert len(generated) > 0

    def test_generate_with_temperature(self) -> None:
        """Test generation with different temperatures."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Greedy (temperature=0)
        greedy = engine.generate("Hello", temperature=0.0, max_new_tokens=10)
        assert isinstance(greedy, str)

        # Sampling (temperature=1.0)
        sampling = engine.generate("Hello", temperature=1.0, max_new_tokens=10)
        assert isinstance(sampling, str)

    def test_generate_with_top_k(self) -> None:
        """Test generation with top-k sampling."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", top_k=10, max_new_tokens=10)
        assert isinstance(generated, str)

    def test_generate_with_top_p(self) -> None:
        """Test generation with top-p sampling."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", top_p=0.9, max_new_tokens=10)
        assert isinstance(generated, str)

    def test_generate_with_repetition_penalty(self) -> None:
        """Test generation with repetition penalty."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", repetition_penalty=1.5, max_new_tokens=10)
        assert isinstance(generated, str)

    def test_generate_with_stop_tokens(self) -> None:
        """Test generation with stop tokens."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Use EOS token as stop token
        stop_token_id = tokenizer.vocab[tokenizer.eos_token]
        generated = engine.generate(
            "Hello", stop_tokens=[stop_token_id], max_new_tokens=100
        )
        assert isinstance(generated, str)

    def test_generate_with_max_context(self) -> None:
        """Test generation with max context window."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", max_context_len=10, max_new_tokens=5)
        assert isinstance(generated, str)

    def test_generate_do_sample_false(self) -> None:
        """Test generation with do_sample=False (greedy)."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", do_sample=False, max_new_tokens=10)
        assert isinstance(generated, str)


# ==================== Streaming Generation Tests ====================


class TestStreamingGeneration:
    """Test streaming generation functionality."""

    def test_stream_generate_basic(self) -> None:
        """Test basic streaming generation."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        chunks = list(engine.stream_generate("Hello", max_new_tokens=10))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_stream_generate_with_params(self) -> None:
        """Test streaming with various parameters."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        chunks = list(
            engine.stream_generate(
                "Hello", temperature=0.8, top_k=10, max_new_tokens=10
            )
        )

        assert len(chunks) > 0

    def test_stream_generate_with_stop_tokens(self) -> None:
        """Test streaming with stop tokens."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        stop_token_id = tokenizer.vocab[tokenizer.eos_token]
        chunks = list(
            engine.stream_generate(
                "Hello", stop_tokens=[stop_token_id], max_new_tokens=100
            )
        )

        assert len(chunks) > 0


# ==================== Beam Search Tests ====================


class TestBeamSearch:
    """Test beam search functionality."""

    def test_beam_search_basic(self) -> None:
        """Test basic beam search."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        results = engine.beam_search("Hello", max_new_tokens=10, beam_width=2)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        assert all(isinstance(item[0], float) for item in results)
        assert all(isinstance(item[1], str) for item in results)

    def test_beam_search_width(self) -> None:
        """Test beam search with different widths."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        results = engine.beam_search("Hello", max_new_tokens=10, beam_width=3)

        # Should return at most beam_width results
        assert len(results) <= 3

    def test_beam_search_with_stop_tokens(self) -> None:
        """Test beam search with stop tokens."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        stop_token_id = tokenizer.vocab[tokenizer.eos_token]
        results = engine.beam_search(
            "Hello", max_new_tokens=20, beam_width=2, stop_tokens=[stop_token_id]
        )

        assert isinstance(results, list)


# ==================== Batch Generation Tests ====================


class TestBatchGeneration:
    """Test batch generation functionality."""

    def test_batch_generate(self) -> None:
        """Test batch generation."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        prompts = ["Hello", "World", "Test"]
        results = engine.batch_generate(prompts, max_new_tokens=10)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_batch_stream_generate(self) -> None:
        """Test batch streaming generation."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        prompts = ["Hello", "World"]
        generators = engine.batch_stream_generate(prompts, max_new_tokens=10)

        assert len(generators) == 2
        assert all(hasattr(g, "__iter__") for g in generators)


# ==================== Conversation Memory Tests ====================


class TestConversationMemory:
    """Test conversation memory functionality."""

    def test_memory_add_turn(self) -> None:
        """Test adding turns to memory."""
        memory = ConversationMemory(max_turns=5)

        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi there!")

        assert len(memory) == 2

    def test_memory_max_turns(self) -> None:
        """Test memory respects max_turns."""
        memory = ConversationMemory(max_turns=3)

        for i in range(5):
            memory.add_turn("user", f"Message {i}")

        assert len(memory) == 3
        assert memory.history[0].content == "Message 2"

    def test_memory_get_context(self) -> None:
        """Test getting conversation context."""
        memory = ConversationMemory()

        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi!")
        memory.add_turn("user", "How are you?")

        context = memory.get_context()

        assert "User: Hello" in context
        assert "Assistant: Hi!" in context
        assert "User: How are you?" in context

    def test_memory_clear(self) -> None:
        """Test clearing memory."""
        memory = ConversationMemory()

        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi!")

        assert len(memory) == 2

        memory.clear()

        assert len(memory) == 0


# ==================== Conversation Engine Tests ====================


class TestConversationEngine:
    """Test conversation engine functionality."""

    def test_chat_basic(self) -> None:
        """Test basic chat functionality."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        response = engine.chat("Hello", max_new_tokens=10)

        assert isinstance(response, str)
        assert len(engine.memory) == 2  # User + Assistant

    def test_chat_with_system_prompt(self) -> None:
        """Test chat with system prompt."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        engine.set_system_prompt("You are a helpful assistant.")
        response = engine.chat("Hello", max_new_tokens=10)

        assert isinstance(response, str)

    def test_chat_stream(self) -> None:
        """Test streaming chat."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        chunks = list(engine.stream_chat("Hello", max_new_tokens=10))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chat_clear_history(self) -> None:
        """Test clearing chat history."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        engine.chat("Hello", max_new_tokens=10)
        assert len(engine.memory) == 2

        engine.clear_history()
        assert len(engine.memory) == 0

    def test_chat_multiple_turns(self) -> None:
        """Test multiple conversation turns."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        engine.chat("Hello", max_new_tokens=10)
        engine.chat("How are you?", max_new_tokens=10)
        engine.chat("Goodbye", max_new_tokens=10)

        assert len(engine.memory) == 6  # 3 user + 3 assistant


# ==================== KV Cache Tests ====================


class TestKVCache:
    """Test KV cache functionality."""

    def test_streaming_engine_cache(self) -> None:
        """Test streaming engine with cache."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Generate twice with same prompt
        text1 = engine.generate("Hello", max_new_tokens=10)
        text2 = engine.generate("Hello", max_new_tokens=10)

        # Both should work (cache is simplified in this implementation)
        assert isinstance(text1, str)
        assert isinstance(text2, str)


# ==================== Integration Tests ====================


class TestInferenceIntegration:
    """Test inference pipeline integration."""

    def test_full_inference_pipeline(self) -> None:
        """Test complete inference pipeline."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Test different generation methods
        greedy = engine.generate("Hello", temperature=0.0, max_new_tokens=10)
        sampled = engine.generate("Hello", temperature=1.0, max_new_tokens=10)
        top_k = engine.generate("Hello", top_k=10, max_new_tokens=10)
        top_p = engine.generate("Hello", top_p=0.9, max_new_tokens=10)

        assert all(isinstance(text, str) for text in [greedy, sampled, top_k, top_p])

    def test_conversation_pipeline(self) -> None:
        """Test complete conversation pipeline."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = ConversationEngine(model, tokenizer)

        # Set system prompt
        engine.set_system_prompt("You are helpful.")

        # Have a conversation
        response1 = engine.chat("Hello", max_new_tokens=10)
        response2 = engine.chat("How are you?", max_new_tokens=10)

        assert isinstance(response1, str)
        assert isinstance(response2, str)
        assert len(engine.memory) == 4

    def test_streaming_pipeline(self) -> None:
        """Test streaming generation pipeline."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        chunks = list(engine.stream_generate("Hello", max_new_tokens=20))

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0


# ==================== Edge Case Tests ====================


class TestInferenceEdgeCases:
    """Test edge cases in inference."""

    def test_empty_prompt(self) -> None:
        """Test generation with empty prompt."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Should handle empty prompt gracefully
        generated = engine.generate("", max_new_tokens=5)
        assert isinstance(generated, str)

    def test_very_long_prompt(self) -> None:
        """Test generation with long prompt."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        long_prompt = "Hello " * 100
        generated = engine.generate(long_prompt, max_new_tokens=5)
        assert isinstance(generated, str)

    def test_max_new_tokens_zero(self) -> None:
        """Test generation with max_new_tokens=0."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        generated = engine.generate("Hello", max_new_tokens=0)
        assert isinstance(generated, str)
        assert len(generated) == 0

    def test_temperature_zero_greedy(self) -> None:
        """Test that temperature=0 gives greedy decoding."""
        model, tokenizer = create_test_model_and_tokenizer()
        engine = InferenceEngine(model, tokenizer)

        # Generate twice with same input
        text1 = engine.generate("Hello", temperature=0.0, max_new_tokens=10)
        text2 = engine.generate("Hello", temperature=0.0, max_new_tokens=10)

        # Should be deterministic
        assert text1 == text2


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
