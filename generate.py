from __future__ import annotations

from config import InferenceConfig, ModelConfig
from inference.generator import generate_text
from model.gpt_model import GPTModel
from tokenizer.base import BPETokenizer


def main() -> None:
    tokenizer = BPETokenizer(vocab_size=2000)
    tokenizer.load("tokenizer/vocab.json")
    config = ModelConfig(
        vocab_size=max(tokenizer.vocab.values()) + 1,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ff_hidden_dim=128,
        max_seq_len=128,
    )
    model = GPTModel(config)
    model.load_state_dict(
        __import__("torch").load(
            "checkpoints/checkpoint_step_5.pt", map_location="cpu"
        )["model_state"]
    )
    prompt = "The future of artificial intelligence"
    print(generate_text(model, tokenizer, prompt, InferenceConfig(max_new_tokens=20)))


if __name__ == "__main__":
    main()
