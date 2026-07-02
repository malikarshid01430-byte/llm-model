from __future__ import annotations

import torch

from config import InferenceConfig
from model import GPTModel


def generate_text(
    model: GPTModel, tokenizer, prompt: str, config: InferenceConfig | None = None
) -> str:
    config = config or InferenceConfig()
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.vocab[tokenizer.eos_token]]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)
