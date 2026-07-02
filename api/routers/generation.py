from __future__ import annotations

from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import InferenceConfig, ModelConfig
from inference.generator import generate_text
from model.gpt_model import GPTModel
from tokenizer.base import BPETokenizer

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20


def _load_model(tokenizer: BPETokenizer) -> GPTModel:
    model_config = ModelConfig(
        vocab_size=max(tokenizer.vocab.values()) + 1,
    )
    model = GPTModel(model_config)
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            checkpoint = torch.load(
                checkpoints[0], map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@router.post("/generate")
def generate(request: GenerateRequest) -> dict:
    vocab_path = Path("tokenizer/vocab.json")
    if not vocab_path.exists():
        raise HTTPException(status_code=503, detail="Tokenizer vocabulary not found")

    tokenizer = BPETokenizer(vocab_size=2000)
    tokenizer.load(vocab_path)
    model = _load_model(tokenizer)
    return {
        "response": generate_text(
            model,
            tokenizer,
            request.prompt,
            InferenceConfig(max_new_tokens=request.max_new_tokens),
        )
    }
