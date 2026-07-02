from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
MODEL_DIR = ROOT_DIR / "model"
TRAINING_DIR = ROOT_DIR / "training"


@dataclass
class TokenizerConfig:
    vocab_size: int = 2000
    min_frequency: int = 2
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    eos_token: str = "</s>"


@dataclass
class ModelConfig:
    vocab_size: int = 2000
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_hidden_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    tie_weights: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 8
    max_steps: int = 200
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 20
    device: str = "cpu"
    use_mixed_precision: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_every: int = 50
    val_every: int = 100
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0


@dataclass
class InferenceConfig:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    stop_tokens: list[int] = field(default_factory=list)
    max_context_len: int | None = None


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


def get_default_config() -> dict:
    return {
        "tokenizer": asdict(TokenizerConfig()),
        "model": asdict(ModelConfig()),
        "training": asdict(TrainingConfig()),
        "inference": asdict(InferenceConfig()),
        "app": asdict(AppConfig()),
    }
