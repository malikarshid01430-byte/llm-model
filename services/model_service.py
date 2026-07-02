from __future__ import annotations

from config import ModelConfig
from model.gpt_model import GPTModel


class ModelService:
    """Service layer for constructing and loading model instances."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def build_model(self) -> GPTModel:
        return GPTModel(self.config)

    def load_model(self, checkpoint_path: str) -> GPTModel:
        model = self.build_model()
        checkpoint = __import__("torch").load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        return model
