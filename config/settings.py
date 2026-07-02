from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv as _dotenv_load_dotenv
except ImportError:  # pragma: no cover - fallback for minimal environments
    _dotenv_load_dotenv = None  # type: ignore[assignment]


def load_dotenv(*args: Any, **kwargs: Any) -> bool:
    if _dotenv_load_dotenv is None:
        return False
    return bool(_dotenv_load_dotenv(*args, **kwargs))


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "llm-from-scratch")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./checkpoints"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))


settings = Settings()
