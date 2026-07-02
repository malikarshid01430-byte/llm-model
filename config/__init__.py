from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .settings import Settings, settings

_root_config_path = Path(__file__).resolve().parents[1] / "config.py"
_spec = importlib.util.spec_from_file_location("root_config", _root_config_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load config module from {_root_config_path}")
_root_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _root_config
_spec.loader.exec_module(_root_config)

ModelConfig = _root_config.ModelConfig
TokenizerConfig = _root_config.TokenizerConfig
TrainingConfig = _root_config.TrainingConfig
InferenceConfig = _root_config.InferenceConfig
AppConfig = _root_config.AppConfig
get_default_config = _root_config.get_default_config

__all__ = [
    "Settings",
    "settings",
    "ModelConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "InferenceConfig",
    "AppConfig",
    "get_default_config",
]
