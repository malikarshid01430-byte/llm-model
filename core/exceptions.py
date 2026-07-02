from __future__ import annotations


class EduLLMError(Exception):
    """Base exception for EduLLM application errors."""


class ConfigurationError(EduLLMError):
    """Raised when configuration is invalid."""


class TokenizerError(EduLLMError):
    """Raised when tokenizer operations fail."""


class TrainingError(EduLLMError):
    """Raised when training fails."""


class InferenceError(EduLLMError):
    """Raised when inference fails."""
