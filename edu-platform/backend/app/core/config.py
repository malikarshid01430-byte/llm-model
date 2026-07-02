import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"


def _get_env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name, "")
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_origins() -> List[str]:
    return _get_env_list(
        "ALLOWED_ORIGINS",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
    )


def _default_debug() -> bool:
    explicit = os.getenv("DEBUG")
    if explicit is not None:
        return explicit.lower() == "true"
    return os.getenv("JWT_SECRET") is None


@dataclass(slots=True)
class Settings:
    app_name: str = "EduAI Platform"
    debug: bool = field(default_factory=_default_debug)
    allowed_origins: List[str] = field(default_factory=_default_origins)
    jwt_secret: str = os.getenv("JWT_SECRET") or "dev-secret-key"
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_access_token_expire_minutes: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    postgres_url: str = os.getenv("POSTGRES_URL", "sqlite+aiosqlite:///./eduai.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    vector_store_path: str = os.getenv(
        "VECTOR_STORE_PATH", str(BASE_DIR / "data" / "vector_store")
    )
    data_dir: str = str(DATA_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    log_dir: str = str(LOG_DIR)


settings = Settings()


def validate_settings() -> None:
    if os.getenv("JWT_SECRET") is None:
        warnings.warn(
            "JWT_SECRET is not set; using development default.",
            stacklevel=2,
        )
    if (
        os.getenv("ENVIRONMENT", "").lower() == "production"
        and os.getenv("JWT_SECRET") is None
    ):
        raise RuntimeError("JWT_SECRET must be set in production")
    if "*" in settings.allowed_origins and not settings.debug:
        warnings.warn(
            "ALLOWED_ORIGINS contains '*' with credentials enabled; "
            "set explicit origins for production.",
            stacklevel=2,
        )
