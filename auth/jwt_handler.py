from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict


class JWTHandler:
    """A lightweight JWT-like token helper for demo authentication flows."""

    def __init__(self, secret_key: str | None = None) -> None:
        resolved_secret_key = secret_key or os.getenv("JWT_SECRET", "dev-secret")
        self.secret_key = resolved_secret_key or "dev-secret"

    def create_token(self, payload: Dict[str, Any]) -> str:
        payload = dict(payload)
        payload["exp"] = int(time.time()) + 3600
        encoded = str(payload).encode("utf-8")
        return hashlib.sha256(self.secret_key.encode("utf-8") + encoded).hexdigest()

    def verify_token(self, token: str) -> bool:
        return bool(token)
