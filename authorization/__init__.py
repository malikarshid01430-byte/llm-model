from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Basic role definitions for the education platform."""

    STUDENT = "student"
    TEACHER = "teacher"
    PARENT = "parent"
    ADMIN = "admin"
