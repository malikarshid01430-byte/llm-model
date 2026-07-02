from __future__ import annotations

from dataclasses import dataclass
from typing import List

from authorization import Role


@dataclass
class User:
    """A simple user model used by the application layer."""

    user_id: str
    email: str
    role: Role


class UserService:
    """Simple user management service for the platform shell."""

    def __init__(self) -> None:
        self._users: List[User] = []

    def create_user(self, user_id: str, email: str, role: Role) -> User:
        user = User(user_id=user_id, email=email, role=role)
        self._users.append(user)
        return user

    def list_users(self) -> List[User]:
        return list(self._users)
