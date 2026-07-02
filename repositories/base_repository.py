from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """A minimal reusable repository abstraction for domain services."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def list(self) -> list[T]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
