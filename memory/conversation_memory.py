from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple


class ConversationMemory:
    """A simple in-memory conversation buffer for generation workflows."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._messages: Deque[Tuple[str, str]] = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append((role, content))

    def get_messages(self) -> List[Tuple[str, str]]:
        return list(self._messages)

    def summarize(self) -> str:
        return " | ".join(content for _, content in self._messages)

    def clear(self) -> None:
        self._messages.clear()
