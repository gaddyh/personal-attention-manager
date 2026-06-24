from __future__ import annotations

from datetime import datetime
from typing import Protocol


class MessageLike(Protocol):
    sender: str
    text: str
    sent_time: datetime


def format_messages(messages: list[MessageLike]) -> str:
    if not messages:
        return "No recent messages."

    sorted_messages = sorted(messages, key=lambda m: m.sent_time)

    lines = []
    for msg in sorted_messages:
        lines.append(
            f"[{msg.sent_time.isoformat()}] {msg.sender}: {msg.text}"
        )

    return "\n".join(lines)
