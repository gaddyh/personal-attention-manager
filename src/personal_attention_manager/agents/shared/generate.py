from __future__ import annotations

import enum
from datetime import datetime
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field


class GeneratedMessage(BaseModel):
    sender: str = Field(description="Name or role of the message sender.")
    text: str = Field(description="The message text.")
    sent_time: str = Field(
        description="ISO datetime string, for example 2026-06-24T09:30:00."
    )


class Difficulty(enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GeneratedExampleLike(Protocol):
    id: str
    recent_messages: list[GeneratedMessage]
    notes: str


ExT = TypeVar("ExT", bound=GeneratedExampleLike)


def validate_example(ex: GeneratedExampleLike) -> tuple[bool, str]:
    if not ex.id.strip():
        return False, "missing id"

    if not (1 <= len(ex.recent_messages) <= 5):
        return False, "recent_messages must contain 1 to 5 messages"

    for msg in ex.recent_messages:
        if not msg.sender.strip():
            return False, "empty sender"

        if not msg.text.strip():
            return False, "empty text"

        try:
            datetime.fromisoformat(msg.sent_time)
        except ValueError:
            return False, f"invalid sent_time: {msg.sent_time}"

    if not ex.notes.strip():
        return False, "missing notes"

    return True, "ok"


def dedupe_and_validate(
    examples: list[ExT],
    existing_ids: set[str],
) -> tuple[list[ExT], list[str]]:
    accepted: list[ExT] = []
    rejected: list[str] = []

    seen_ids = set(existing_ids)

    for ex in examples:
        if ex.id in seen_ids:
            rejected.append(f"{ex.id}: duplicate id")
            continue

        valid, reason = validate_example(ex)
        if not valid:
            rejected.append(f"{ex.id}: {reason}")
            continue

        accepted.append(ex)
        seen_ids.add(ex.id)

    return accepted, rejected
