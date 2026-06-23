# src/personal_attention_manager/models.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from typing import Optional


class MessageDirection(StrEnum):
    INBOUND = "inbound"    # customer -> owner
    OUTBOUND = "outbound"  # owner -> customer


class ChatStatus(StrEnum):
    HANDLED = "handled"
    WAITING = "waiting"
    SNOOZED = "snoozed"
    IGNORED = "ignored"


@dataclass(frozen=True)
class NotificationPolicy:
    """
    Per-owner policy for deciding when a chat should trigger a reminder.
    """

    enabled: bool = True

    # Example: notify if customer has waited more than 2 hours.
    waiting_threshold_minutes: int = 120

    # Example: remind again after 4 hours if still unanswered.
    repeat_after_minutes: int = 240

    quiet_hours_enabled: bool = True
    quiet_hours_start: time = time(hour=22, minute=0)
    quiet_hours_end: time = time(hour=8, minute=0)

    daily_digest_enabled: bool = True
    daily_digest_time: time = time(hour=9, minute=0)

    max_notifications_per_chat_per_day: int = 3


@dataclass
class ChatState:
    """
    Minimal state we keep per WhatsApp chat.

    We do not need full message history for the MVP.
    The product only needs to know whether the last meaningful message
    is waiting for the owner to reply.
    """

    chat_id: str
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None

    last_inbound_message_at: Optional[datetime] = None
    last_outbound_message_at: Optional[datetime] = None
    last_message_direction: Optional[MessageDirection] = None

    waiting_since: Optional[datetime] = None
    status: ChatStatus = ChatStatus.HANDLED

    snoozed_until: Optional[datetime] = None
    ignored_until: Optional[datetime] = None

    last_notification_sent_at: Optional[datetime] = None
    next_notification_at: Optional[datetime] = None

    notification_count_today: int = 0

    # Optional. In v1, keep this short or disable it entirely.
    last_message_snippet: Optional[str] = None


@dataclass(frozen=True)
class MessageEventResult:
    chat_id: str
    direction: MessageDirection
    status: ChatStatus
    duplicate: bool = False


@dataclass(frozen=True)
class WaitingChatNotification:
    chat_id: str
    contact_name: str | None
    contact_phone: str | None
    waiting_since: datetime
    waiting_minutes: int
    message: str
