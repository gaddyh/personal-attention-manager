# src/personal_attention_manager/domain.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from personal_attention_manager.core.models import (
    ChatState,
    ChatStatus,
    MessageDirection,
    NotificationPolicy,
)

# Re-exported so existing imports like `from personal_attention_manager.domain import ChatStatus`
# continue to work without changes.
__all__ = [
    "ChatState",
    "ChatStatus",
    "MessageDirection",
    "NotificationPolicy",
    "now_utc",
    "record_inbound_message",
    "record_outbound_message",
    "is_quiet_hours",
    "should_mark_waiting",
    "should_send_notification",
    "mark_notification_sent",
    "snooze_chat",
    "ignore_chat",
    "mark_handled",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def record_inbound_message(
    chat: ChatState,
    message_time: datetime,
    snippet: Optional[str] = None,
) -> ChatState:
    """
    Customer sent a message to the owner.
    This can create or refresh a waiting state.
    """

    chat.last_inbound_message_at = message_time
    chat.last_message_direction = MessageDirection.INBOUND
    chat.waiting_since = message_time

    if chat.status != ChatStatus.IGNORED:
        chat.status = ChatStatus.WAITING

    if snippet:
        chat.last_message_snippet = snippet[:200]

    return chat


def record_outbound_message(
    chat: ChatState,
    message_time: datetime,
) -> ChatState:
    """
    Owner replied to the customer.
    This resolves the waiting state.
    """

    chat.last_outbound_message_at = message_time
    chat.last_message_direction = MessageDirection.OUTBOUND

    chat.waiting_since = None
    chat.status = ChatStatus.HANDLED

    chat.snoozed_until = None
    chat.next_notification_at = None

    return chat


def is_quiet_hours(
    current_time: datetime,
    policy: NotificationPolicy,
) -> bool:
    """
    Supports quiet hours that cross midnight.

    Example:
    quiet start = 22:00
    quiet end   = 08:00

    23:00 is quiet.
    03:00 is quiet.
    12:00 is not quiet.
    """

    if not policy.quiet_hours_enabled:
        return False

    local_t = current_time.timetz().replace(tzinfo=None)
    start = policy.quiet_hours_start
    end = policy.quiet_hours_end

    if start < end:
        return start <= local_t < end

    return local_t >= start or local_t < end


def should_mark_waiting(
    chat: ChatState,
    policy: NotificationPolicy,
    current_time: datetime,
) -> bool:
    """
    A chat is waiting when:
    - policy is enabled
    - last message is from customer
    - owner has not replied after that
    - waiting time passed the configured threshold
    """

    if not policy.enabled:
        return False

    if chat.status in {ChatStatus.IGNORED, ChatStatus.SNOOZED}:
        return False

    if chat.last_message_direction != MessageDirection.INBOUND:
        return False

    if chat.waiting_since is None:
        return False

    threshold = timedelta(minutes=policy.waiting_threshold_minutes)
    return current_time - chat.waiting_since >= threshold


def should_send_notification(
    chat: ChatState,
    policy: NotificationPolicy,
    current_time: datetime,
) -> bool:
    """
    Decides whether to send a reminder now.

    This is stricter than should_mark_waiting:
    it also checks quiet hours, snooze, repeat interval,
    and daily notification limit.
    """

    if not should_mark_waiting(chat, policy, current_time):
        return False

    if is_quiet_hours(current_time, policy):
        return False

    if chat.snoozed_until and current_time < chat.snoozed_until:
        return False

    if chat.notification_count_today >= policy.max_notifications_per_chat_per_day:
        return False

    if chat.last_notification_sent_at is None:
        return True

    repeat_after = timedelta(minutes=policy.repeat_after_minutes)
    return current_time - chat.last_notification_sent_at >= repeat_after


def mark_notification_sent(
    chat: ChatState,
    sent_at: datetime,
    policy: NotificationPolicy,
) -> ChatState:
    chat.last_notification_sent_at = sent_at
    chat.notification_count_today += 1
    chat.next_notification_at = sent_at + timedelta(
        minutes=policy.repeat_after_minutes
    )
    return chat


def snooze_chat(
    chat: ChatState,
    until: datetime,
) -> ChatState:
    chat.status = ChatStatus.SNOOZED
    chat.snoozed_until = until
    chat.next_notification_at = until
    return chat


def ignore_chat(
    chat: ChatState,
    until: Optional[datetime] = None,
) -> ChatState:
    chat.status = ChatStatus.IGNORED
    chat.ignored_until = until
    chat.next_notification_at = None
    return chat


def mark_handled(chat: ChatState) -> ChatState:
    chat.status = ChatStatus.HANDLED
    chat.waiting_since = None
    chat.snoozed_until = None
    chat.next_notification_at = None
    return chat