# tests/test_domain.py

from datetime import datetime, time, timedelta, timezone

from personal_attention_manager.core.domain import (
    ChatState,
    ChatStatus,
    MessageDirection,
    NotificationPolicy,
    ignore_chat,
    is_quiet_hours,
    mark_handled,
    mark_notification_sent,
    record_inbound_message,
    record_outbound_message,
    should_mark_waiting,
    should_send_notification,
    snooze_chat,
)


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_inbound_message_marks_chat_as_waiting():
    chat = ChatState(chat_id="chat-1")

    record_inbound_message(
        chat=chat,
        message_time=dt(10),
        snippet="Hi, are you available today?",
    )

    assert chat.last_message_direction == MessageDirection.INBOUND
    assert chat.last_inbound_message_at == dt(10)
    assert chat.waiting_since == dt(10)
    assert chat.status == ChatStatus.WAITING
    assert chat.last_message_snippet == "Hi, are you available today?"


def test_outbound_message_marks_chat_as_handled():
    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))

    record_outbound_message(chat, dt(11))

    assert chat.last_message_direction == MessageDirection.OUTBOUND
    assert chat.last_outbound_message_at == dt(11)
    assert chat.waiting_since is None
    assert chat.status == ChatStatus.HANDLED
    assert chat.next_notification_at is None


def test_chat_is_not_waiting_before_threshold():
    policy = NotificationPolicy(waiting_threshold_minutes=120)
    chat = ChatState(chat_id="chat-1")

    record_inbound_message(chat, dt(10))

    assert should_mark_waiting(chat, policy, dt(11, 59)) is False


def test_chat_is_waiting_after_threshold():
    policy = NotificationPolicy(waiting_threshold_minutes=120)
    chat = ChatState(chat_id="chat-1")

    record_inbound_message(chat, dt(10))

    assert should_mark_waiting(chat, policy, dt(12)) is True


def test_chat_does_not_notify_during_quiet_hours():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        quiet_hours_enabled=True,
        quiet_hours_start=time(22, 0),
        quiet_hours_end=time(8, 0),
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(20))

    assert should_mark_waiting(chat, policy, dt(23)) is True
    assert should_send_notification(chat, policy, dt(23)) is False


def test_chat_notifies_after_threshold_outside_quiet_hours():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        quiet_hours_enabled=True,
        quiet_hours_start=time(22, 0),
        quiet_hours_end=time(8, 0),
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))

    assert should_send_notification(chat, policy, dt(12)) is True


def test_notification_repeat_interval_blocks_too_early_repeat():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        repeat_after_minutes=240,
        quiet_hours_enabled=False,
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))

    assert should_send_notification(chat, policy, dt(12)) is True

    mark_notification_sent(chat, dt(12), policy)

    assert should_send_notification(chat, policy, dt(15, 59)) is False
    assert should_send_notification(chat, policy, dt(16)) is True


def test_snoozed_chat_does_not_notify_before_snooze_expires():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        quiet_hours_enabled=False,
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))
    snooze_chat(chat, until=dt(15))

    assert chat.status == ChatStatus.SNOOZED
    assert should_send_notification(chat, policy, dt(14, 59)) is False


def test_ignored_chat_does_not_notify():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        quiet_hours_enabled=False,
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))
    ignore_chat(chat)

    assert chat.status == ChatStatus.IGNORED
    assert should_send_notification(chat, policy, dt(13)) is False


def test_mark_handled_clears_waiting_state():
    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))

    mark_handled(chat)

    assert chat.status == ChatStatus.HANDLED
    assert chat.waiting_since is None
    assert chat.snoozed_until is None
    assert chat.next_notification_at is None


def test_daily_notification_limit_blocks_extra_notifications():
    policy = NotificationPolicy(
        waiting_threshold_minutes=120,
        max_notifications_per_chat_per_day=1,
        quiet_hours_enabled=False,
    )

    chat = ChatState(chat_id="chat-1")
    record_inbound_message(chat, dt(10))

    assert should_send_notification(chat, policy, dt(12)) is True

    mark_notification_sent(chat, dt(12), policy)

    assert chat.notification_count_today == 1
    assert should_send_notification(chat, policy, dt(20)) is False


def test_quiet_hours_cross_midnight():
    policy = NotificationPolicy(
        quiet_hours_enabled=True,
        quiet_hours_start=time(22, 0),
        quiet_hours_end=time(8, 0),
    )

    assert is_quiet_hours(dt(23), policy) is True
    assert is_quiet_hours(dt(3), policy) is True
    assert is_quiet_hours(dt(12), policy) is False