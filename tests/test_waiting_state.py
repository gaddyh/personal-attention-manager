# tests/test_waiting_state.py

from datetime import datetime, timedelta, timezone

from personal_attention_manager.core.domain import (
    ChatStatus,
    NotificationPolicy,
    snooze_chat,
)
from personal_attention_manager.integrations.green_api import GreenApiMessageEvent
from personal_attention_manager.core.models import MessageDirection as WebhookMessageDirection
from personal_attention_manager.state.waiting_state import InMemoryWaitingStateService


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


def event(
    *,
    message_id: str,
    chat_id: str = "972501111111@c.us",
    chat_name: str = "Dana",
    direction: WebhookMessageDirection = WebhookMessageDirection.INBOUND,
    message_time: datetime = dt(10),
    text: str | None = "Hi, are you available today?",
) -> GreenApiMessageEvent:
    return GreenApiMessageEvent(
        provider_message_id=message_id,
        chat_id=chat_id,
        chat_name=chat_name,
        direction=direction,
        message_type="textMessage",
        message_time=message_time,
        text=text,
        raw_type_webhook="incomingMessageReceived",
    )


def test_inbound_event_creates_waiting_chat():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(quiet_hours_enabled=False)
    )

    result = service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    chat = service.get_chat("972501111111@c.us")

    assert result.duplicate is False
    assert result.status == ChatStatus.WAITING

    assert chat is not None
    assert chat.chat_id == "972501111111@c.us"
    assert chat.contact_name == "Dana"
    assert chat.status == ChatStatus.WAITING
    assert chat.waiting_since == dt(10)
    assert chat.last_message_snippet == "Hi, are you available today?"


def test_outbound_event_marks_chat_as_handled():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(quiet_hours_enabled=False)
    )

    service.apply_message_event(
        event(
            message_id="msg-1",
            direction=WebhookMessageDirection.INBOUND,
            message_time=dt(10),
        )
    )

    result = service.apply_message_event(
        event(
            message_id="msg-2",
            direction=WebhookMessageDirection.OUTBOUND,
            message_time=dt(11),
            text="Yes, I am available.",
        )
    )

    chat = service.get_chat("972501111111@c.us")

    assert result.status == ChatStatus.HANDLED

    assert chat is not None
    assert chat.status == ChatStatus.HANDLED
    assert chat.waiting_since is None
    assert chat.last_outbound_message_at == dt(11)


def test_duplicate_message_event_is_ignored():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(quiet_hours_enabled=False)
    )

    first_result = service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    duplicate_result = service.apply_message_event(
        event(message_id="msg-1", message_time=dt(11), text="Duplicate")
    )

    chat = service.get_chat("972501111111@c.us")

    assert first_result.duplicate is False
    assert duplicate_result.duplicate is True

    assert chat is not None
    assert chat.waiting_since == dt(10)
    assert chat.last_message_snippet == "Hi, are you available today?"


def test_scan_due_notifications_returns_waiting_chat_after_threshold():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    notifications = service.scan_due_notifications(current_time=dt(12))

    assert len(notifications) == 1

    notification = notifications[0]

    assert notification.chat_id == "972501111111@c.us"
    assert notification.contact_name == "Dana"
    assert notification.waiting_since == dt(10)
    assert notification.waiting_minutes == 120
    assert notification.message == "Dana is waiting for your reply for 2h."


def test_scan_due_notifications_does_not_notify_before_threshold():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    notifications = service.scan_due_notifications(current_time=dt(11, 59))

    assert notifications == []


def test_mark_notification_sent_blocks_repeat_until_repeat_interval():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            repeat_after_minutes=240,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    notifications = service.scan_due_notifications(current_time=dt(12))
    assert len(notifications) == 1

    service.mark_notification_sent(notifications[0], sent_at=dt(12))

    too_early = service.scan_due_notifications(current_time=dt(15, 59))
    due_again = service.scan_due_notifications(current_time=dt(16))

    assert too_early == []
    assert len(due_again) == 1


def test_daily_notification_limit_blocks_extra_notifications():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            repeat_after_minutes=60,
            max_notifications_per_chat_per_day=1,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    first = service.scan_due_notifications(current_time=dt(12))
    assert len(first) == 1

    service.mark_notification_sent(first[0], sent_at=dt(12))

    second = service.scan_due_notifications(current_time=dt(13))
    assert second == []


def test_snoozed_chat_does_not_notify_until_snooze_expires():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(message_id="msg-1", message_time=dt(10))
    )

    chat = service.get_chat("972501111111@c.us")
    assert chat is not None

    snooze_chat(chat, until=dt(15))

    before_expiry = service.scan_due_notifications(current_time=dt(14, 59))
    after_expiry = service.scan_due_notifications(current_time=dt(15))

    assert before_expiry == []
    assert len(after_expiry) == 1
    assert chat.status == ChatStatus.WAITING


def test_outbound_reply_prevents_notification():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(
            message_id="msg-1",
            direction=WebhookMessageDirection.INBOUND,
            message_time=dt(10),
        )
    )

    service.apply_message_event(
        event(
            message_id="msg-2",
            direction=WebhookMessageDirection.OUTBOUND,
            message_time=dt(11),
            text="Replied.",
        )
    )

    notifications = service.scan_due_notifications(current_time=dt(13))

    assert notifications == []


def test_non_text_inbound_message_still_counts_as_waiting():
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            quiet_hours_enabled=False,
        )
    )

    service.apply_message_event(
        event(
            message_id="msg-1",
            message_time=dt(10),
            text=None,
        )
    )

    chat = service.get_chat("972501111111@c.us")
    notifications = service.scan_due_notifications(current_time=dt(12))

    assert chat is not None
    assert chat.status == ChatStatus.WAITING
    assert chat.last_message_snippet is None
    assert len(notifications) == 1