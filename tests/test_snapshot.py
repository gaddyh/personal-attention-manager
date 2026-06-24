# tests/test_snapshot.py

from datetime import datetime, timedelta, timezone
from pathlib import Path

from personal_attention_manager.core.domain import (
    ChatStatus,
    NotificationPolicy,
    snooze_chat,
)
from personal_attention_manager.integrations.green_api import MessageEvent
from personal_attention_manager.core.models import MessageDirection as WebhookMessageDirection
from personal_attention_manager.state.snapshot import (
    create_state_snapshot,
    get_resync_start_time,
    load_state_snapshot,
    restore_state_snapshot,
    save_state_snapshot,
    snapshot_from_dict,
)
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
) -> MessageEvent:
    return MessageEvent(
        provider_message_id=message_id,
        chat_id=chat_id,
        chat_name=chat_name,
        direction=direction,
        message_type="textMessage",
        message_time=message_time,
        text=text,
        raw_type_webhook="incomingMessageReceived",
    )


def build_service_with_state() -> InMemoryWaitingStateService:
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(
            waiting_threshold_minutes=120,
            repeat_after_minutes=240,
            quiet_hours_enabled=False,
            max_notifications_per_chat_per_day=3,
        ),
        timezone_name="Asia/Jerusalem",
    )

    service.apply_message_event(
        event(
            message_id="msg-1",
            chat_id="972501111111@c.us",
            chat_name="Dana",
            message_time=dt(10),
            text="Hi, are you available today?",
        )
    )

    service.apply_message_event(
        event(
            message_id="msg-2",
            chat_id="972502222222@c.us",
            chat_name="Yossi",
            message_time=dt(11),
            text="Can you call me?",
        )
    )

    notifications = service.scan_due_notifications(current_time=dt(13))
    assert len(notifications) == 2

    service.mark_notification_sent(notifications[0], sent_at=dt(13))

    yossi = service.get_chat("972502222222@c.us")
    assert yossi is not None
    snooze_chat(yossi, until=dt(15))

    return service


def test_create_snapshot_contains_policy_chats_seen_ids_and_last_processed_time():
    service = build_service_with_state()

    snapshot = create_state_snapshot(
        service,
        reason="test",
        created_at=dt(14),
    )

    assert snapshot.schema_version == 1
    assert snapshot.created_at == dt(14).isoformat()
    assert snapshot.reason == "test"
    assert snapshot.timezone_name == "Asia/Jerusalem"

    assert snapshot.policy.waiting_threshold_minutes == 120
    assert snapshot.policy.repeat_after_minutes == 240
    assert snapshot.policy.quiet_hours_enabled is False

    assert len(snapshot.chats) == 2
    assert sorted(snapshot.seen_provider_message_ids) == ["msg-1", "msg-2"]
    assert snapshot.last_processed_message_at == dt(11).isoformat()

    chat_ids = {chat.chat_id for chat in snapshot.chats}
    assert chat_ids == {"972501111111@c.us", "972502222222@c.us"}


def test_restore_snapshot_recreates_service_state():
    service = build_service_with_state()

    snapshot = create_state_snapshot(service, reason="test")
    restored = restore_state_snapshot(snapshot)

    assert restored.policy.waiting_threshold_minutes == 120
    assert restored.policy.repeat_after_minutes == 240
    assert restored.policy.quiet_hours_enabled is False
    assert str(restored.timezone) == "Asia/Jerusalem"

    dana = restored.get_chat("972501111111@c.us")
    yossi = restored.get_chat("972502222222@c.us")

    assert dana is not None
    assert dana.contact_name == "Dana"
    assert dana.status == ChatStatus.WAITING
    assert dana.waiting_since == dt(10)
    assert dana.last_message_snippet == "Hi, are you available today?"
    assert dana.last_notification_sent_at == dt(13)
    assert dana.notification_count_today == 1

    assert yossi is not None
    assert yossi.contact_name == "Yossi"
    assert yossi.status == ChatStatus.SNOOZED
    assert yossi.snoozed_until == dt(15)

    assert restored.seen_provider_message_ids == {"msg-1", "msg-2"}


def test_restored_service_still_dedupes_replayed_messages():
    service = build_service_with_state()

    snapshot = create_state_snapshot(service, reason="before-resync")
    restored = restore_state_snapshot(snapshot)

    duplicate_result = restored.apply_message_event(
        event(
            message_id="msg-1",
            chat_id="972501111111@c.us",
            chat_name="Dana",
            message_time=dt(12),
            text="Duplicate replay",
        )
    )

    dana = restored.get_chat("972501111111@c.us")

    assert duplicate_result.duplicate is True
    assert dana is not None
    assert dana.waiting_since == dt(10)
    assert dana.last_message_snippet == "Hi, are you available today?"


def test_restored_service_can_apply_new_message_after_recovery():
    service = build_service_with_state()

    snapshot = create_state_snapshot(service, reason="before-resync")
    restored = restore_state_snapshot(snapshot)

    result = restored.apply_message_event(
        event(
            message_id="msg-3",
            chat_id="972501111111@c.us",
            chat_name="Dana",
            message_time=dt(14),
            text="Are you there?",
        )
    )

    dana = restored.get_chat("972501111111@c.us")

    assert result.duplicate is False
    assert result.status == ChatStatus.WAITING

    assert dana is not None
    assert dana.waiting_since == dt(14)
    assert dana.last_message_snippet == "Are you there?"
    assert "msg-3" in restored.seen_provider_message_ids


def test_save_and_load_snapshot_roundtrip(tmp_path: Path):
    service = build_service_with_state()
    path = tmp_path / "snapshots" / "waiting-state.json"

    saved = save_state_snapshot(service, path, reason="test-save")
    loaded = load_state_snapshot(path)
    restored = restore_state_snapshot(loaded)

    assert path.exists()
    assert saved.reason == "test-save"
    assert loaded.reason == "test-save"

    assert restored.get_chat("972501111111@c.us") is not None
    assert restored.get_chat("972502222222@c.us") is not None
    assert restored.seen_provider_message_ids == {"msg-1", "msg-2"}


def test_snapshot_from_dict_rebuilds_snapshot_object():
    service = build_service_with_state()
    snapshot = create_state_snapshot(service, reason="dict-test")

    raw = {
        "schema_version": snapshot.schema_version,
        "created_at": snapshot.created_at,
        "reason": snapshot.reason,
        "timezone_name": snapshot.timezone_name,
        "policy": snapshot.policy.__dict__,
        "chats": [chat.__dict__ for chat in snapshot.chats],
        "seen_provider_message_ids": snapshot.seen_provider_message_ids,
        "notification_count_day_by_chat_id": snapshot.notification_count_day_by_chat_id,
        "last_processed_message_at": snapshot.last_processed_message_at,
    }

    rebuilt = snapshot_from_dict(raw)

    assert rebuilt.reason == "dict-test"
    assert rebuilt.policy.waiting_threshold_minutes == 120
    assert len(rebuilt.chats) == 2
    assert rebuilt.seen_provider_message_ids == ["msg-1", "msg-2"]


def test_get_resync_start_time_uses_overlap():
    service = build_service_with_state()
    snapshot = create_state_snapshot(service, reason="before-resync")

    resync_start = get_resync_start_time(
        snapshot,
        overlap=timedelta(hours=6),
    )

    assert resync_start == dt(5)


def test_get_resync_start_time_returns_none_without_processed_message_time():
    service = InMemoryWaitingStateService()
    snapshot = create_state_snapshot(service, reason="empty")

    assert snapshot.last_processed_message_at is None
    assert get_resync_start_time(snapshot) is None


def test_restore_rejects_unsupported_snapshot_schema_version():
    service = build_service_with_state()
    snapshot = create_state_snapshot(service, reason="bad-version")

    bad_snapshot = snapshot_from_dict(
        {
            "schema_version": 999,
            "created_at": snapshot.created_at,
            "reason": snapshot.reason,
            "timezone_name": snapshot.timezone_name,
            "policy": snapshot.policy.__dict__,
            "chats": [chat.__dict__ for chat in snapshot.chats],
            "seen_provider_message_ids": snapshot.seen_provider_message_ids,
            "notification_count_day_by_chat_id": snapshot.notification_count_day_by_chat_id,
            "last_processed_message_at": snapshot.last_processed_message_at,
        }
    )

    try:
        restore_state_snapshot(bad_snapshot)
    except ValueError as exc:
        assert "Unsupported snapshot schema version" in str(exc)
    else:
        raise AssertionError("Expected restore_state_snapshot to reject bad schema")