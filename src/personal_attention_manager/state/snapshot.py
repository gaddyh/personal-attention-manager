# src/personal_attention_manager/snapshot.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from personal_attention_manager.core.models import (
    ChatState,
    ChatStatus,
    MessageDirection,
    NotificationPolicy,
)
from personal_attention_manager.state.waiting_state import InMemoryWaitingStateService


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NotificationPolicySnapshot:
    enabled: bool
    waiting_threshold_minutes: int
    repeat_after_minutes: int

    quiet_hours_enabled: bool
    quiet_hours_start: str
    quiet_hours_end: str

    daily_digest_enabled: bool
    daily_digest_time: str

    max_notifications_per_chat_per_day: int


@dataclass(frozen=True)
class ChatStateSnapshot:
    chat_id: str
    contact_name: str | None
    contact_phone: str | None

    last_inbound_message_at: str | None
    last_outbound_message_at: str | None
    last_message_direction: str | None

    waiting_since: str | None
    status: str

    snoozed_until: str | None
    ignored_until: str | None

    last_notification_sent_at: str | None
    next_notification_at: str | None

    notification_count_today: int

    last_message_snippet: str | None


@dataclass(frozen=True)
class WaitingStateSnapshot:
    """
    Durable snapshot of the waiting-chat state.

    This is not a full message backup.
    It is the minimum state needed to recover the product behavior after
    downtime, reconnect, deploy, or process restart.
    """

    schema_version: int
    created_at: str
    reason: str

    timezone_name: str

    policy: NotificationPolicySnapshot
    chats: list[ChatStateSnapshot]

    seen_provider_message_ids: list[str]
    notification_count_day_by_chat_id: dict[str, str]

    last_processed_message_at: str | None


def create_state_snapshot(
    service: InMemoryWaitingStateService,
    *,
    reason: str = "manual",
    created_at: datetime | None = None,
) -> WaitingStateSnapshot:
    now = created_at or datetime.now(timezone.utc)

    return WaitingStateSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        created_at=_dt_to_iso(now),
        reason=reason,
        timezone_name=str(service.timezone),
        policy=_policy_to_snapshot(service.policy),
        chats=[
            _chat_to_snapshot(chat)
            for chat in service.list_chats()
        ],
        seen_provider_message_ids=sorted(service.seen_provider_message_ids),
        notification_count_day_by_chat_id={
            chat_id: day.isoformat()
            for chat_id, day in service.notification_count_day_by_chat_id.items()
        },
        last_processed_message_at=_dt_to_iso(
            _get_last_processed_message_at(service)
        ),
    )


def restore_state_snapshot(
    snapshot: WaitingStateSnapshot,
) -> InMemoryWaitingStateService:
    _validate_snapshot(snapshot)

    service = InMemoryWaitingStateService(
        policy=_policy_from_snapshot(snapshot.policy),
        timezone_name=snapshot.timezone_name,
    )

    service.chats_by_id = {
        chat_snapshot.chat_id: _chat_from_snapshot(chat_snapshot)
        for chat_snapshot in snapshot.chats
    }

    service.seen_provider_message_ids = set(snapshot.seen_provider_message_ids)

    service.notification_count_day_by_chat_id = {
        chat_id: date.fromisoformat(day_text)
        for chat_id, day_text in snapshot.notification_count_day_by_chat_id.items()
    }

    return service


def save_state_snapshot(
    service: InMemoryWaitingStateService,
    path: Path,
    *,
    reason: str = "manual",
) -> WaitingStateSnapshot:
    snapshot = create_state_snapshot(service, reason=reason)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(snapshot), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return snapshot


def load_state_snapshot(path: Path) -> WaitingStateSnapshot:
    data = json.loads(path.read_text(encoding="utf-8"))
    return snapshot_from_dict(data)


def snapshot_from_dict(data: dict[str, Any]) -> WaitingStateSnapshot:
    return WaitingStateSnapshot(
        schema_version=data["schema_version"],
        created_at=data["created_at"],
        reason=data["reason"],
        timezone_name=data["timezone_name"],
        policy=NotificationPolicySnapshot(**data["policy"]),
        chats=[
            ChatStateSnapshot(**chat_data)
            for chat_data in data["chats"]
        ],
        seen_provider_message_ids=list(data["seen_provider_message_ids"]),
        notification_count_day_by_chat_id=dict(
            data["notification_count_day_by_chat_id"]
        ),
        last_processed_message_at=data.get("last_processed_message_at"),
    )


def get_resync_start_time(
    snapshot: WaitingStateSnapshot,
    *,
    overlap: timedelta = timedelta(hours=6),
) -> datetime | None:
    """
    When Green API comes back, fetch/replay messages from this time.

    We intentionally go back with overlap because:
    - webhooks may arrive late
    - timestamps may be slightly inconsistent
    - duplicate replay is safe because provider_message_id is deduped

    Example:
        last processed message: 14:00
        overlap: 6h
        resync from: 08:00
    """

    if snapshot.last_processed_message_at is None:
        return None

    last_processed = _dt_from_iso(snapshot.last_processed_message_at)
    return last_processed - overlap


def _policy_to_snapshot(
    policy: NotificationPolicy,
) -> NotificationPolicySnapshot:
    return NotificationPolicySnapshot(
        enabled=policy.enabled,
        waiting_threshold_minutes=policy.waiting_threshold_minutes,
        repeat_after_minutes=policy.repeat_after_minutes,
        quiet_hours_enabled=policy.quiet_hours_enabled,
        quiet_hours_start=_time_to_text(policy.quiet_hours_start),
        quiet_hours_end=_time_to_text(policy.quiet_hours_end),
        daily_digest_enabled=policy.daily_digest_enabled,
        daily_digest_time=_time_to_text(policy.daily_digest_time),
        max_notifications_per_chat_per_day=policy.max_notifications_per_chat_per_day,
    )


def _policy_from_snapshot(
    snapshot: NotificationPolicySnapshot,
) -> NotificationPolicy:
    return NotificationPolicy(
        enabled=snapshot.enabled,
        waiting_threshold_minutes=snapshot.waiting_threshold_minutes,
        repeat_after_minutes=snapshot.repeat_after_minutes,
        quiet_hours_enabled=snapshot.quiet_hours_enabled,
        quiet_hours_start=_time_from_text(snapshot.quiet_hours_start),
        quiet_hours_end=_time_from_text(snapshot.quiet_hours_end),
        daily_digest_enabled=snapshot.daily_digest_enabled,
        daily_digest_time=_time_from_text(snapshot.daily_digest_time),
        max_notifications_per_chat_per_day=snapshot.max_notifications_per_chat_per_day,
    )


def _chat_to_snapshot(chat: ChatState) -> ChatStateSnapshot:
    return ChatStateSnapshot(
        chat_id=chat.chat_id,
        contact_name=chat.contact_name,
        contact_phone=chat.contact_phone,
        last_inbound_message_at=_dt_to_iso(chat.last_inbound_message_at),
        last_outbound_message_at=_dt_to_iso(chat.last_outbound_message_at),
        last_message_direction=(
            chat.last_message_direction.value
            if chat.last_message_direction
            else None
        ),
        waiting_since=_dt_to_iso(chat.waiting_since),
        status=chat.status.value,
        snoozed_until=_dt_to_iso(chat.snoozed_until),
        ignored_until=_dt_to_iso(chat.ignored_until),
        last_notification_sent_at=_dt_to_iso(chat.last_notification_sent_at),
        next_notification_at=_dt_to_iso(chat.next_notification_at),
        notification_count_today=chat.notification_count_today,
        last_message_snippet=chat.last_message_snippet,
    )


def _chat_from_snapshot(snapshot: ChatStateSnapshot) -> ChatState:
    return ChatState(
        chat_id=snapshot.chat_id,
        contact_name=snapshot.contact_name,
        contact_phone=snapshot.contact_phone,
        last_inbound_message_at=_dt_from_iso_or_none(
            snapshot.last_inbound_message_at
        ),
        last_outbound_message_at=_dt_from_iso_or_none(
            snapshot.last_outbound_message_at
        ),
        last_message_direction=(
            MessageDirection(snapshot.last_message_direction)
            if snapshot.last_message_direction
            else None
        ),
        waiting_since=_dt_from_iso_or_none(snapshot.waiting_since),
        status=ChatStatus(snapshot.status),
        snoozed_until=_dt_from_iso_or_none(snapshot.snoozed_until),
        ignored_until=_dt_from_iso_or_none(snapshot.ignored_until),
        last_notification_sent_at=_dt_from_iso_or_none(
            snapshot.last_notification_sent_at
        ),
        next_notification_at=_dt_from_iso_or_none(snapshot.next_notification_at),
        notification_count_today=snapshot.notification_count_today,
        last_message_snippet=snapshot.last_message_snippet,
    )


def _get_last_processed_message_at(
    service: InMemoryWaitingStateService,
) -> datetime | None:
    latest: datetime | None = None

    for chat in service.list_chats():
        candidates = [
            chat.last_inbound_message_at,
            chat.last_outbound_message_at,
        ]

        for candidate in candidates:
            if candidate is None:
                continue

            if latest is None or candidate > latest:
                latest = candidate

    return latest


def _validate_snapshot(snapshot: WaitingStateSnapshot) -> None:
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema version: {snapshot.schema_version}"
        )


def _dt_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.isoformat()


def _dt_from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _dt_from_iso_or_none(value: str | None) -> datetime | None:
    if value is None:
        return None

    return _dt_from_iso(value)


def _time_to_text(value: time) -> str:
    return value.strftime("%H:%M")


def _time_from_text(value: str) -> time:
    hour_text, minute_text = value.split(":", maxsplit=1)
    return time(hour=int(hour_text), minute=int(minute_text))