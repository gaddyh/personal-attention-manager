# tests/test_webhook_handler.py

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import personal_attention_manager.api.scheduler as scheduler
import personal_attention_manager.api.webhook_handler as webhook_handler
from personal_attention_manager.core.domain import ChatStatus, NotificationPolicy
from personal_attention_manager.state.waiting_state import InMemoryWaitingStateService


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


def green_api_payload(
    *,
    type_webhook: str,
    message_id: str,
    chat_id: str = "972501111111@c.us",
    chat_name: str = "Dana",
    message_time: datetime = dt(10),
    text: str = "Hi, are you available today?",
) -> dict:
    return {
        "typeWebhook": type_webhook,
        "idMessage": message_id,
        "timestamp": int(message_time.timestamp()),
        "senderData": {
            "chatId": chat_id,
            "chatName": chat_name,
        },
        "messageData": {
            "typeMessage": "textMessage",
            "textMessageData": {
                "textMessage": text,
            },
        },
    }


def setup_test_service(
    monkeypatch,
    tmp_path: Path,
    *,
    allowed_chat_ids: set[str] | None = None,
) -> InMemoryWaitingStateService:
    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(quiet_hours_enabled=False),
        timezone_name="Asia/Jerusalem",
    )

    monkeypatch.setattr(scheduler, "_waiting_state_service", service)

    monkeypatch.setenv(
        "WAITING_STATE_SNAPSHOT_PATH",
        str(tmp_path / "waiting-state-snapshot.json"),
    )

    monkeypatch.setattr(
        webhook_handler,
        "settings",
        SimpleNamespace(allowed_chat_ids=allowed_chat_ids or set()),
    )

    return service


def test_inbound_webhook_updates_waiting_state(monkeypatch, tmp_path: Path):
    service = setup_test_service(monkeypatch, tmp_path)

    payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(10),
        text="Hi, are you available today?",
    )

    result = asyncio.run(webhook_handler.handle_green_api_webhook(payload))

    chat = service.get_chat("972501111111@c.us")

    assert result["ok"] is True
    assert result["handled"] == "message_event"
    assert result["duplicate"] is False

    assert result["event"]["providerMessageId"] == "msg-1"
    assert result["event"]["chatId"] == "972501111111@c.us"
    assert result["event"]["chatName"] == "Dana"
    assert result["event"]["direction"] == "inbound"
    assert result["event"]["hasText"] is True

    assert result["chatState"]["status"] == "waiting"
    assert result["chatState"]["waitingSince"] == dt(10).isoformat()

    assert chat is not None
    assert chat.status == ChatStatus.WAITING
    assert chat.waiting_since == dt(10)
    assert chat.last_message_snippet == "Hi, are you available today?"


def test_outbound_webhook_marks_chat_handled(monkeypatch, tmp_path: Path):
    service = setup_test_service(monkeypatch, tmp_path)

    inbound_payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(10),
        text="Hi, are you available today?",
    )

    outbound_payload = green_api_payload(
        type_webhook="outgoingMessageReceived",
        message_id="msg-2",
        message_time=dt(11),
        text="Yes, I am available.",
    )

    asyncio.run(webhook_handler.handle_green_api_webhook(inbound_payload))
    result = asyncio.run(webhook_handler.handle_green_api_webhook(outbound_payload))

    chat = service.get_chat("972501111111@c.us")

    assert result["ok"] is True
    assert result["handled"] == "message_event"
    assert result["event"]["direction"] == "outbound"
    assert result["chatState"]["status"] == "handled"
    assert result["chatState"]["waitingSince"] is None

    assert chat is not None
    assert chat.status == ChatStatus.HANDLED
    assert chat.waiting_since is None
    assert chat.last_outbound_message_at == dt(11)


def test_duplicate_webhook_does_not_change_state(monkeypatch, tmp_path: Path):
    service = setup_test_service(monkeypatch, tmp_path)

    first_payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(10),
        text="First message",
    )

    duplicate_payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(11),
        text="Duplicate message",
    )

    first_result = asyncio.run(webhook_handler.handle_green_api_webhook(first_payload))
    duplicate_result = asyncio.run(
        webhook_handler.handle_green_api_webhook(duplicate_payload)
    )

    chat = service.get_chat("972501111111@c.us")

    assert first_result["duplicate"] is False
    assert duplicate_result["duplicate"] is True

    assert chat is not None
    assert chat.waiting_since == dt(10)
    assert chat.last_message_snippet == "First message"


def test_unsupported_webhook_type_is_ignored(monkeypatch, tmp_path: Path):
    setup_test_service(monkeypatch, tmp_path)

    payload = green_api_payload(
        type_webhook="stateInstanceChanged",
        message_id="msg-1",
    )

    result = asyncio.run(webhook_handler.handle_green_api_webhook(payload))

    assert result == {
        "ok": True,
        "ignored": "unsupported_webhook_type",
        "typeWebhook": "stateInstanceChanged",
    }


def test_non_allowed_chat_is_ignored(monkeypatch, tmp_path: Path):
    service = setup_test_service(
        monkeypatch,
        tmp_path,
        allowed_chat_ids={"allowed-chat@c.us"},
    )

    payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        chat_id="blocked-chat@c.us",
        chat_name="Blocked",
    )

    result = asyncio.run(webhook_handler.handle_green_api_webhook(payload))

    assert result == {
        "ok": True,
        "ignored": "chat_not_allowed",
        "chatId": "blocked-chat@c.us",
        "chatName": "Blocked",
    }

    assert service.get_chat("blocked-chat@c.us") is None


def test_webhook_event_saves_snapshot(monkeypatch, tmp_path: Path):
    setup_test_service(monkeypatch, tmp_path)

    snapshot_path = tmp_path / "waiting-state-snapshot.json"

    payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(10),
    )

    asyncio.run(webhook_handler.handle_green_api_webhook(payload))

    assert snapshot_path.exists()

    snapshot_text = snapshot_path.read_text(encoding="utf-8")

    assert '"reason": "webhook_event"' in snapshot_text
    assert '"chat_id": "972501111111@c.us"' in snapshot_text
    assert '"seen_provider_message_ids": [' in snapshot_text
    assert '"msg-1"' in snapshot_text