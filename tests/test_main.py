# tests/test_main.py

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from personal_attention_manager.core.domain import ChatStatus, NotificationPolicy
from personal_attention_manager.state.waiting_state import InMemoryWaitingStateService


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=timezone.utc)


def green_api_payload(
    *,
    type_webhook: str = "incomingMessageReceived",
    message_id: str = "msg-1",
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


def import_app_with_test_env(monkeypatch, tmp_path: Path):
    """
    Import the FastAPI app after setting env vars.

    Important because config.py creates `settings` at import time.
    """

    monkeypatch.setenv("GREEN_API_ID_INSTANCE", "test-instance")
    monkeypatch.setenv("GREEN_API_TOKEN_INSTANCE", "test-token")
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "")
    monkeypatch.setenv("TIMEZONE", "Asia/Jerusalem")
    monkeypatch.setenv("SCHEDULER_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv(
        "WAITING_STATE_SNAPSHOT_PATH",
        str(tmp_path / "waiting-state-snapshot.json"),
    )

    import personal_attention_manager.config as config
    import personal_attention_manager.api.scheduler as scheduler
    import personal_attention_manager.api.security as security
    import personal_attention_manager.api.webhook_handler as webhook_handler
    import personal_attention_manager.api.main as main

    importlib.reload(config)
    importlib.reload(scheduler)
    importlib.reload(security)
    importlib.reload(webhook_handler)
    importlib.reload(main)

    service = InMemoryWaitingStateService(
        policy=NotificationPolicy(quiet_hours_enabled=False),
        timezone_name="Asia/Jerusalem",
    )

    monkeypatch.setattr(scheduler, "_waiting_state_service", service)

    return main.app, scheduler, service


def test_health_endpoint(monkeypatch, tmp_path: Path):
    app, _, _ = import_app_with_test_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "waiting-for-you",
    }


def test_green_api_endpoint_rejects_invalid_authorization(monkeypatch, tmp_path: Path):
    app, _, _ = import_app_with_test_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/green-api",
            json=green_api_payload(),
            headers={"Authorization": "Bearer wrong-secret"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid webhook authorization"


def test_green_api_endpoint_ingests_inbound_message(monkeypatch, tmp_path: Path):
    app, _, service = import_app_with_test_env(monkeypatch, tmp_path)

    payload = green_api_payload(
        type_webhook="incomingMessageReceived",
        message_id="msg-1",
        message_time=dt(10),
        text="Hi, are you available today?",
    )

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/green-api",
            json=payload,
            headers={"Authorization": "Bearer test-secret"},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["ok"] is True
    assert body["handled"] == "message_event"
    assert body["duplicate"] is False

    assert body["event"]["providerMessageId"] == "msg-1"
    assert body["event"]["chatId"] == "972501111111@c.us"
    assert body["event"]["chatName"] == "Dana"
    assert body["event"]["direction"] == "inbound"
    assert body["event"]["hasText"] is True

    assert body["chatState"]["status"] == "waiting"
    assert body["chatState"]["waitingSince"] == dt(10).isoformat()

    chat = service.get_chat("972501111111@c.us")

    assert chat is not None
    assert chat.status == ChatStatus.WAITING
    assert chat.waiting_since == dt(10)
    assert chat.last_message_snippet == "Hi, are you available today?"


def test_green_api_endpoint_ingests_outbound_reply(monkeypatch, tmp_path: Path):
    app, _, service = import_app_with_test_env(monkeypatch, tmp_path)

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

    with TestClient(app) as client:
        first_response = client.post(
            "/webhooks/green-api",
            json=inbound_payload,
            headers={"Authorization": "Bearer test-secret"},
        )

        second_response = client.post(
            "/webhooks/green-api",
            json=outbound_payload,
            headers={"Authorization": "Bearer test-secret"},
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    body = second_response.json()

    assert body["ok"] is True
    assert body["event"]["direction"] == "outbound"
    assert body["chatState"]["status"] == "handled"
    assert body["chatState"]["waitingSince"] is None

    chat = service.get_chat("972501111111@c.us")

    assert chat is not None
    assert chat.status == ChatStatus.HANDLED
    assert chat.waiting_since is None
    assert chat.last_outbound_message_at == dt(11)


def test_green_api_endpoint_saves_snapshot(monkeypatch, tmp_path: Path):
    app, _, _ = import_app_with_test_env(monkeypatch, tmp_path)

    snapshot_path = tmp_path / "waiting-state-snapshot.json"

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/green-api",
            json=green_api_payload(),
            headers={"Authorization": "Bearer test-secret"},
        )

    assert response.status_code == 200
    assert snapshot_path.exists()

    snapshot_text = snapshot_path.read_text(encoding="utf-8")

    assert '"reason": "shutdown"' in snapshot_text or '"reason": "webhook_event"' in snapshot_text
    assert '"chat_id": "972501111111@c.us"' in snapshot_text
    assert '"msg-1"' in snapshot_text