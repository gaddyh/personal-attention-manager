# src/personal_attention_manager/webhook_handler.py

from __future__ import annotations

import logging
from typing import Any

from personal_attention_manager.config import settings
from personal_attention_manager.api.scheduler import (
    get_waiting_state_service,
    save_runtime_snapshot,
)
from personal_attention_manager.integrations.green_api import normalize_green_api_message_event

logger = logging.getLogger("waiting-for-you")


async def handle_green_api_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Green API webhook entry point.

    This handler only ingests WhatsApp message events.

    It does not:
    - call an AI agent
    - reply to the customer through Green API
    - transcribe audio
    - send business notifications directly

    The scheduler is responsible for notification timing.
    """

    event = normalize_green_api_message_event(payload)

    if event is None:
        logger.info(
            "Ignoring unsupported webhook type=%s",
            payload.get("typeWebhook"),
        )

        return {
            "ok": True,
            "ignored": "unsupported_webhook_type",
            "typeWebhook": payload.get("typeWebhook"),
        }

    logger.info(
        "Green API message event chat_id=%s chat_name=%s direction=%s type=%s",
        event.chat_id,
        event.chat_name,
        event.direction,
        event.message_type,
    )

    if settings.allowed_chat_ids and event.chat_id not in settings.allowed_chat_ids:
        logger.info(
            "Ignoring message from non-allowed chat_id=%s chat_name=%s",
            event.chat_id,
            event.chat_name,
        )

        return {
            "ok": True,
            "ignored": "chat_not_allowed",
            "chatId": event.chat_id,
            "chatName": event.chat_name,
        }

    service = get_waiting_state_service()
    result = service.apply_message_event(event)

    save_runtime_snapshot(reason="webhook_event")

    chat = service.get_chat(event.chat_id)

    return {
        "ok": True,
        "handled": "message_event",
        "duplicate": result.duplicate,
        "event": {
            "providerMessageId": event.provider_message_id,
            "chatId": event.chat_id,
            "chatName": event.chat_name,
            "direction": event.direction.value,
            "messageType": event.message_type,
            "messageTime": event.message_time.isoformat(),
            "hasText": event.text is not None,
        },
        "chatState": None
        if chat is None
        else {
            "chatId": chat.chat_id,
            "contactName": chat.contact_name,
            "status": chat.status.value,
            "waitingSince": (
                chat.waiting_since.isoformat()
                if chat.waiting_since
                else None
            ),
            "lastInboundMessageAt": (
                chat.last_inbound_message_at.isoformat()
                if chat.last_inbound_message_at
                else None
            ),
            "lastOutboundMessageAt": (
                chat.last_outbound_message_at.isoformat()
                if chat.last_outbound_message_at
                else None
            ),
            "lastMessageDirection": (
                chat.last_message_direction.value
                if chat.last_message_direction
                else None
            ),
        },
    }