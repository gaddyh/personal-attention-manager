# src/personal_attention_manager/main.py

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request

from personal_attention_manager.config import settings
from personal_attention_manager.api.scheduler import start_scheduler, stop_scheduler
from personal_attention_manager.api.security import verify_green_api_authorization
from personal_attention_manager.api.webhook_handler import handle_green_api_webhook


_log_level = getattr(
    logging,
    os.getenv("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)

logging.basicConfig(level=_log_level)

logger = logging.getLogger("waiting-for-you")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle.

    Startup:
      - initialize shared app state
      - start background scheduler

    Shutdown:
      - stop scheduler cleanly
      - save final state snapshot
    """

    logger.info("Starting waiting-for-you app")

    start_scheduler()

    try:
        yield
    finally:
        logger.info("Stopping waiting-for-you app")
        stop_scheduler()


app = FastAPI(
    title="waiting-for-you",
    description="Never miss a client who is waiting for your WhatsApp reply.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "waiting-for-you",
    }


@app.get("/debug/settings")
async def debug_settings() -> dict[str, Any]:
    """
    Temporary dev endpoint.

    Do not return secrets.
    Remove or protect this before production.
    """

    return {
        "green_api_base_url": settings.green_api_base_url,
        "green_api_id_instance_configured": bool(settings.green_api_id_instance),
        "green_api_token_instance_configured": bool(
            settings.green_api_token_instance
        ),
        "webhook_secret_configured": bool(settings.webhook_secret),
        "allowed_chat_ids_count": len(settings.allowed_chat_ids),
        "timezone": settings.timezone,
    }


@app.post("/webhooks/green-api")
async def green_api_webhook(request: Request) -> dict[str, Any]:
    """
    Green API webhook endpoint.

    This endpoint only ingests message events.
    It does not reply to the customer's WhatsApp chat.
    """

    verify_green_api_authorization(request)

    payload = await request.json()

    logger.info(
        "Green API webhook received type=%s idMessage=%s",
        payload.get("typeWebhook"),
        payload.get("idMessage"),
    )

    return await handle_green_api_webhook(payload)