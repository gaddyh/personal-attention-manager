# src/personal_attention_manager/scheduler.py

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from personal_attention_manager.config import settings
from personal_attention_manager.state.snapshot import (
    load_state_snapshot,
    restore_state_snapshot,
    save_state_snapshot,
)
from personal_attention_manager.state.waiting_state import (
    InMemoryWaitingStateService,
    WaitingChatNotification,
)

logger = logging.getLogger("waiting-for-you")

DEFAULT_SNAPSHOT_PATH = Path(".data/waiting-state-snapshot.json")

_scheduler_task: asyncio.Task[None] | None = None

_waiting_state_service: InMemoryWaitingStateService = InMemoryWaitingStateService(
    timezone_name=settings.timezone,
)


def get_waiting_state_service() -> InMemoryWaitingStateService:
    """
    Shared runtime state.

    Webhook ingestion and scheduler scanning must use the same service instance.
    Later this will be replaced by a real repository / database-backed service.
    """

    return _waiting_state_service


def start_scheduler() -> None:
    """
    Start the background scheduler.

    Called from FastAPI lifespan startup.
    """

    global _scheduler_task
    global _waiting_state_service

    if _scheduler_task is not None and not _scheduler_task.done():
        logger.info("Scheduler already running")
        return

    snapshot_path = get_snapshot_path()

    if snapshot_path.exists():
        try:
            snapshot = load_state_snapshot(snapshot_path)
            _waiting_state_service = restore_state_snapshot(snapshot)

            logger.info(
                "Restored waiting-state snapshot path=%s created_at=%s reason=%s chats=%s",
                snapshot_path,
                snapshot.created_at,
                snapshot.reason,
                len(snapshot.chats),
            )
        except Exception:
            logger.exception(
                "Failed to restore snapshot path=%s. Starting with empty state.",
                snapshot_path,
            )
            _waiting_state_service = InMemoryWaitingStateService(
                timezone_name=settings.timezone,
            )
    else:
        logger.info("No snapshot found. Starting with empty waiting state.")

    _scheduler_task = asyncio.create_task(
        _scheduler_loop(),
        name="waiting-for-you-scheduler",
    )

    logger.info("Scheduler started")


def stop_scheduler() -> None:
    """
    Stop the background scheduler.

    Called from FastAPI lifespan shutdown.
    """

    global _scheduler_task

    logger.info("Stopping scheduler")

    try:
        save_runtime_snapshot(reason="shutdown")
    except Exception:
        logger.exception("Failed saving shutdown snapshot")

    if _scheduler_task is not None:
        _scheduler_task.cancel()
        _scheduler_task = None

    logger.info("Scheduler stopped")


async def _scheduler_loop() -> None:
    interval_seconds = get_scheduler_interval_seconds()

    logger.info("Scheduler loop running interval_seconds=%s", interval_seconds)

    try:
        while True:
            await run_scheduler_tick()
            await asyncio.sleep(interval_seconds)

    except asyncio.CancelledError:
        logger.info("Scheduler loop cancelled")
        raise


async def run_scheduler_tick() -> None:
    """
    One scheduler iteration.

    Responsibilities:
    - scan waiting chats
    - send due owner notifications
    - mark notifications as sent
    - save a durable snapshot

    This function is intentionally public so tests can call it directly.
    """

    service = get_waiting_state_service()
    now = datetime.now(timezone.utc)

    notifications = service.scan_due_notifications(current_time=now)

    if notifications:
        logger.info("Found %s due waiting-chat notifications", len(notifications))

    for notification in notifications:
        try:
            await send_owner_notification(notification)
            service.mark_notification_sent(notification, sent_at=now)

        except Exception:
            logger.exception(
                "Failed sending notification chat_id=%s",
                notification.chat_id,
            )

    save_runtime_snapshot(reason="scheduler_tick")


async def send_owner_notification(
    notification: WaitingChatNotification,
) -> None:
    """
    Temporary notification sender.

    Later this should call the official WABA sender.

    For now we only log the notification so the scheduler is runnable before
    WABA integration exists.
    """

    logger.info(
        "OWNER NOTIFICATION chat_id=%s message=%s",
        notification.chat_id,
        notification.message,
    )


def save_runtime_snapshot(
    *,
    reason: str,
) -> None:
    service = get_waiting_state_service()
    snapshot_path = get_snapshot_path()

    snapshot = save_state_snapshot(
        service,
        snapshot_path,
        reason=reason,
    )

    logger.info(
        "Saved waiting-state snapshot path=%s reason=%s chats=%s last_processed_message_at=%s",
        snapshot_path,
        reason,
        len(snapshot.chats),
        snapshot.last_processed_message_at,
    )


def get_snapshot_path() -> Path:
    return Path(
        os.getenv(
            "WAITING_STATE_SNAPSHOT_PATH",
            str(DEFAULT_SNAPSHOT_PATH),
        )
    )


def get_scheduler_interval_seconds() -> int:
    raw_value = os.getenv("SCHEDULER_INTERVAL_SECONDS", "60")

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid SCHEDULER_INTERVAL_SECONDS=%r. Falling back to 60.",
            raw_value,
        )
        return 60

    return max(value, 5)