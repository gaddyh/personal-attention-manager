# src/personal_attention_manager/waiting_state.py

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from personal_attention_manager.core.domain import (
    mark_notification_sent,
    record_inbound_message,
    record_outbound_message,
    should_send_notification,
)
from personal_attention_manager.integrations.green_api import MessageEvent
from personal_attention_manager.core.models import (
    ChatState,
    ChatStatus,
    MessageDirection,
    MessageEventResult,
    NotificationPolicy,
    WaitingChatNotification,
)


class InMemoryWaitingStateService:
    """
    MVP in-memory waiting-chat state.

    Later this becomes Postgres tables:
    - contacts
    - chat_states
    - message_events
    - notification_events

    For now, this lets us validate the product logic without DB.
    """

    def __init__(
        self,
        policy: NotificationPolicy | None = None,
        timezone_name: str = "Asia/Jerusalem",
    ) -> None:
        self.policy = policy or NotificationPolicy()
        self.timezone = ZoneInfo(timezone_name)

        self.chats_by_id: dict[str, ChatState] = {}
        self.seen_provider_message_ids: set[str] = set()

        # chat_id -> local date when notification_count_today was last reset
        self.notification_count_day_by_chat_id: dict[str, date] = {}

    def apply_message_event(
        self,
        event: MessageEvent,
    ) -> MessageEventResult:
        """
        Apply one inbound/outbound WhatsApp message event.

        Inbound:
            customer wrote to owner -> chat may become waiting.

        Outbound:
            owner replied to customer -> chat becomes handled.
        """

        if event.provider_message_id:
            if event.provider_message_id in self.seen_provider_message_ids:
                chat = self.get_or_create_chat(event.chat_id, event.chat_name)
                return MessageEventResult(
                    chat_id=chat.chat_id,
                    direction=self._normalize_direction(event.direction),
                    status=chat.status,
                    duplicate=True,
                )

            self.seen_provider_message_ids.add(event.provider_message_id)

        chat = self.get_or_create_chat(
            chat_id=event.chat_id,
            contact_name=event.chat_name,
        )

        direction = self._normalize_direction(event.direction)

        if direction == MessageDirection.INBOUND:
            record_inbound_message(
                chat=chat,
                message_time=event.message_time,
                snippet=event.text,
            )

        elif direction == MessageDirection.OUTBOUND:
            record_outbound_message(
                chat=chat,
                message_time=event.message_time,
            )

        else:
            raise ValueError(f"Unsupported message direction: {event.direction}")

        return MessageEventResult(
            chat_id=chat.chat_id,
            direction=direction,
            status=chat.status,
        )

    def get_or_create_chat(
        self,
        chat_id: str,
        contact_name: str | None = None,
        contact_phone: str | None = None,
    ) -> ChatState:
        chat = self.chats_by_id.get(chat_id)

        if chat is None:
            chat = ChatState(
                chat_id=chat_id,
                contact_name=contact_name,
                contact_phone=contact_phone,
            )
            self.chats_by_id[chat_id] = chat
            return chat

        if contact_name and not chat.contact_name:
            chat.contact_name = contact_name

        if contact_phone and not chat.contact_phone:
            chat.contact_phone = contact_phone

        return chat

    def get_chat(self, chat_id: str) -> ChatState | None:
        return self.chats_by_id.get(chat_id)

    def list_chats(self) -> list[ChatState]:
        return list(self.chats_by_id.values())

    def list_waiting_chats(self) -> list[ChatState]:
        return [
            chat
            for chat in self.chats_by_id.values()
            if chat.status == ChatStatus.WAITING
        ]

    def scan_due_notifications(
        self,
        current_time: datetime | None = None,
    ) -> list[WaitingChatNotification]:
        """
        Find chats that should notify the owner now.

        This is the method a scheduler will call every few minutes.
        """

        now = current_time or datetime.now(timezone.utc)
        now_local = self._to_user_time(now)

        notifications: list[WaitingChatNotification] = []

        for chat in self.chats_by_id.values():
            self._reset_daily_notification_count_if_needed(chat, now_local)
            self._release_expired_snooze_or_ignore(chat, now)

            if not should_send_notification(chat, self.policy, now_local):
                continue

            if chat.waiting_since is None:
                continue

            notification = self._build_notification(chat, now)
            notifications.append(notification)

        return notifications

    def mark_notification_sent(
        self,
        notification: WaitingChatNotification,
        sent_at: datetime | None = None,
    ) -> None:
        chat = self.chats_by_id[notification.chat_id]
        now = sent_at or datetime.now(timezone.utc)
        mark_notification_sent(chat, now, self.policy)

    def _build_notification(
        self,
        chat: ChatState,
        current_time: datetime,
    ) -> WaitingChatNotification:
        if chat.waiting_since is None:
            raise ValueError("Cannot build notification without waiting_since")

        waiting_minutes = int(
            (current_time - chat.waiting_since).total_seconds() // 60
        )

        name = chat.contact_name or chat.contact_phone or chat.chat_id
        duration = self._format_minutes(waiting_minutes)

        message = f"{name} is waiting for your reply for {duration}."

        return WaitingChatNotification(
            chat_id=chat.chat_id,
            contact_name=chat.contact_name,
            contact_phone=chat.contact_phone,
            waiting_since=chat.waiting_since,
            waiting_minutes=waiting_minutes,
            message=message,
        )

    def _release_expired_snooze_or_ignore(
        self,
        chat: ChatState,
        current_time: datetime,
    ) -> None:
        """
        Domain functions can mark a chat as snoozed/ignored.
        This service decides when temporary snooze/ignore expires.
        """

        if (
            chat.status == ChatStatus.SNOOZED
            and chat.snoozed_until is not None
            and current_time >= chat.snoozed_until
        ):
            chat.snoozed_until = None

            if chat.last_message_direction == MessageDirection.INBOUND:
                chat.status = ChatStatus.WAITING
            else:
                chat.status = ChatStatus.HANDLED

        if (
            chat.status == ChatStatus.IGNORED
            and chat.ignored_until is not None
            and current_time >= chat.ignored_until
        ):
            chat.ignored_until = None

            if chat.last_message_direction == MessageDirection.INBOUND:
                chat.status = ChatStatus.WAITING
            else:
                chat.status = ChatStatus.HANDLED

    def _reset_daily_notification_count_if_needed(
        self,
        chat: ChatState,
        current_time_local: datetime,
    ) -> None:
        current_day = current_time_local.date()
        previous_day = self.notification_count_day_by_chat_id.get(chat.chat_id)

        if previous_day != current_day:
            chat.notification_count_today = 0
            self.notification_count_day_by_chat_id[chat.chat_id] = current_day

    def _to_user_time(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(self.timezone)

    @staticmethod
    def _normalize_direction(value: Any) -> MessageDirection:
        raw = value.value if hasattr(value, "value") else str(value)

        if raw == MessageDirection.INBOUND:
            return MessageDirection.INBOUND

        if raw == MessageDirection.OUTBOUND:
            return MessageDirection.OUTBOUND

        raise ValueError(f"Unknown message direction: {value}")

    @staticmethod
    def _format_minutes(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes}m"

        hours = minutes // 60
        remaining_minutes = minutes % 60

        if remaining_minutes == 0:
            return f"{hours}h"

        return f"{hours}h {remaining_minutes}m"