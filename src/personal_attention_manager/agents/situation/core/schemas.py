# situation/core/schemas.py

from __future__ import annotations

import enum
from pydantic import BaseModel
from personal_attention_manager.agents.shared.schemas import Message

class SituationType(enum.Enum):
    QUESTION_NEEDS_REPLY = "question_needs_reply"
    USER_ACTION_REQUESTED = "user_action_requested"
    USER_DECISION_NEEDED = "user_decision_needed"
    USER_INFO_OR_ARTIFACT_NEEDED = "user_info_or_artifact_needed"
    SCHEDULING_COORDINATION = "scheduling_coordination"
    WAITING_FOR_OTHER_PERSON = "waiting_for_other_person"
    FYI_ONLY = "fyi_only"
    RESOLVED = "resolved"
    SOCIAL_OR_EMOTIONAL = "social_or_emotional"
    UNCLEAR = "unclear"


class Urgency(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SituationInput(BaseModel):
    recent_messages: list[Message]
    chat_type: str | None = None


class SituationOutput(BaseModel):
    situation_type: SituationType
    should_user_respond: bool
    is_waiting_on_user: bool
    is_waiting_on_other: bool
    urgency: Urgency
    confidence: float
    short_summary: str
    next_action: str | None
    reasoning: str