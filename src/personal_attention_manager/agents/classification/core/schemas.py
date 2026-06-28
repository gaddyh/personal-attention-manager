
import enum
from pydantic import BaseModel, Field
from personal_attention_manager.agents.shared.schemas import Message

class ChatType(enum.Enum):
    FAMILY = "family"
    WORK = "work"
    HOME = "home"
    OTHER = "other"




class ClassificationInput(BaseModel):
    recent_messages: list[Message]


class ClassificationOutput(BaseModel):
    chat_type: ChatType = Field(
        description="The best classification for the chat: family, work, home, or other."
    )

    chat_type_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence between 0.0 and 1.0."
    )

    reasoning: str = Field(
        description=(
            "Short user-facing explanation. Do not expose hidden chain-of-thought. "
            "Mention only the main evidence."
        )
    )
