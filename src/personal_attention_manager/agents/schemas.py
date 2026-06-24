
class ChatCard:
    chat_id: str

    chat_type: str
    chat_type_confidence: float

    waiting_on: str
    open_loop: str | None

    situation_type: str
    delay_cost: str
    deadline: str | None

    priority_reason: str
    evidence: list[str]

    last_inbound_age_minutes: int
    suggested_action: str