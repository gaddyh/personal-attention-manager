import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("greenapi-bot")


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class Settings:
    green_api_base_url: str
    green_api_id_instance: str
    green_api_token_instance: str

    openai_transcribe_model: str

    webhook_secret: str
    expected_authorization_header: str

    allowed_chat_ids: set[str]

    dspy_model: str

    timezone: str


def load_settings() -> Settings:
    green_api_base_url = os.getenv(
        "GREEN_API_BASE_URL",
        "https://api.green-api.com",
    ).rstrip("/")

    green_api_id_instance = os.getenv("GREEN_API_ID_INSTANCE", "")
    green_api_token_instance = os.getenv("GREEN_API_TOKEN_INSTANCE", "")

    openai_transcribe_model = os.getenv(
        "OPENAI_TRANSCRIBE_MODEL",
        "gpt-4o-transcribe",
    )

    webhook_secret = os.getenv("WEBHOOK_SECRET", "")

    allowed_chat_ids = parse_csv_set(os.getenv("ALLOWED_CHAT_IDS", ""))

    if not green_api_id_instance:
        raise RuntimeError("Missing GREEN_API_ID_INSTANCE")

    if not green_api_token_instance:
        raise RuntimeError("Missing GREEN_API_TOKEN_INSTANCE")

    if not webhook_secret:
        raise RuntimeError("Missing WEBHOOK_SECRET")

    if not allowed_chat_ids:
        logger.warning("ALLOWED_CHAT_IDS is empty. Bot will ignore all incoming messages.")

    return Settings(
        green_api_base_url=green_api_base_url,
        green_api_id_instance=green_api_id_instance,
        green_api_token_instance=green_api_token_instance,
        openai_transcribe_model=openai_transcribe_model,
        webhook_secret=webhook_secret,
        expected_authorization_header=f"Bearer {webhook_secret}",
        allowed_chat_ids=allowed_chat_ids,
        dspy_model=os.getenv("DSPY_MODEL", "openai/gpt-5.4-mini"),
        timezone=os.getenv("TIMEZONE", "Asia/Jerusalem"),
    )


settings = load_settings()
