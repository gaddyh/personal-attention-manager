import logging

from fastapi import HTTPException, Request

from personal_attention_manager.config import settings

logger = logging.getLogger("waiting-for-you")


def verify_green_api_authorization(request: Request) -> None:
    authorization = request.headers.get("authorization")

    if authorization != settings.expected_authorization_header:
        logger.warning("Rejected webhook with invalid Authorization header")
        raise HTTPException(status_code=403, detail="Invalid webhook authorization")