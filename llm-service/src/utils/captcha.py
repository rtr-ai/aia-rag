import httpx
import os
from fastapi import Form, HTTPException, Depends
from typing import Optional
from models.chat_request import ChatRequest
from utils.logger import get_logger


CAPTCHA_OVERRIDE_SECRET = os.getenv("CAPTCHA_OVERRIDE_SECRET")
CAPTCHA_API_KEY = os.getenv("CAPTCHA_API_KEY")
CAPTCHA_SITEKEY = os.getenv("FRIENDLY_CAPTCHA_SITEKEY")
LOGGER = get_logger("Friendly Captcha")


async def verify_captcha(req: ChatRequest) -> bool:
    """
    Verify a Friendly CAPTCHA solution with an option to bypass for automated testing.

    Args:
        frc_captcha_solution: The solution value submitted by the user
        captcha_override: Secret value to bypass captcha verification (for testing)
        api_secret: Your Friendly CAPTCHA API key
        site_key: Optional sitekey to verify the puzzle was generated from

    Returns:
        bool: True if verification was successful or bypassed, False otherwise

    Raises:
        HTTPException: If there's an error with the verification request
    """

    # If no API Key is configured, disable captcha verification
    if not CAPTCHA_API_KEY:
        LOGGER.debug("No Friendly CAPTCHA API key provided, skipping verification")
        return True
    # Mechanism to allow API tests when sending pre-set CAPTCHA_OVERRIDE_SECRET
    if (
        req.frc_captcha_solution
        and CAPTCHA_OVERRIDE_SECRET
        and CAPTCHA_OVERRIDE_SECRET == req.frc_captcha_solution
    ):
        LOGGER.debug("CAPTCHA override secret, skipping verification")
        return True

    if not req.frc_captcha_solution:
        LOGGER.error("No CAPTCHA solution provided. Denying request")
        raise HTTPException(status_code=400, detail="CAPTCHA solution required")

    payload = {"solution": req.frc_captcha_solution, "secret": CAPTCHA_API_KEY}

    if CAPTCHA_SITEKEY:
        payload["sitekey"] = CAPTCHA_SITEKEY

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.friendlycaptcha.com/api/v1/siteverify", json=payload
            )

            result = response.json()

            if response.status_code == 200:
                return result.get("success", False)
            else:
                error_codes = result.get("errors", ["unknown_error"])
                LOGGER.error(
                    f"Error response from Friendly Captcha API: {', '.join(error_codes)}"
                )
                raise HTTPException(
                    status_code=400, detail="CAPTCHA verification failed"
                )

        except httpx.RequestError as e:
            LOGGER.error(f"Error connecting to Friendly CAPTCHA API: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Error connecting to CAPTCHA verification service",
            )
