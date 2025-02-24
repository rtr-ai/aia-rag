import json
import os
from typing import Optional, Dict, Any
import requests
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class MatomoTrackingService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(MatomoTrackingService, cls).__new__(cls)
            cls._initialize(cls._instance)
        return cls._instance

    @classmethod
    def _initialize(cls, instance):
        instance.matomo_url = os.getenv("MATOMO_ENDPOINT")
        instance.matomo_token = os.getenv("MATOMO_TOKEN")
        instance.matomo_site_id = os.getenv("MATOMO_SITE_ID")

        if instance.matomo_url and instance.matomo_token and instance.matomo_site_id:
            instance.enabled = True
            LOGGER.debug(
                f"Matomo tracking initialized for site ID: {instance.matomo_site_id}"
            )
        else:
            LOGGER.debug("Matomo tracking not configured. Skipping initialization.")
            instance.enabled = False

    def track_event(
        self,
        action: str,
        category: str = "llm_service",
        value: Optional[Dict[str, Any] | str] = None,
    ):
        """
        Track an event to Matomo with optional JSON or string data

        :param category: Event category, defaults to "llm_service"
        :param action: Event action
        :param value: Optional String or Dict data. If Dict, it's serialized to JSON.
        """
        if not self.enabled:
            LOGGER.debug("Matomo tracking is disabled. Skipping event.")
            return

        try:
            params = {
                "idsite": self.matomo_site_id,
                "rec": 1,
                "e_c": category,
                "e_a": action,
                "token_auth": self.matomo_token,
            }

            if value is not None:
                if not isinstance(value, str):
                    value = json.dumps(value)
                params["e_n"] = value
                params["e_v"] = len(value)

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(self.matomo_url, data=params, headers=headers)

            if response.status_code == 200:
                LOGGER.debug(f"Tracked Matomo event: {category} - {action}")
            else:
                LOGGER.error(
                    f"Failed to track Matomo event: HTTP {response.status_code}"
                )

        except Exception as e:
            LOGGER.error(f"Failed to track Matomo event: {e}")


matomo_service = MatomoTrackingService()
