import json
from utils.logger import get_logger
import os
from typing import Optional, Dict, Any
from matomo_api import MatomoApi

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
        matomo_url = os.getenv("MATOMO_ENDPOINT")
        matomo_token = os.getenv("MATOMO_TOKEN")
        matomo_site_id = os.getenv("MATOMO_SITE_ID")

        if matomo_url and matomo_token and matomo_site_id:
            try:
                instance.client = MatomoApi(
                    url=matomo_url, token_auth=matomo_token, id_site=matomo_site_id
                )
                instance.enabled = True
                LOGGER.debug(
                    f"Matomo tracking initialized for site ID: {matomo_site_id}"
                )
            except Exception as e:
                LOGGER.error(f"Failed to initialize Matomo tracking: {e}")
                instance.enabled = False
        else:
            LOGGER.debug("Matomo tracking not configured. Skipping initialization.")
            instance.enabled = False
            instance.client = None

    def track_event(
        self,
        action: str,
        category: str = "llm_service",
        name: Optional[str] = None,
        value: Optional[Dict[str, Any] | str] = None,
    ):
        """
        Track an event to Matomo with optional JSON or string data

        :param category: Event category, defaults to "llm_service"
        :param action: Event action
        :param name: Optional event name
        :param value: Optional String or  Dict data  send. If Dict, data serialized to JSON data and included with the event
        """
        if not self.enabled:
            LOGGER.debug("Matomo tracking is disabled. Skipping event.")
            return

        try:
            params = {
                "e_c": category,
                "e_a": action,
            }

            if name:
                params["e_n"] = name

            if value is not None:
                if isinstance(value, str):
                    params["e_v"] = value
                else:
                    params["e_v"] = json.dumps(value)

            self.client.track(params)
            LOGGER.debug(f"Tracked Matomo event: {category} - {action}")
        except Exception as e:
            LOGGER.error(f"Failed to track Matomo event: {e}")


matomo_service = MatomoTrackingService()
