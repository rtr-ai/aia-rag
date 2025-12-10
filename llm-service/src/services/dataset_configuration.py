import json
import os
import sys
import time
from typing import Dict
from utils.logger import get_logger
from pathlib import Path

LOGGER = get_logger(__name__)


class ConfigError(Exception):
    pass


class DatasetConfiguration:
    def __init__(self):
        self.datasets = self._load_datasets()
        self.prompts = self._load_prompts()
        self._validate_consistency()

    def _load_datasets(self) -> Dict[str, str]:
        LOGGER.debug("Loading datasets")
        raw = os.getenv("DATASETS")
        if raw is None:
            LOGGER.error("DATASETS environment variable is not set.")
            raise ConfigError("DATASETS environment variable is not set.")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in DATASETS environment variable: {e}")

        if not isinstance(data, dict):
            LOGGER.error("DATASETS environment variable must be a JSON object.")
            raise ConfigError("DATASETS environment variable must be a JSON object.")

        for key, value in data.items():
            if not isinstance(key, str):

                raise ConfigError(
                    f"Invalid mapping key (not string): <{key}>. Must be the name of the dataset "
                )
            if not isinstance(value, str):
                raise ConfigError(
                    f"Invalid mapping value for <{key}> (not string). Must be name of the dataset file."
                )
            file_path = Path("/app/data") / value
            if not file_path.is_file():
                raise FileNotFoundError(f"File not found: <{file_path}>")

            LOGGER.info(f"Found dataset  <{key}> with chunks from path <{value}>")

        return data

    def _load_prompts(self) -> Dict[str, str]:

        file_path = Path("/app/data") / os.getenv(
            "SYSTEM_PROMPTS_FILE", "system_prompts"
        )
        if not file_path:
            raise ConfigError("SYSTEM_PROMPTS_FILE is not set.")

        if not os.path.exists(file_path):
            raise ConfigError(f"Prompts file does not exist: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in prompt file: {e}")

        if not isinstance(data, dict):
            raise ConfigError("Prompt file must be a JSON object.")

        for key, value in data.items():
            if not isinstance(key, str):
                raise ConfigError(f"Invalid prompt key (not string): {key}")
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"Invalid or empty prompt text for {key}.")

        return data

    def _validate_consistency(self):
        dataset_keys = set(self.datasets.keys())
        prompt_keys = set(self.prompts.keys())

        missing_prompts = dataset_keys - prompt_keys
        missing_mappings = prompt_keys - dataset_keys

        if missing_prompts:
            raise ConfigError(
                f"Missing system prompts for datasets: {', '.join(missing_prompts)}"
            )
        if missing_mappings:
            raise ConfigError(
                f"System prompt file contains keys not present in DATASETS: "
                f"{', '.join(missing_mappings)}"
            )

    def get_prompt(self, dataset: str) -> str:
        return self.prompts[dataset]

    def get_datasets(self, dataset: str) -> str:
        return self.datasets[dataset]

    def get_all_datasets(self) -> Dict[str, str]:
        """Return all dataset mappings (id -> filename)."""
        return self.datasets.copy()

    def dataset_exists(self, dataset: str) -> bool:
        """Return True if dataset exists, else False."""
        return dataset in self.datasets
