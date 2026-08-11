import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))
sys.modules.setdefault(
    "ollama",
    types.SimpleNamespace(AsyncClient=object, EmbedResponse=dict),
)
sys.modules.setdefault("pynvml", types.SimpleNamespace())

from services.embedding_service import (
    EmbeddingService,
    configured_embedding_models,
    _with_prefix,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        return {"embeddings": [[1.0, 0.0]]}


class EmbeddingServiceConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def build_service(self):
        service = object.__new__(EmbeddingService)
        service.model = "snowflake-arctic-embed2"
        service.models_by_dataset = {
            "ai_act_de": "snowflake-arctic-embed2-ai-act"
        }
        service.query_prefixes = {
            "snowflake-arctic-embed2-ai-act": "query: "
        }
        service.passage_prefixes = {}
        service.client = FakeClient()
        return service

    def test_configured_models_include_dataset_specific_models_once(self):
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_MODELS": "snowflake-arctic-embed2",
                "EMBEDDING_MODELS_BY_DATASET": (
                    '{"ai_act_de":"snowflake-arctic-embed2-ai-act",'
                    '"ai_act_en":"snowflake-arctic-embed2"}'
                ),
            },
        ):
            models = configured_embedding_models()

        self.assertEqual(
            models,
            [
                "snowflake-arctic-embed2",
                "snowflake-arctic-embed2-ai-act",
            ],
        )

    async def test_dataset_model_and_query_prefix_are_applied_once(self):
        service = self.build_service()

        await service.generate_embedding("Frage", dataset_id="ai_act_de")
        await service.generate_embedding("query: Frage", dataset_id="ai_act_de")

        self.assertEqual(
            service.client.calls,
            [
                {
                    "model": "snowflake-arctic-embed2-ai-act",
                    "input": "query: Frage",
                },
                {
                    "model": "snowflake-arctic-embed2-ai-act",
                    "input": "query: Frage",
                },
            ],
        )

    def test_other_dataset_uses_default_model(self):
        service = self.build_service()

        self.assertEqual(
            service.model_for_dataset("ai_act_en"),
            "snowflake-arctic-embed2",
        )


    def test_prefix_is_applied_exactly_once_for_passages(self):
        self.assertEqual(_with_prefix("Text", "passage: "), "passage: Text")
        self.assertEqual(
            _with_prefix("passage: Text", "passage: "), "passage: Text"
        )

if __name__ == "__main__":
    unittest.main()

