import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from models.chat_request import ChatRequest


class FakeEncoder:
    def encode(self, text):
        return text.split()


sys.modules.setdefault(
    "fastapi", types.SimpleNamespace(HTTPException=Exception)
)
sys.modules.setdefault(
    "ollama",
    types.SimpleNamespace(AsyncClient=object, EmbedResponse=object),
)
sys.modules.setdefault("pynvml", types.SimpleNamespace())
sys.modules.setdefault(
    "tiktoken",
    types.SimpleNamespace(get_encoding=lambda name: FakeEncoder()),
)

from services import chat_service as chat_service_module

ChatService = chat_service_module.ChatService


class FakeMeasurement:
    cpu_watts = 1.0
    gpu_watts = 2.0
    ram_watts = 3.0
    duration_seconds = 0.5
    total_watts = 6.0


class FakePowerMeter:
    def start(self):
        return None

    def stop(self):
        return FakeMeasurement()

    def get_initial_power_consumption(self):
        return {"total_kWh": 0.0}

    def sample_power(self):
        return FakeMeasurement()

    def get_median_power(self, measurements):
        return FakeMeasurement()


class FakeIndexService:
    def __init__(self):
        self.calls = []
        self.embedding_service = types.SimpleNamespace(
            model_for_dataset=lambda dataset_id: "dataset-embedding"
        )

    async def query_index(self, **kwargs):
        self.calls.append(kwargs)
        return [], 0.25, {
            "requested": kwargs["use_rerank"],
            "enabled": True,
            "applied": kwargs["use_rerank"],
        }


async def collect_events(service, request):
    events = []
    async for raw_event in service.chat(request, 1, object()):
        payload = raw_event.removeprefix("data: ").strip()
        events.append(json.loads(payload))
    return events


class ChatServiceRetrievalOnlyTest(unittest.IsolatedAsyncioTestCase):
    def build_service(self):
        service = object.__new__(ChatService)
        service.model = "test-llm"
        service.embedding_service = types.SimpleNamespace(model="test-embedding")
        service.index_service = FakeIndexService()
        service.prompt_ollama = AsyncMock()
        return service

    def test_generate_answer_defaults_to_true(self):
        request = ChatRequest(prompt="Question")

        self.assertTrue(request.generate_answer)

    def test_skip_retrieval_requires_final_prompt(self):
        with self.assertRaisesRegex(ValueError, "final_prompt is required"):
            ChatRequest(prompt="Question", skip_retrieval=True)

    def test_skip_retrieval_requires_answer_generation(self):
        with self.assertRaisesRegex(ValueError, "generate_answer must be true"):
            ChatRequest(
                prompt="Question",
                final_prompt="Complete prompt",
                skip_retrieval=True,
                generate_answer=False,
            )

    def test_final_prompt_requires_skip_retrieval(self):
        with self.assertRaisesRegex(ValueError, "skip_retrieval must be true"):
            ChatRequest(prompt="Question", final_prompt="Complete prompt")

    async def test_retrieval_only_emits_sources_and_skips_llm(self):
        service = self.build_service()
        request = ChatRequest(
            prompt="Question",
            use_rerank=True,
            generate_answer=False,
        )

        with patch(
            "services.chat_service.PowerMeterService", FakePowerMeter
        ), patch(
            "services.chat_service.generate_prompt"
        ) as generate_prompt, patch.object(
            sys.modules["services.chat_service"].matomo_service,
            "track_event",
        ):
            events = await collect_events(service, request)

        event_types = [event["type"] for event in events]
        self.assertEqual(
            event_types,
            [
                "heartbeat",
                "queue_position",
                "power_index",
                "metadata",
                "sources",
            ],
        )
        metadata = next(
            event["content"] for event in events if event["type"] == "metadata"
        )
        self.assertFalse(metadata["generate_answer"])
        self.assertFalse(metadata["llm_used"])
        self.assertTrue(metadata["rerank"]["requested"])
        self.assertEqual(metadata["embedding_model"], "dataset-embedding")
        self.assertTrue(service.index_service.calls[0]["use_rerank"])
        generate_prompt.assert_not_called()
        service.prompt_ollama.assert_not_called()

    async def test_default_mode_generates_answer(self):
        service = self.build_service()

        async def ollama_response(prompt):
            yield {"message": {"content": "Answer"}}

        service.prompt_ollama = ollama_response
        request = ChatRequest(prompt="Question")

        with patch(
            "services.chat_service.PowerMeterService", FakePowerMeter
        ), patch(
            "services.chat_service.generate_prompt", return_value="Prompt"
        ) as generate_prompt, patch.object(
            sys.modules["services.chat_service"].matomo_service,
            "track_event",
        ):
            events = await collect_events(service, request)

        metadata = next(
            event["content"] for event in events if event["type"] == "metadata"
        )
        self.assertTrue(metadata["generate_answer"])
        self.assertTrue(metadata["llm_used"])
        self.assertIn("assistant", [event["type"] for event in events])
        generate_prompt.assert_called_once()

    async def test_final_prompt_mode_skips_retrieval_and_prompt_construction(self):
        service = self.build_service()
        received_prompts = []

        async def ollama_response(prompt):
            received_prompts.append(prompt)
            yield {"message": {"content": "Approved-context answer"}}

        service.prompt_ollama = ollama_response
        request = ChatRequest(
            prompt="Question",
            final_prompt="Exact complete prompt",
            skip_retrieval=True,
        )

        with patch(
            "services.chat_service.PowerMeterService", FakePowerMeter
        ), patch(
            "services.chat_service.generate_prompt"
        ) as generate_prompt, patch.object(
            sys.modules["services.chat_service"].matomo_service,
            "track_event",
        ):
            events = await collect_events(service, request)

        event_types = [event["type"] for event in events]
        self.assertEqual(
            event_types,
            [
                "heartbeat",
                "queue_position",
                "metadata",
                "sources",
                "user",
                "assistant",
                "power_response",
            ],
        )
        metadata = next(
            event["content"] for event in events if event["type"] == "metadata"
        )
        self.assertTrue(metadata["retrieval_skipped"])
        self.assertIsNone(metadata["embedding_model"])
        self.assertEqual(metadata["rerank"]["reason"], "retrieval_skipped")
        self.assertEqual(service.index_service.calls, [])
        generate_prompt.assert_not_called()
        self.assertEqual(received_prompts, ["Exact complete prompt"])
        user_event = next(event for event in events if event["type"] == "user")
        self.assertEqual(user_event["content"], "Exact complete prompt")


if __name__ == "__main__":
    unittest.main()
