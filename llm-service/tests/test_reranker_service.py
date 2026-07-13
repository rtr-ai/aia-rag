import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from services.reranker_service import (
    QwenCrossEncoderRerankerAdapter,
    RerankResult,
    RerankerRuntimeConfig,
    RerankerService,
)


class FakeCuda:
    def __init__(self, available=True, bf16_supported=True):
        self.available = available
        self.bf16_supported = bf16_supported

    def is_available(self):
        return self.available

    def is_bf16_supported(self):
        return self.bf16_supported


def fake_torch(available=True, bf16_supported=True):
    return types.SimpleNamespace(
        cuda=FakeCuda(available, bf16_supported),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )


def runtime_config(dtype="auto"):
    return RerankerRuntimeConfig(
        llama_embedding_path="llama-embedding",
        llama_tokenize_path="llama-tokenize",
        timeout_seconds=60,
        context_size=20000,
        ubatch_size=512,
        document_batch_size=4,
        gpu_layers=5,
        flash_attention=True,
        qwen_max_length=8192,
        qwen_dtype=dtype,
        qwen_attention="sdpa",
    )


class QwenCrossEncoderRerankerAdapterTest(unittest.TestCase):
    def test_model_is_reused_and_instruction_is_passed_per_call(self):
        constructed = []

        class FakeCrossEncoder:
            def __init__(self, model_path, **kwargs):
                self.model_path = model_path
                self.kwargs = kwargs
                self.predict_calls = []
                constructed.append(self)

            def predict(self, pairs, **kwargs):
                self.predict_calls.append((pairs, kwargs))
                return [0.1, 0.9]

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.CrossEncoder = FakeCrossEncoder
        torch = fake_torch()

        adapter = QwenCrossEncoderRerankerAdapter(
            "qwen-model", runtime_config()
        )
        with patch(
            "services.reranker_service.os.path.exists", return_value=True
        ), patch.dict(
            sys.modules,
            {"sentence_transformers": sentence_transformers, "torch": torch},
        ):
            adapter.rerank(
                "query", ["first", "second"], instruction="German instruction"
            )
            adapter.rerank(
                "query", ["first", "second"], instruction="English instruction"
            )

        self.assertEqual(len(constructed), 1)
        self.assertEqual(
            constructed[0].kwargs["model_kwargs"]["torch_dtype"], torch.bfloat16
        )
        self.assertEqual(
            constructed[0].kwargs["model_kwargs"]["attn_implementation"], "sdpa"
        )
        self.assertEqual(constructed[0].kwargs["max_length"], 8192)
        self.assertEqual(
            constructed[0].predict_calls[0][1]["prompt"], "German instruction"
        )
        self.assertEqual(
            constructed[0].predict_calls[1][1]["prompt"], "English instruction"
        )
        self.assertEqual(adapter.runtime_metadata["device"], "cuda:0")
        self.assertEqual(adapter.runtime_metadata["dtype"], "bfloat16")

    def test_auto_dtype_uses_float16_when_bfloat16_is_unavailable(self):
        adapter = QwenCrossEncoderRerankerAdapter("model", runtime_config())
        torch = fake_torch(bf16_supported=False)

        device, dtype, dtype_name = adapter._resolve_torch_runtime(torch)

        self.assertEqual(device, "cuda:0")
        self.assertIs(dtype, torch.float16)
        self.assertEqual(dtype_name, "float16")

    def test_cpu_uses_float32(self):
        adapter = QwenCrossEncoderRerankerAdapter("model", runtime_config())
        torch = fake_torch(available=False)

        device, dtype, dtype_name = adapter._resolve_torch_runtime(torch)

        self.assertEqual(device, "cpu")
        self.assertIs(dtype, torch.float32)
        self.assertEqual(dtype_name, "float32")


class RerankerFallbackTest(unittest.TestCase):
    def test_failed_qwen_advances_to_next_adapter(self):
        class FailingAdapter:
            backend_name = "qwen_cross_encoder"
            model_path = "qwen"

            def rerank(self, **kwargs):
                raise RuntimeError("Qwen failed")

        class WorkingAdapter:
            backend_name = "jina_gguf"
            model_path = "jina"
            runtime_metadata = {"device": "cuda:0"}

            def rerank(self, **kwargs):
                return [RerankResult(index=0, relevance_score=0.75)]

        service = object.__new__(RerankerService)
        service.enabled = True
        service.adapters = [FailingAdapter(), WorkingAdapter()]

        execution = service.rerank_with_metadata(
            "query", ["document"], instruction="instruction"
        )

        self.assertEqual(execution.backend_name, "jina_gguf")
        self.assertEqual(execution.model_path, "jina")
        self.assertEqual(execution.runtime, {"device": "cuda:0"})


if __name__ == "__main__":
    unittest.main()
