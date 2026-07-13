import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from services.reranker_service import (
    BgeCrossEncoderRerankerAdapter,
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
        inference_mode=lambda: nullcontext(),
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
        bge_max_length=8192,
        bge_dtype=dtype,
        bge_attention="sdpa",
        bge_batch_size=2,
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


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def to(self, device):
        return self

    def view(self, *shape):
        return self

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBgeTokenizer:
    lengths = {"long": 5, "short": 1, "medium": 3}
    score_tokens = {"long": 2, "short": 9, "medium": 5}

    def __init__(self):
        self.call_kwargs = []
        self.padded_batches = []

    def __call__(self, queries, documents, **kwargs):
        self.call_kwargs.append(kwargs)
        input_ids = []
        for document in documents:
            length = self.lengths[document]
            input_ids.append(
                [self.score_tokens[document]] + [0] * (length - 1)
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }

    def pad(self, items, **kwargs):
        score_tokens = [item["input_ids"][0] for item in items]
        self.padded_batches.append(score_tokens)
        return {
            "input_ids": FakeTensor(score_tokens),
            "attention_mask": FakeTensor([1] * len(items)),
        }


class FakeBgeModel:
    def __init__(self, oom_error=None):
        self.oom_error = oom_error
        self.batch_sizes = []
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return iter(())

    def __call__(self, input_ids, **kwargs):
        self.batch_sizes.append(len(input_ids.values))
        if self.oom_error and len(input_ids.values) > 1:
            raise self.oom_error("CUDA out of memory")
        return types.SimpleNamespace(
            logits=FakeTensor([value / 10 for value in input_ids.values])
        )


class BgeCrossEncoderRerankerAdapterTest(unittest.TestCase):
    def test_model_is_loaded_once_with_optimized_runtime(self):
        tokenizer = FakeBgeTokenizer()
        model = FakeBgeModel()
        tokenizer_loads = []
        model_loads = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_path):
                tokenizer_loads.append(model_path)
                return tokenizer

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_path, **kwargs):
                model_loads.append((model_path, kwargs))
                return model

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = FakeAutoTokenizer
        transformers.AutoModelForSequenceClassification = FakeAutoModel
        torch = fake_torch()
        adapter = BgeCrossEncoderRerankerAdapter(
            "bge-model", runtime_config()
        )

        with patch(
            "services.reranker_service.os.path.exists", return_value=True
        ), patch.dict(
            sys.modules, {"transformers": transformers, "torch": torch}
        ):
            adapter._ensure_ready()
            adapter._ensure_ready()

        self.assertEqual(tokenizer_loads, ["bge-model"])
        self.assertEqual(len(model_loads), 1)
        self.assertIs(model_loads[0][1]["torch_dtype"], torch.bfloat16)
        self.assertEqual(
            model_loads[0][1]["attn_implementation"], "sdpa"
        )
        self.assertEqual(adapter.runtime_metadata["device"], "cuda:0")
        self.assertEqual(adapter.runtime_metadata["dtype"], "bfloat16")
        self.assertEqual(adapter.runtime_metadata["max_length"], 8192)
        self.assertEqual(
            adapter.runtime_metadata["configured_batch_size"], 2
        )

    def test_auto_dtype_falls_back_to_float16_and_cpu_float32(self):
        adapter = BgeCrossEncoderRerankerAdapter(
            "bge-model", runtime_config()
        )
        gpu_torch = fake_torch(bf16_supported=False)
        cpu_torch = fake_torch(available=False)

        gpu_device, gpu_dtype, gpu_name = adapter._resolve_torch_runtime(
            gpu_torch
        )
        cpu_device, cpu_dtype, cpu_name = adapter._resolve_torch_runtime(
            cpu_torch
        )

        self.assertEqual((gpu_device, gpu_name), ("cuda:0", "float16"))
        self.assertIs(gpu_dtype, gpu_torch.float16)
        self.assertEqual((cpu_device, cpu_name), ("cpu", "float32"))
        self.assertIs(cpu_dtype, cpu_torch.float32)

    def test_length_sorting_restores_original_document_indexes(self):
        adapter = self._ready_adapter()
        results = adapter.rerank(
            "query", ["long", "short", "medium"]
        )

        self.assertEqual([result.index for result in results], [1, 2, 0])
        self.assertEqual(
            adapter.tokenizer.padded_batches, [[2, 5], [9]]
        )
        self.assertEqual(
            adapter.tokenizer.call_kwargs[0]["truncation"], "only_second"
        )
        self.assertEqual(
            adapter.tokenizer.call_kwargs[0]["max_length"], 8192
        )

    def test_cuda_oom_retries_all_documents_with_batch_size_one(self):
        class FakeCudaOom(RuntimeError):
            pass

        empty_cache_calls = []
        torch = fake_torch()
        torch.cuda.OutOfMemoryError = FakeCudaOom
        torch.cuda.empty_cache = lambda: empty_cache_calls.append(True)
        adapter = self._ready_adapter(
            torch=torch, model=FakeBgeModel(FakeCudaOom)
        )

        adapter.rerank("query", ["long", "short", "medium"])

        self.assertEqual(adapter.model.batch_sizes, [2, 1, 1, 1])
        self.assertEqual(empty_cache_calls, [True])
        self.assertEqual(
            adapter.runtime_metadata["effective_batch_size"], 1
        )

    def _ready_adapter(self, torch=None, model=None):
        adapter = BgeCrossEncoderRerankerAdapter(
            "bge-model", runtime_config()
        )
        adapter.tokenizer = FakeBgeTokenizer()
        adapter.model = model or FakeBgeModel()
        adapter.torch = torch or fake_torch()
        adapter.device = "cuda:0"
        adapter.runtime_metadata = {
            "backend": "bge_cross_encoder",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "max_length": 8192,
            "configured_batch_size": 2,
            "effective_batch_size": 2,
        }
        return adapter


class RerankerFallbackTest(unittest.TestCase):
    def test_failed_bge_advances_to_qwen(self):
        class FailingAdapter:
            backend_name = "bge_cross_encoder"
            model_path = "bge"

            def rerank(self, **kwargs):
                raise RuntimeError("BGE failed")

        class WorkingAdapter:
            backend_name = "qwen_cross_encoder"
            model_path = "qwen"
            runtime_metadata = {"device": "cuda:0"}

            def rerank(self, **kwargs):
                return [RerankResult(index=0, relevance_score=0.8)]

        service = object.__new__(RerankerService)
        service.enabled = True
        service.adapters = [FailingAdapter(), WorkingAdapter()]

        execution = service.rerank_with_metadata(
            "query", ["document"], instruction="instruction"
        )

        self.assertEqual(execution.backend_name, "qwen_cross_encoder")
        self.assertEqual(execution.model_path, "qwen")

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
