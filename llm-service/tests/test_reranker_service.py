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
    MixedbreadCrossEncoderRerankerAdapter,
    NvidiaCrossEncoderRerankerAdapter,
    QwenCrossEncoderRerankerAdapter,
    RerankResult,
    RerankerRuntimeConfig,
    RerankerService,
    _torch_runtime_details,
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
        __version__="2.11.0+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
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
        mixedbread_max_length=8192,
        mixedbread_dtype=dtype,
        mixedbread_attention="sdpa",
        mixedbread_batch_size=2,
        nvidia_max_length=8192,
        nvidia_dtype=dtype,
        nvidia_attention="sdpa",
        nvidia_batch_size=1,
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
        self.assertEqual(
            adapter.runtime_metadata["configured_batch_size"], 4
        )
        self.assertEqual(adapter.runtime_metadata["effective_batch_size"], 4)

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


class FakeMixedbreadCrossEncoder:
    def __init__(self, oom_error=None):
        self.tokenizer = FakeBgeTokenizer()
        self.model = types.SimpleNamespace(parameters=lambda: iter(()))
        self.oom_error = oom_error
        self.predict_calls = []

    def predict(self, pairs, **kwargs):
        self.predict_calls.append((pairs, kwargs))
        if self.oom_error and len(pairs) > 1:
            raise self.oom_error("CUDA out of memory")
        return [
            FakeBgeTokenizer.score_tokens[document] / 10
            for _, document in pairs
        ]


class MixedbreadCrossEncoderRerankerAdapterTest(unittest.TestCase):
    def test_model_is_reused_and_instruction_is_passed_per_call(self):
        constructed = []

        class FakeCrossEncoder(FakeMixedbreadCrossEncoder):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                self.model_path = model_path
                self.kwargs = kwargs
                constructed.append(self)

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.CrossEncoder = FakeCrossEncoder
        torch = fake_torch()
        adapter = MixedbreadCrossEncoderRerankerAdapter(
            "mixedbread-model", runtime_config()
        )

        with patch(
            "services.reranker_service.os.path.exists", return_value=True
        ), patch.dict(
            sys.modules,
            {"sentence_transformers": sentence_transformers, "torch": torch},
        ):
            adapter.rerank(
                "query", ["long", "short"], instruction="German instruction"
            )
            adapter.rerank(
                "query", ["long", "short"], instruction="English instruction"
            )

        self.assertEqual(len(constructed), 1)
        self.assertIs(
            constructed[0].kwargs["model_kwargs"]["torch_dtype"], torch.bfloat16
        )
        self.assertEqual(
            constructed[0].kwargs["model_kwargs"]["attn_implementation"],
            "sdpa",
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
        self.assertTrue(adapter.runtime_metadata["instruction_used"])

    def test_auto_dtype_falls_back_to_float16_and_cpu_float32(self):
        adapter = MixedbreadCrossEncoderRerankerAdapter(
            "mixedbread-model", runtime_config()
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
            "query",
            ["long", "short", "medium"],
            instruction="instruction",
        )

        self.assertEqual([result.index for result in results], [1, 2, 0])
        scored_documents = [
            document
            for pairs, _ in adapter.model.predict_calls
            for _, document in pairs
        ]
        self.assertEqual(scored_documents, ["long", "medium", "short"])
        self.assertEqual(
            adapter.model.tokenizer.call_kwargs[0]["max_length"], 8192
        )
        self.assertEqual(adapter.runtime_metadata["document_count"], 3)

    def test_cuda_oom_retries_all_documents_with_batch_size_one(self):
        class FakeCudaOom(RuntimeError):
            pass

        empty_cache_calls = []
        torch = fake_torch()
        torch.cuda.OutOfMemoryError = FakeCudaOom
        torch.cuda.empty_cache = lambda: empty_cache_calls.append(True)
        model = FakeMixedbreadCrossEncoder(FakeCudaOom)
        adapter = self._ready_adapter(torch=torch, model=model)

        adapter.rerank("query", ["long", "short", "medium"])

        self.assertEqual(
            [len(pairs) for pairs, _ in model.predict_calls],
            [2, 1, 1, 1],
        )
        self.assertEqual(empty_cache_calls, [True])
        self.assertEqual(
            adapter.runtime_metadata["effective_batch_size"], 1
        )

    def _ready_adapter(self, torch=None, model=None):
        adapter = MixedbreadCrossEncoderRerankerAdapter(
            "mixedbread-model", runtime_config()
        )
        adapter.model = model or FakeMixedbreadCrossEncoder()
        adapter.torch = torch or fake_torch()
        adapter.device = "cuda:0"
        adapter.runtime_metadata = {
            "backend": "mixedbread_cross_encoder",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "max_length": 8192,
            "configured_batch_size": 2,
            "effective_batch_size": 2,
            "document_count": 0,
            "instruction_used": False,
        }
        return adapter


class FakeNvidiaTokenizer:
    lengths = {"long": 5, "short": 1, "medium": 3}
    score_tokens = {"long": 2, "short": 9, "medium": 5}

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"
        self.eos_token_id = 17
        self.call_kwargs = []
        self.texts = []
        self.padded_batches = []

    def __call__(self, texts, **kwargs):
        self.call_kwargs.append(kwargs)
        self.texts.extend(texts)
        input_ids = []
        for text in texts:
            document = text.split("passage:", 1)[1]
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
        self.padded_batches.append((score_tokens, kwargs))
        return {
            "input_ids": FakeTensor(score_tokens),
            "attention_mask": FakeTensor([1] * len(items)),
        }


class FakeNvidiaModel:
    def __init__(self, oom_error=None):
        self.config = types.SimpleNamespace(
            pad_token_id=None,
            use_cache=True,
        )
        self.oom_error = oom_error
        self.batch_sizes = []
        self.call_kwargs = []
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
        self.call_kwargs.append(kwargs)
        if self.oom_error and len(input_ids.values) > 1:
            raise self.oom_error("CUDA out of memory")
        return types.SimpleNamespace(
            logits=FakeTensor([value / 10 for value in input_ids.values])
        )


class NvidiaCrossEncoderRerankerAdapterTest(unittest.TestCase):
    def test_model_is_loaded_once_with_optimized_runtime(self):
        tokenizer = FakeNvidiaTokenizer()
        model = FakeNvidiaModel()
        tokenizer_loads = []
        model_loads = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, **kwargs):
                tokenizer_loads.append((model_path, kwargs))
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
        adapter = NvidiaCrossEncoderRerankerAdapter(
            "nvidia-model", runtime_config()
        )

        with patch(
            "services.reranker_service.os.path.exists", return_value=True
        ), patch.dict(
            sys.modules, {"transformers": transformers, "torch": torch}
        ):
            adapter._ensure_ready()
            adapter._ensure_ready()

        self.assertEqual(len(tokenizer_loads), 1)
        self.assertEqual(tokenizer_loads[0][1]["padding_side"], "left")
        self.assertTrue(tokenizer_loads[0][1]["trust_remote_code"])
        self.assertEqual(len(model_loads), 1)
        self.assertTrue(model_loads[0][1]["trust_remote_code"])
        self.assertIs(model_loads[0][1]["torch_dtype"], torch.bfloat16)
        self.assertEqual(model_loads[0][1]["attn_implementation"], "sdpa")
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(model.config.pad_token_id, tokenizer.eos_token_id)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(adapter.runtime_metadata["device"], "cuda:0")
        self.assertEqual(
            adapter.runtime_metadata["torch_version"], "2.11.0+cu128"
        )
        self.assertEqual(
            adapter.runtime_metadata["torch_cuda_version"], "12.8"
        )
        self.assertTrue(adapter.runtime_metadata["cuda_available"])
        self.assertEqual(adapter.runtime_metadata["dtype"], "bfloat16")
        self.assertEqual(adapter.runtime_metadata["max_length"], 8192)
        self.assertFalse(adapter.runtime_metadata["use_cache"])

    def test_cpu_fallback_reports_runtime_and_logs_warning(self):
        torch = fake_torch(available=False)

        with patch("services.reranker_service.LOGGER.warning") as warning:
            details = _torch_runtime_details(
                torch, "cpu", "nvidia_cross_encoder"
            )

        self.assertEqual(details["torch_version"], "2.11.0+cu128")
        self.assertEqual(details["torch_cuda_version"], "12.8")
        self.assertFalse(details["cuda_available"])
        warning.assert_called_once()
        self.assertIn("using CPU", warning.call_args.args[0])

    def test_template_sorting_scores_and_cache_setting(self):
        adapter = self._ready_adapter()
        results = adapter.rerank(
            "query",
            ["long", "short", "medium"],
            instruction="must not be added",
        )

        self.assertEqual([result.index for result in results], [1, 2, 0])
        self.assertEqual(
            adapter.tokenizer.texts,
            [
                "question:query \n \n passage:long",
                "question:query \n \n passage:short",
                "question:query \n \n passage:medium",
            ],
        )
        self.assertNotIn("must not be added", "".join(adapter.tokenizer.texts))
        self.assertEqual(
            adapter.tokenizer.call_kwargs[0]["max_length"], 8192
        )
        self.assertEqual(
            [batch[0] for batch in adapter.tokenizer.padded_batches],
            [[2], [5], [9]],
        )
        self.assertTrue(
            all(
                batch[1]["pad_to_multiple_of"] == 8
                for batch in adapter.tokenizer.padded_batches
            )
        )
        self.assertTrue(
            all(call["use_cache"] is False for call in adapter.model.call_kwargs)
        )
        self.assertEqual(adapter.runtime_metadata["document_count"], 3)

    def test_configured_batch_two_retries_with_batch_one_after_cuda_oom(self):
        class FakeCudaOom(RuntimeError):
            pass

        empty_cache_calls = []
        torch = fake_torch()
        torch.cuda.OutOfMemoryError = FakeCudaOom
        torch.cuda.empty_cache = lambda: empty_cache_calls.append(True)
        config = runtime_config()
        config.nvidia_batch_size = 2
        model = FakeNvidiaModel(FakeCudaOom)
        adapter = self._ready_adapter(config=config, torch=torch, model=model)

        adapter.rerank("query", ["long", "short", "medium"])

        self.assertEqual(model.batch_sizes, [2, 1, 1, 1])
        self.assertEqual(empty_cache_calls, [True])
        self.assertEqual(
            adapter.runtime_metadata["effective_batch_size"], 1
        )

    def test_auto_dtype_falls_back_to_float16_and_cpu_float32(self):
        adapter = NvidiaCrossEncoderRerankerAdapter(
            "nvidia-model", runtime_config()
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

    def test_service_does_not_forward_dataset_instruction(self):
        received_instructions = []

        class RecordingAdapter:
            backend_name = "nvidia_cross_encoder"
            model_path = "nvidia"
            runtime_metadata = {}

            def rerank(self, **kwargs):
                received_instructions.append(kwargs["instruction"])
                return [RerankResult(index=0, relevance_score=0.9)]

        service = object.__new__(RerankerService)
        service.enabled = True
        service.adapters = [RecordingAdapter()]

        service.rerank_with_metadata(
            "query", ["document"], instruction="dataset instruction"
        )

        self.assertEqual(received_instructions, [None])


    def _ready_adapter(self, config=None, torch=None, model=None):
        adapter = NvidiaCrossEncoderRerankerAdapter(
            "nvidia-model", config or runtime_config()
        )
        adapter.tokenizer = FakeNvidiaTokenizer()
        adapter.model = model or FakeNvidiaModel()
        adapter.torch = torch or fake_torch()
        adapter.device = "cuda:0"
        adapter.runtime_metadata = {
            "backend": "nvidia_cross_encoder",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "max_length": 8192,
            "configured_batch_size": adapter.config.nvidia_batch_size,
            "effective_batch_size": adapter.config.nvidia_batch_size,
            "use_cache": False,
            "document_count": 0,
        }
        return adapter


class RerankerFallbackTest(unittest.TestCase):
    def test_failed_nvidia_advances_to_mixedbread(self):
        class FailingAdapter:
            backend_name = "nvidia_cross_encoder"
            model_path = "nvidia"

            def rerank(self, **kwargs):
                raise RuntimeError("NVIDIA failed")

        class WorkingAdapter:
            backend_name = "mixedbread_cross_encoder"
            model_path = "mixedbread"
            runtime_metadata = {"device": "cuda:0"}

            def rerank(self, **kwargs):
                return [RerankResult(index=0, relevance_score=0.9)]

        service = object.__new__(RerankerService)
        service.enabled = True
        service.adapters = [FailingAdapter(), WorkingAdapter()]

        execution = service.rerank_with_metadata(
            "query", ["document"], instruction="instruction"
        )

        self.assertEqual(execution.backend_name, "mixedbread_cross_encoder")
        self.assertEqual(execution.model_path, "mixedbread")
        self.assertEqual(execution.runtime, {"device": "cuda:0"})


    def test_failed_mixedbread_advances_to_bge(self):
        class FailingAdapter:
            backend_name = "mixedbread_cross_encoder"
            model_path = "mixedbread"

            def rerank(self, **kwargs):
                raise RuntimeError("Mixedbread failed")

        class WorkingAdapter:
            backend_name = "bge_cross_encoder"
            model_path = "bge"
            runtime_metadata = {"device": "cuda:0"}

            def rerank(self, **kwargs):
                return [RerankResult(index=0, relevance_score=0.85)]

        service = object.__new__(RerankerService)
        service.enabled = True
        service.adapters = [FailingAdapter(), WorkingAdapter()]

        execution = service.rerank_with_metadata(
            "query", ["document"], instruction="instruction"
        )

        self.assertEqual(execution.backend_name, "bge_cross_encoder")
        self.assertEqual(execution.model_path, "bge")
        self.assertEqual(execution.runtime, {"device": "cuda:0"})


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
