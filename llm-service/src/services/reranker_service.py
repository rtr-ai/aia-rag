import gc
import json
import os
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import TypeAdapter

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _torch_runtime_details(
    torch, device: str, backend: str
) -> Dict[str, object]:
    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda_version = getattr(
        getattr(torch, "version", None), "cuda", None
    )
    cuda_available = str(device).startswith("cuda")
    details: Dict[str, object] = {
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "cuda_available": cuda_available,
    }
    if not cuda_available:
        LOGGER.warning(
            "CUDA is unavailable for %s; using CPU. PyTorch version: %s, "
            "compiled CUDA version: %s"
            % (backend, torch_version, torch_cuda_version or "none")
        )
    return details


def _env_bool(name: str, default: str = "false") -> bool:
    return TypeAdapter(bool).validate_python(os.getenv(name, default))


def _env_list(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _cuda_memory_snapshot() -> Optional[Dict[str, float]]:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        memory_info = getattr(torch.cuda, "mem_get_info", None)
        if memory_info is None:
            return None
        free_bytes, total_bytes = memory_info()
        divisor = 1024 * 1024
        return {
            "free_mib": round(free_bytes / divisor, 2),
            "total_mib": round(total_bytes / divisor, 2),
        }
    except Exception:
        return None


@dataclass
class RerankResult:
    index: int
    relevance_score: float


@dataclass
class RerankExecution:
    results: List[RerankResult]
    backend_name: str
    model_path: str
    runtime: Dict[str, object] = field(default_factory=dict)


@dataclass
class RerankerRuntimeConfig:
    llama_embedding_path: str
    llama_tokenize_path: str
    timeout_seconds: int
    context_size: int
    ubatch_size: int
    document_batch_size: int
    gpu_layers: int
    flash_attention: bool
    jina_context_size: int
    jina_max_query_length: int
    jina_max_doc_length: int
    jina_dtype: str
    jina_attention: str
    jina_require_cuda: bool
    qwen_max_length: int
    qwen_dtype: str
    qwen_attention: str
    bge_max_length: int
    bge_dtype: str
    bge_attention: str
    bge_batch_size: int
    mixedbread_max_length: int
    mixedbread_dtype: str
    mixedbread_attention: str
    mixedbread_batch_size: int
    nvidia_max_length: int
    nvidia_dtype: str
    nvidia_attention: str
    nvidia_batch_size: int


class MLPProjector:
    def __init__(self, linear1_weight, linear2_weight):
        self.linear1_weight = linear1_weight
        self.linear2_weight = linear2_weight

    def __call__(self, value):
        value = value @ self.linear1_weight.T
        value = np.maximum(0, value)
        return value @ self.linear2_weight.T


class JinaGgufRerankerAdapter:
    backend_name = "jina_gguf"

    def __init__(
        self,
        model_path: str,
        projector_path: str,
        config: RerankerRuntimeConfig,
    ):
        self.model_path = model_path
        self.projector_path = projector_path
        self.config = config
        self.projector: Optional[MLPProjector] = None
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.query_embed_token_id = 151671
        self.doc_embed_token_id = 151670

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        results = []
        for batch_start in range(0, len(documents), self.config.document_batch_size):
            batch_documents = documents[
                batch_start : batch_start + self.config.document_batch_size
            ]
            batch_scores = self._score_batch(query, batch_documents, instruction)
            results.extend(
                RerankResult(
                    index=batch_start + batch_index,
                    relevance_score=float(score),
                )
                for batch_index, score in enumerate(batch_scores)
            )

        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.debug(
            f"Reranked {len(documents)} candidate chunks to {len(results)} chunks "
            f"in {time.perf_counter() - start:.2f}s using backend "
            f"<{self.backend_name}> model <{self.model_path}> and document batch "
            f"size {self.config.document_batch_size}"
        )
        return results

    def _score_batch(
        self, query: str, documents: List[str], instruction: Optional[str]
    ) -> np.ndarray:
        prompt = self._format_prompt(query, documents, instruction)
        embeddings = self._get_hidden_states(prompt)
        tokens = self._tokenize(prompt)
        scores = self._score_embeddings(embeddings, tokens)
        if len(scores) != len(documents):
            raise ValueError(
                f"Reranker produced {len(scores)} scores for "
                f"{len(documents)} documents"
            )
        return scores

    def _ensure_ready(self):
        required_paths = [
            self.model_path,
            self.projector_path,
            self.config.llama_embedding_path,
            self.config.llama_tokenize_path,
        ]
        missing_paths = [item for item in required_paths if not os.path.exists(item)]
        if missing_paths:
            raise FileNotFoundError(
                "Jina GGUF reranker required files are missing: "
                + ", ".join(missing_paths)
            )
        if self.projector is None:
            self.projector = self._load_projector()

    def _load_projector(self) -> MLPProjector:
        from safetensors import safe_open

        with safe_open(self.projector_path, framework="numpy") as tensor_file:
            linear1_weight = tensor_file.get_tensor("projector.0.weight")
            linear2_weight = tensor_file.get_tensor("projector.2.weight")
        return MLPProjector(linear1_weight, linear2_weight)

    def _format_prompt(
        self, query: str, documents: List[str], instruction: Optional[str]
    ) -> str:
        sanitized_query = self._sanitize(query)
        sanitized_documents = [self._sanitize(document) for document in documents]
        prompt = (
            "<|im_start|>system\n"
            "You are a search relevance expert. Rank passages by how well they "
            "answer or satisfy the query."
            "<|im_end|>\n<|im_start|>user\n"
            f"I will provide you with {len(sanitized_documents)} passages, each "
            f"indicated by a numerical identifier. Rank the passages based on "
            f"their relevance to query: {sanitized_query}\n"
        )
        if instruction:
            prompt += f"<instruct>\n{self._sanitize(instruction)}\n</instruct>\n"

        doc_embed_token = self.special_tokens["doc_embed_token"]
        query_embed_token = self.special_tokens["query_embed_token"]
        prompt += "\n".join(
            f'<passage id="{index}">\n{document}{doc_embed_token}\n</passage>'
            for index, document in enumerate(sanitized_documents)
        )
        prompt += f"\n<query>\n{sanitized_query}{query_embed_token}\n</query>"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _sanitize(self, text: str) -> str:
        sanitized = text or ""
        for token in self.special_tokens.values():
            sanitized = sanitized.replace(token, "")
        return sanitized

    def _get_hidden_states(self, prompt: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        command = [
            self.config.llama_embedding_path,
            "-m",
            self.model_path,
            "-f",
            prompt_file_path,
            "--pooling",
            "none",
            "--embd-separator",
            "<#JINA_SEP#>",
            "--embd-normalize",
            "-1",
            "--embd-output-format",
            "json",
            "--ubatch-size",
            str(self.config.ubatch_size),
            "--ctx-size",
            str(self.config.context_size),
            "-ngl",
            str(self.config.gpu_layers),
        ]
        if self.config.flash_attention:
            command.append("--flash-attn")

        try:
            result = self._run_command(command)
            output = json.loads(result.stdout)
            return np.array([item["embedding"] for item in output["data"]])
        finally:
            os.unlink(prompt_file_path)

    def _tokenize(self, prompt: str) -> List[int]:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        try:
            result = self._run_command(
                [
                    self.config.llama_tokenize_path,
                    "-m",
                    self.model_path,
                    "-f",
                    prompt_file_path,
                ]
            )
            tokens = []
            for line in result.stdout.strip().splitlines():
                if "->" in line:
                    tokens.append(int(line.split("->", 1)[0].strip()))
            return tokens
        finally:
            os.unlink(prompt_file_path)

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.CalledProcessError as error:
            stderr = self._tail_output(error.stderr)
            stdout = self._tail_output(error.stdout)
            detail = stderr or stdout or "no output"
            raise RuntimeError(
                f"Reranker command failed with exit code {error.returncode}: {detail}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"Reranker command timed out after {self.config.timeout_seconds}s: "
                + " ".join(command)
            ) from error

    def _tail_output(self, output: Optional[str], max_chars: int = 4000) -> str:
        if not output:
            return ""
        output = output.strip()
        if len(output) <= max_chars:
            return output
        return "... " + output[-max_chars:]

    def _score_embeddings(self, embeddings: np.ndarray, tokens: List[int]) -> np.ndarray:
        if self.projector is None:
            raise RuntimeError("Reranker projector is not loaded")

        tokens_array = np.array(tokens)
        query_positions = np.where(tokens_array == self.query_embed_token_id)[0]
        doc_positions = np.where(tokens_array == self.doc_embed_token_id)[0]
        if len(query_positions) == 0:
            raise ValueError("Query rerank token was not found in tokenizer output")
        if len(doc_positions) == 0:
            raise ValueError("Document rerank tokens were not found in tokenizer output")

        query_hidden = embeddings[query_positions[0] : query_positions[0] + 1]
        doc_hidden = embeddings[doc_positions]
        query_embedding = self.projector(query_hidden)
        doc_embeddings = self.projector(doc_hidden)
        query_embeddings = np.tile(query_embedding, (len(doc_embeddings), 1))

        dot_product = np.sum(doc_embeddings * query_embeddings, axis=-1)
        doc_norm = np.sqrt(np.sum(doc_embeddings * doc_embeddings, axis=-1))
        query_norm = np.sqrt(np.sum(query_embeddings * query_embeddings, axis=-1))
        return dot_product / (doc_norm * query_norm)


class TorchRerankerAdapterMixin:
    def release(self):
        torch_module = getattr(self, "torch", None)
        self.model = None
        if hasattr(self, "tokenizer"):
            self.tokenizer = None
        if hasattr(self, "loaded_instruction"):
            self.loaded_instruction = None
        if hasattr(self, "device"):
            self.device = None
        if hasattr(self, "runtime_metadata"):
            self.runtime_metadata = {}

        gc.collect()
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError:
                return
        try:
            if not torch_module.cuda.is_available():
                return
            empty_cache = getattr(torch_module.cuda, "empty_cache", None)
            if empty_cache:
                empty_cache()
        except Exception as error:
            LOGGER.warning(
                "Could not clear CUDA cache after releasing reranker %s: %s"
                % (getattr(self, "backend_name", "unknown"), error)
            )


class JinaTransformersRerankerAdapter(TorchRerankerAdapterMixin):
    backend_name = "jina_transformers"

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.torch = None
        self.device = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._load_duration = 0.0
        self.runtime_metadata: Dict[str, object] = {}

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        model_reused = self.model is not None
        self._ensure_ready()
        start = time.perf_counter()
        with self._inference_lock, self.torch.inference_mode():
            scores, group_count = self._score_documents(
                query,
                documents,
                instruction.strip() if instruction else None,
            )
        scoring_duration = time.perf_counter() - start

        results = [
            RerankResult(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        self.runtime_metadata.update(
            {
                "document_count": len(documents),
                "group_count": group_count,
                "instruction_used": bool(instruction and instruction.strip()),
                "model_reused": model_reused,
                "load_seconds": round(self._load_duration, 3),
                "scoring_seconds": round(scoring_duration, 3),
            }
        )
        LOGGER.info(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend "
            "%s model %s across %s listwise groups on %s with dtype %s"
            % (
                len(documents),
                len(results),
                scoring_duration,
                self.backend_name,
                self.model_path,
                group_count,
                self.runtime_metadata["device"],
                self.runtime_metadata["dtype"],
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if not os.path.isdir(self.model_path):
                raise FileNotFoundError(
                    f"Native Jina reranker model path does not exist: {self.model_path}"
                )

            try:
                import torch
                from transformers import AutoModel
            except ImportError as error:
                raise RuntimeError(
                    "Native Jina reranker requires transformers and torch"
                ) from error

            self.torch = torch
            device, dtype, dtype_name = self._resolve_torch_runtime(torch)
            if self.config.jina_require_cuda and device == "cpu":
                raise RuntimeError(
                    "Native Jina reranker requires CUDA, but CUDA is unavailable"
                )
            attention = self.config.jina_attention.strip().lower()
            if attention not in {"eager", "sdpa", "flash_attention_2"}:
                raise ValueError(
                    "RERANK_JINA_ATTENTION must be eager, sdpa, or flash_attention_2"
                )

            start = time.perf_counter()
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation=attention,
            ).eval()
            model.to(device)
            self._load_duration = time.perf_counter() - start
            self.model = model
            self.device, dtype_name = self._loaded_runtime(device, dtype_name)
            self.runtime_metadata = {
                **_torch_runtime_details(torch, self.device, self.backend_name),
                "backend": self.backend_name,
                "device": self.device,
                "dtype": dtype_name,
                "attention": attention,
                "context_size": self.config.jina_context_size,
                "max_query_length": self.config.jina_max_query_length,
                "max_doc_length": self.config.jina_max_doc_length,
                "require_cuda": self.config.jina_require_cuda,
            }
            LOGGER.info(
                "Loaded native Jina reranker model %s in %.2fs on %s with "
                "dtype %s, attention %s, and context size %s"
                % (
                    self.model_path,
                    self._load_duration,
                    self.device,
                    dtype_name,
                    attention,
                    self.config.jina_context_size,
                )
            )

    def _score_documents(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str],
    ):
        if self.model is None:
            raise RuntimeError("Native Jina reranker model is not loaded")

        query, truncated_documents, document_lengths, query_length = (
            self.model._truncate_texts(
                query,
                documents,
                self.config.jina_max_query_length,
                self.config.jina_max_doc_length,
            )
        )
        groups = self._group_documents(
            document_lengths,
            query_length,
            instruction,
        )
        document_embeddings = []
        query_embeddings = []
        group_weights = []
        for group in groups:
            outputs = self.model._compute_single_batch(
                query,
                [truncated_documents[index] for index in group],
                instruction=instruction,
            )
            document_embeddings.extend(self._to_numpy(outputs.doc_embeds[0]))
            query_embeddings.append(self._to_numpy(outputs.query_embeds[0]))
            group_scores = self._to_numpy(outputs.scores).reshape(-1)
            group_weights.append(float(((1.0 + group_scores) / 2.0).max()))

        averaged_query = np.average(
            np.asarray(query_embeddings), axis=0, weights=group_weights
        )
        scores = self._cosine_scores(
            averaged_query,
            np.asarray(document_embeddings),
        )
        return scores, len(groups)

    def _group_documents(
        self,
        document_lengths: List[int],
        query_length: int,
        instruction: Optional[str],
    ) -> List[List[int]]:
        instruction_tokens = 0
        tokenizer = getattr(self.model, "_tokenizer", None)
        if instruction and tokenizer is not None:
            instruction_tokens = len(tokenizer(instruction)["input_ids"])
        fixed_tokens = (2 * query_length) + instruction_tokens + 256
        capacity = self.config.jina_context_size - fixed_tokens
        if capacity <= 0:
            raise ValueError(
                "RERANK_JINA_CONTEXT_SIZE is too small for the query and prompt"
            )

        groups: List[List[int]] = []
        current_group: List[int] = []
        remaining = capacity
        for index, length in enumerate(document_lengths):
            required = length + 16
            if required > capacity:
                raise ValueError(
                    "A Jina document does not fit after configured truncation; "
                    "increase RERANK_JINA_CONTEXT_SIZE or reduce "
                    "RERANK_JINA_MAX_DOC_LENGTH"
                )
            if current_group and required > remaining:
                groups.append(current_group)
                current_group = []
                remaining = capacity
            current_group.append(index)
            remaining -= required
        if current_group:
            groups.append(current_group)
        return groups

    def _to_numpy(self, value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return value.detach().float().cpu().numpy()

    def _cosine_scores(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        query = np.asarray(query_embedding).reshape(-1)
        dots = document_embeddings @ query
        denominator = np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(
            query
        )
        return dots / denominator

    def _resolve_torch_runtime(self, torch):
        requested_dtype = self.config.jina_dtype.strip().lower()
        if requested_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "RERANK_JINA_DTYPE must be auto, bfloat16, float16, or float32"
            )
        if not torch.cuda.is_available():
            return "cpu", torch.float32, "float32"

        device = "cuda:0"
        bf16_supported = getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()
        if requested_dtype == "auto":
            if bf16_supported:
                return device, torch.bfloat16, "bfloat16"
            return device, torch.float16, "float16"
        if requested_dtype == "bfloat16":
            if not bf16_supported:
                raise RuntimeError("The configured GPU does not support bfloat16")
            return device, torch.bfloat16, "bfloat16"
        if requested_dtype == "float16":
            return device, torch.float16, "float16"
        return device, torch.float32, "float32"

    def _loaded_runtime(self, default_device: str, default_dtype: str):
        try:
            parameter = next(self.model.parameters())
        except (AttributeError, StopIteration):
            return default_device, default_dtype
        dtype = str(parameter.dtype).removeprefix("torch.")
        return str(parameter.device), dtype


class SentenceTransformersCrossEncoderRerankerAdapter(
    TorchRerankerAdapterMixin
):
    backend_name = "sentence_transformers_cross_encoder"
    supports_instruction = False

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        self.model_path = model_path
        self.config = config
        self.torch = None
        self.device = None
        self.model = None
        self.loaded_instruction: Optional[str] = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready(instruction)
        start = time.perf_counter()
        pairs = [(query, document) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.document_batch_size,
            show_progress_bar=False,
        )
        results = [
            RerankResult(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.debug(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s and document batch size %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.config.document_batch_size,
            )
        )
        return results

    def _ensure_ready(self, instruction: Optional[str]):
        normalized_instruction = (
            instruction.strip() if instruction and self.supports_instruction else None
        )
        if self.model is not None and self.loaded_instruction == normalized_instruction:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformers reranker requires the sentence-transformers package"
            ) from e
        self.torch = torch

        kwargs = {}
        if _env_bool("RERANK_TRUST_REMOTE_CODE"):
            kwargs["trust_remote_code"] = True
        if torch.cuda.is_available():
            kwargs["device"] = "cuda"
            self.device = "cuda"
        else:
            self.device = "cpu"

        if normalized_instruction:
            kwargs["prompts"] = {"rerank": normalized_instruction}
            kwargs["default_prompt_name"] = "rerank"

        self.model = CrossEncoder(self.model_path, **kwargs)
        self.loaded_instruction = normalized_instruction


class QwenCrossEncoderRerankerAdapter(SentenceTransformersCrossEncoderRerankerAdapter):
    backend_name = "qwen_cross_encoder"
    supports_instruction = True

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        super().__init__(model_path, config)
        self.runtime_metadata: Dict[str, object] = {}

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        pairs = [(query, document) for document in documents]
        predict_kwargs: Dict[str, object] = {
            "batch_size": self.config.document_batch_size,
            "show_progress_bar": False,
        }
        if instruction and instruction.strip():
            predict_kwargs["prompt"] = instruction.strip()

        scores = self.model.predict(pairs, **predict_kwargs)
        results = [
            RerankResult(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.info(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s, device %s, dtype %s, attention %s, max length %s, and "
            "document batch size %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.runtime_metadata["device"],
                self.runtime_metadata["dtype"],
                self.runtime_metadata["attention"],
                self.runtime_metadata["max_length"],
                self.runtime_metadata["batch_size"],
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "Qwen reranker requires sentence-transformers and torch"
            ) from e
        self.torch = torch

        device, dtype, dtype_name = self._resolve_torch_runtime(torch)
        self.device = device
        attention = self.config.qwen_attention.strip().lower()
        if attention not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "RERANK_QWEN_ATTENTION must be eager, sdpa, or flash_attention_2"
            )

        start = time.perf_counter()
        self.model = CrossEncoder(
            self.model_path,
            device=device,
            max_length=self.config.qwen_max_length,
            model_kwargs={
                "torch_dtype": dtype,
                "attn_implementation": attention,
            },
        )
        load_duration = time.perf_counter() - start
        device, dtype_name = self._loaded_runtime(device, dtype_name)
        self.runtime_metadata = {
            **_torch_runtime_details(torch, device, self.backend_name),
            "device": device,
            "dtype": dtype_name,
            "attention": attention,
            "max_length": self.config.qwen_max_length,
            "batch_size": self.config.document_batch_size,
            "configured_batch_size": self.config.document_batch_size,
            "effective_batch_size": self.config.document_batch_size,
        }
        LOGGER.info(
            "Loaded Qwen reranker model %s in %.2fs on %s with dtype %s, "
            "attention %s, and max length %s"
            % (
                self.model_path,
                load_duration,
                device,
                dtype_name,
                attention,
                self.config.qwen_max_length,
            )
        )

    def _resolve_torch_runtime(self, torch):
        requested_dtype = self.config.qwen_dtype.strip().lower()
        if requested_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "RERANK_QWEN_DTYPE must be auto, bfloat16, float16, or float32"
            )

        if not torch.cuda.is_available():
            return "cpu", torch.float32, "float32"

        device = "cuda:0"
        bf16_supported = getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()
        if requested_dtype == "auto":
            if bf16_supported:
                return device, torch.bfloat16, "bfloat16"
            return device, torch.float16, "float16"
        if requested_dtype == "bfloat16":
            if not bf16_supported:
                raise RuntimeError("The configured GPU does not support bfloat16")
            return device, torch.bfloat16, "bfloat16"
        if requested_dtype == "float16":
            return device, torch.float16, "float16"
        return device, torch.float32, "float32"

    def _loaded_runtime(self, default_device: str, default_dtype: str):
        try:
            parameter = next(self.model.model.parameters())
        except (AttributeError, StopIteration):
            return default_device, default_dtype
        dtype = str(parameter.dtype).removeprefix("torch.")
        return str(parameter.device), dtype


class TransformersSequenceClassificationRerankerAdapter(
    TorchRerankerAdapterMixin
):
    backend_name = "transformers_sequence_classification"

    def __init__(
        self,
        model_path: str,
        config: RerankerRuntimeConfig,
        trust_remote_code: bool = False,
        single_text_template: Optional[str] = None,
    ):
        self.model_path = model_path
        self.config = config
        self.trust_remote_code = trust_remote_code
        self.single_text_template = single_text_template
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        scores = []
        for batch_start in range(0, len(documents), self.config.document_batch_size):
            batch_documents = documents[
                batch_start : batch_start + self.config.document_batch_size
            ]
            scores.extend(self._score_batch(query, batch_documents))

        results = [
            RerankResult(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.debug(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s and document batch size %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.config.document_batch_size,
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Transformers reranker requires the transformers and torch packages"
            ) from e

        self.torch = torch
        kwargs = {"trust_remote_code": self.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, **kwargs
        ).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _score_batch(self, query: str, documents: List[str]) -> List[float]:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Transformers reranker model is not loaded")

        if self.single_text_template:
            inputs = [
                self.single_text_template.format(query=query, document=document)
                for document in documents
            ]
            encoded = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        for key in encoded:
            encoded[key] = encoded[key].to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**encoded)
            return outputs.logits.view(-1).float().detach().cpu().tolist()


class BgeCrossEncoderRerankerAdapter(
    TransformersSequenceClassificationRerankerAdapter
):
    backend_name = "bge_cross_encoder"

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        super().__init__(model_path=model_path, config=config)
        self.runtime_metadata: Dict[str, object] = {}

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        encoded_documents = self._tokenize_documents(query, documents)
        effective_batch_size = self.config.bge_batch_size

        while True:
            try:
                scores = self._score_encoded_documents(
                    encoded_documents, effective_batch_size
                )
                break
            except RuntimeError as error:
                if not self._is_cuda_oom(error) or effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                empty_cache = getattr(self.torch.cuda, "empty_cache", None)
                if empty_cache:
                    empty_cache()
                LOGGER.warning(
                    "BGE reranker ran out of GPU memory; retrying all candidate "
                    "chunks with batch size %s" % effective_batch_size
                )

        self.runtime_metadata["effective_batch_size"] = effective_batch_size
        results = [
            RerankResult(index=index, relevance_score=float(scores[index]))
            for index in range(len(documents))
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.info(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s, device %s, dtype %s, attention %s, max length %s, and "
            "effective batch size %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.runtime_metadata["device"],
                self.runtime_metadata["dtype"],
                self.runtime_metadata["attention"],
                self.runtime_metadata["max_length"],
                effective_batch_size,
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "BGE reranker requires the transformers and torch packages"
            ) from error

        self.torch = torch
        device, dtype, dtype_name = self._resolve_torch_runtime(torch)
        attention = self.config.bge_attention.strip().lower()
        if attention not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "RERANK_BGE_ATTENTION must be eager, sdpa, or flash_attention_2"
            )

        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            attn_implementation=attention,
        ).eval()
        self.model.to(device)
        load_duration = time.perf_counter() - start
        self.device, dtype_name = self._loaded_runtime(device, dtype_name)
        self.runtime_metadata = {
            **_torch_runtime_details(torch, self.device, self.backend_name),
            "backend": self.backend_name,
            "device": self.device,
            "dtype": dtype_name,
            "attention": attention,
            "max_length": self.config.bge_max_length,
            "configured_batch_size": self.config.bge_batch_size,
            "effective_batch_size": self.config.bge_batch_size,
        }
        LOGGER.info(
            "Loaded BGE reranker model %s in %.2fs on %s with dtype %s, "
            "attention %s, max length %s, and configured batch size %s"
            % (
                self.model_path,
                load_duration,
                self.device,
                dtype_name,
                attention,
                self.config.bge_max_length,
                self.config.bge_batch_size,
            )
        )

    def _tokenize_documents(self, query: str, documents: List[str]):
        encoded = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=False,
            truncation="only_second",
            max_length=self.config.bge_max_length,
        )
        encoded_documents = []
        for index in range(len(documents)):
            item = {key: values[index] for key, values in encoded.items()}
            encoded_documents.append((index, item))
        encoded_documents.sort(
            key=lambda indexed_item: len(indexed_item[1]["input_ids"]),
            reverse=True,
        )
        return encoded_documents

    def _score_encoded_documents(self, encoded_documents, batch_size: int):
        scores: Dict[int, float] = {}
        for batch_start in range(0, len(encoded_documents), batch_size):
            batch = encoded_documents[batch_start : batch_start + batch_size]
            padded = self.tokenizer.pad(
                [item for _, item in batch],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            padded = {key: value.to(self.device) for key, value in padded.items()}
            with self.torch.inference_mode():
                logits = self.model(**padded, return_dict=True).logits
                batch_scores = logits.view(-1).float().detach().cpu().tolist()
            for (original_index, _), score in zip(batch, batch_scores):
                scores[original_index] = float(score)
        return scores

    def _resolve_torch_runtime(self, torch):
        requested_dtype = self.config.bge_dtype.strip().lower()
        if requested_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "RERANK_BGE_DTYPE must be auto, bfloat16, float16, or float32"
            )

        if not torch.cuda.is_available():
            return "cpu", torch.float32, "float32"

        device = "cuda:0"
        bf16_supported = getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()
        if requested_dtype == "auto":
            if bf16_supported:
                return device, torch.bfloat16, "bfloat16"
            return device, torch.float16, "float16"
        if requested_dtype == "bfloat16":
            if not bf16_supported:
                raise RuntimeError("The configured GPU does not support bfloat16")
            return device, torch.bfloat16, "bfloat16"
        if requested_dtype == "float16":
            return device, torch.float16, "float16"
        return device, torch.float32, "float32"

    def _loaded_runtime(self, default_device: str, default_dtype: str):
        try:
            parameter = next(self.model.parameters())
        except (AttributeError, StopIteration):
            return default_device, default_dtype
        dtype = str(parameter.dtype).removeprefix("torch.")
        return str(parameter.device), dtype

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        if not str(self.device).startswith("cuda"):
            return False
        oom_error = getattr(self.torch.cuda, "OutOfMemoryError", None)
        return (oom_error is not None and isinstance(error, oom_error)) or (
            "out of memory" in str(error).lower()
        )


class MixedbreadCrossEncoderRerankerAdapter(
    SentenceTransformersCrossEncoderRerankerAdapter
):
    backend_name = "mixedbread_cross_encoder"
    supports_instruction = True

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        super().__init__(model_path=model_path, config=config)
        self.torch = None
        self.device = None
        self.runtime_metadata: Dict[str, object] = {}

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        indexed_pairs = self._length_sorted_pairs(query, documents)
        effective_batch_size = self.config.mixedbread_batch_size
        normalized_instruction = instruction.strip() if instruction else None

        while True:
            try:
                scores = self._score_pairs(
                    indexed_pairs,
                    effective_batch_size,
                    normalized_instruction,
                )
                break
            except RuntimeError as error:
                if not self._is_cuda_oom(error) or effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                empty_cache = getattr(self.torch.cuda, "empty_cache", None)
                if empty_cache:
                    empty_cache()
                LOGGER.warning(
                    "Mixedbread reranker ran out of GPU memory; retrying all "
                    "candidate chunks with batch size %s" % effective_batch_size
                )

        self.runtime_metadata["effective_batch_size"] = effective_batch_size
        self.runtime_metadata["document_count"] = len(documents)
        self.runtime_metadata["instruction_used"] = bool(normalized_instruction)
        results = [
            RerankResult(index=index, relevance_score=float(scores[index]))
            for index in range(len(documents))
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.info(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s, device %s, dtype %s, attention %s, max length %s, "
            "effective batch size %s, and instruction %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.runtime_metadata["device"],
                self.runtime_metadata["dtype"],
                self.runtime_metadata["attention"],
                self.runtime_metadata["max_length"],
                effective_batch_size,
                "enabled" if normalized_instruction else "disabled",
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Mixedbread reranker requires sentence-transformers and torch"
            ) from error

        self.torch = torch
        device, dtype, dtype_name = self._resolve_torch_runtime(torch)
        attention = self.config.mixedbread_attention.strip().lower()
        if attention not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "RERANK_MIXEDBREAD_ATTENTION must be eager, sdpa, or "
                "flash_attention_2"
            )

        start = time.perf_counter()
        self.model = CrossEncoder(
            self.model_path,
            device=device,
            max_length=self.config.mixedbread_max_length,
            model_kwargs={
                "torch_dtype": dtype,
                "attn_implementation": attention,
            },
        )
        load_duration = time.perf_counter() - start
        self.device, dtype_name = self._loaded_runtime(device, dtype_name)
        self.runtime_metadata = {
            **_torch_runtime_details(torch, self.device, self.backend_name),
            "backend": self.backend_name,
            "device": self.device,
            "dtype": dtype_name,
            "attention": attention,
            "max_length": self.config.mixedbread_max_length,
            "configured_batch_size": self.config.mixedbread_batch_size,
            "effective_batch_size": self.config.mixedbread_batch_size,
            "document_count": 0,
            "instruction_used": False,
        }
        LOGGER.info(
            "Loaded Mixedbread reranker model %s in %.2fs on %s with dtype %s, "
            "attention %s, max length %s, and configured batch size %s"
            % (
                self.model_path,
                load_duration,
                self.device,
                dtype_name,
                attention,
                self.config.mixedbread_max_length,
                self.config.mixedbread_batch_size,
            )
        )

    def _length_sorted_pairs(self, query: str, documents: List[str]):
        pairs = [(query, document) for document in documents]
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not documents:
            return list(enumerate(pairs))

        encoded = tokenizer(
            [query] * len(documents),
            documents,
            padding=False,
            truncation=True,
            max_length=self.config.mixedbread_max_length,
        )
        indexed_pairs = list(enumerate(pairs))
        indexed_pairs.sort(
            key=lambda item: len(encoded["input_ids"][item[0]]),
            reverse=True,
        )
        return indexed_pairs

    def _score_pairs(
        self,
        indexed_pairs,
        batch_size: int,
        instruction: Optional[str],
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for batch_start in range(0, len(indexed_pairs), batch_size):
            batch = indexed_pairs[batch_start : batch_start + batch_size]
            predict_kwargs: Dict[str, object] = {
                "batch_size": batch_size,
                "show_progress_bar": False,
            }
            if instruction:
                predict_kwargs["prompt"] = instruction
            batch_scores = self.model.predict(
                [pair for _, pair in batch],
                **predict_kwargs,
            )
            for (original_index, _), score in zip(batch, batch_scores):
                scores[original_index] = float(score)
        return scores

    def _resolve_torch_runtime(self, torch):
        requested_dtype = self.config.mixedbread_dtype.strip().lower()
        if requested_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "RERANK_MIXEDBREAD_DTYPE must be auto, bfloat16, float16, "
                "or float32"
            )

        if not torch.cuda.is_available():
            return "cpu", torch.float32, "float32"

        device = "cuda:0"
        bf16_supported = getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()
        if requested_dtype == "auto":
            if bf16_supported:
                return device, torch.bfloat16, "bfloat16"
            return device, torch.float16, "float16"
        if requested_dtype == "bfloat16":
            if not bf16_supported:
                raise RuntimeError("The configured GPU does not support bfloat16")
            return device, torch.bfloat16, "bfloat16"
        if requested_dtype == "float16":
            return device, torch.float16, "float16"
        return device, torch.float32, "float32"

    def _loaded_runtime(self, default_device: str, default_dtype: str):
        try:
            parameter = next(self.model.model.parameters())
        except (AttributeError, StopIteration):
            return default_device, default_dtype
        dtype = str(parameter.dtype).removeprefix("torch.")
        return str(parameter.device), dtype

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        if not str(self.device).startswith("cuda"):
            return False
        oom_error = getattr(self.torch.cuda, "OutOfMemoryError", None)
        return (oom_error is not None and isinstance(error, oom_error)) or (
            "out of memory" in str(error).lower()
        )


class NvidiaCrossEncoderRerankerAdapter(
    TransformersSequenceClassificationRerankerAdapter
):
    backend_name = "nvidia_cross_encoder"

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        super().__init__(
            model_path=model_path,
            config=config,
            trust_remote_code=True,
            single_text_template="question:{query} \n \n passage:{document}",
        )
        self.runtime_metadata: Dict[str, object] = {}

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        self._ensure_ready()
        start = time.perf_counter()
        encoded_documents = self._tokenize_documents(query, documents)
        effective_batch_size = self.config.nvidia_batch_size

        while True:
            try:
                scores = self._score_encoded_documents(
                    encoded_documents, effective_batch_size
                )
                break
            except RuntimeError as error:
                if not self._is_cuda_oom(error) or effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                empty_cache = getattr(self.torch.cuda, "empty_cache", None)
                if empty_cache:
                    empty_cache()
                LOGGER.warning(
                    "NVIDIA reranker ran out of GPU memory; retrying all "
                    "candidate chunks with batch size %s" % effective_batch_size
                )

        self.runtime_metadata["effective_batch_size"] = effective_batch_size
        self.runtime_metadata["document_count"] = len(documents)
        results = [
            RerankResult(index=index, relevance_score=float(scores[index]))
            for index in range(len(documents))
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.info(
            "Reranked %s candidate chunks to %s chunks in %.2fs using backend %s "
            "model %s, device %s, dtype %s, attention %s, max length %s, "
            "effective batch size %s, and use cache %s"
            % (
                len(documents),
                len(results),
                time.perf_counter() - start,
                self.backend_name,
                self.model_path,
                self.runtime_metadata["device"],
                self.runtime_metadata["dtype"],
                self.runtime_metadata["attention"],
                self.runtime_metadata["max_length"],
                effective_batch_size,
                self.runtime_metadata["use_cache"],
            )
        )
        return results

    def _ensure_ready(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Reranker model path does not exist: {self.model_path}"
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "NVIDIA reranker requires transformers and torch"
            ) from error

        self.torch = torch
        device, dtype, dtype_name = self._resolve_torch_runtime(torch)
        attention = self.config.nvidia_attention.strip().lower()
        if attention not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "RERANK_NVIDIA_ATTENTION must be eager, sdpa, or "
                "flash_attention_2"
            )

        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise RuntimeError(
                    "NVIDIA reranker tokenizer has neither a pad token nor an "
                    "EOS token"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attention,
        ).eval()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = False
        self.model.to(device)
        load_duration = time.perf_counter() - start

        self.device, dtype_name = self._loaded_runtime(device, dtype_name)
        self.runtime_metadata = {
            **_torch_runtime_details(torch, self.device, self.backend_name),
            "backend": self.backend_name,
            "device": self.device,
            "dtype": dtype_name,
            "attention": attention,
            "max_length": self.config.nvidia_max_length,
            "configured_batch_size": self.config.nvidia_batch_size,
            "effective_batch_size": self.config.nvidia_batch_size,
            "use_cache": False,
            "document_count": 0,
        }
        LOGGER.info(
            "Loaded NVIDIA reranker model %s in %.2fs on %s with dtype %s, "
            "attention %s, max length %s, configured batch size %s, and no "
            "KV cache"
            % (
                self.model_path,
                load_duration,
                self.device,
                dtype_name,
                attention,
                self.config.nvidia_max_length,
                self.config.nvidia_batch_size,
            )
        )

    def _format_input(self, query: str, document: str) -> str:
        return self.single_text_template.format(query=query, document=document)

    def _tokenize_documents(self, query: str, documents: List[str]):
        texts = [self._format_input(query, document) for document in documents]
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.config.nvidia_max_length,
        )
        encoded_documents = []
        for index in range(len(documents)):
            item = {key: values[index] for key, values in encoded.items()}
            encoded_documents.append((index, item))
        encoded_documents.sort(
            key=lambda indexed_item: len(indexed_item[1]["input_ids"]),
            reverse=True,
        )
        return encoded_documents

    def _score_encoded_documents(self, encoded_documents, batch_size: int):
        scores: Dict[int, float] = {}
        for batch_start in range(0, len(encoded_documents), batch_size):
            batch = encoded_documents[batch_start : batch_start + batch_size]
            padded = self.tokenizer.pad(
                [item for _, item in batch],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            padded = {key: value.to(self.device) for key, value in padded.items()}
            with self.torch.inference_mode():
                logits = self.model(
                    **padded,
                    use_cache=False,
                    return_dict=True,
                ).logits
                batch_scores = logits.view(-1).float().detach().cpu().tolist()
            for (original_index, _), score in zip(batch, batch_scores):
                scores[original_index] = float(score)
        return scores

    def _resolve_torch_runtime(self, torch):
        requested_dtype = self.config.nvidia_dtype.strip().lower()
        if requested_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "RERANK_NVIDIA_DTYPE must be auto, bfloat16, float16, or float32"
            )

        if not torch.cuda.is_available():
            return "cpu", torch.float32, "float32"

        device = "cuda:0"
        bf16_supported = getattr(
            torch.cuda, "is_bf16_supported", lambda: False
        )()
        if requested_dtype == "auto":
            if bf16_supported:
                return device, torch.bfloat16, "bfloat16"
            return device, torch.float16, "float16"
        if requested_dtype == "bfloat16":
            if not bf16_supported:
                raise RuntimeError("The configured GPU does not support bfloat16")
            return device, torch.bfloat16, "bfloat16"
        if requested_dtype == "float16":
            return device, torch.float16, "float16"
        return device, torch.float32, "float32"

    def _loaded_runtime(self, default_device: str, default_dtype: str):
        try:
            parameter = next(self.model.parameters())
        except (AttributeError, StopIteration):
            return default_device, default_dtype
        dtype = str(parameter.dtype).removeprefix("torch.")
        return str(parameter.device), dtype

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        if not str(self.device).startswith("cuda"):
            return False
        oom_error = getattr(self.torch.cuda, "OutOfMemoryError", None)
        return (oom_error is not None and isinstance(error, oom_error)) or (
            "out of memory" in str(error).lower()
        )


def _infer_model_type(model_path: str) -> str:
    lowered_path = model_path.lower()
    if lowered_path.endswith(".gguf"):
        return "jina_gguf"
    if "jina" in lowered_path and os.path.isdir(model_path):
        return "jina_transformers"
    if "qwen" in lowered_path or "quen3" in lowered_path:
        return "qwen_cross_encoder"
    if "bge" in lowered_path:
        return "bge_cross_encoder"
    if "mixedbread" in lowered_path or "mxbai-rerank" in lowered_path:
        return "mixedbread_cross_encoder"
    if "nvidia" in lowered_path:
        return "nvidia_cross_encoder"
    return "sentence_transformers_cross_encoder"


class RerankerService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = _env_bool("RERANK_ENABLED")
        self.model_paths = (
            _env_list("RERANK_MODEL_PATHS")
            or _env_list("RERANK_MODEL_PATH")
            or ["/app/models/reranker/jina-reranker-v3-Q4_K_M.gguf"]
        )
        self.model_types = _env_list("RERANK_MODEL_TYPES")
        self.projector_paths = (
            _env_list("RERANK_PROJECTOR_PATHS")
            or _env_list("RERANK_PROJECTOR_PATH")
            or ["/app/models/reranker/projector.safetensors"]
        )
        self.rerank_instructions = self._load_rerank_instructions()
        self.config = RerankerRuntimeConfig(
            llama_embedding_path=os.getenv(
                "RERANK_LLAMA_EMBEDDING_PATH", "/usr/local/bin/llama-embedding"
            ),
            llama_tokenize_path=os.getenv(
                "RERANK_LLAMA_TOKENIZE_PATH", "/usr/local/bin/llama-tokenize"
            ),
            timeout_seconds=int(os.getenv("RERANK_TIMEOUT_SECONDS", "60")),
            context_size=int(os.getenv("RERANK_CONTEXT_SIZE", "20000")),
            ubatch_size=int(os.getenv("RERANK_UBATCH_SIZE", "512")),
            document_batch_size=max(
                1, int(os.getenv("RERANK_DOCUMENT_BATCH_SIZE", "4"))
            ),
            gpu_layers=int(os.getenv("RERANK_GPU_LAYERS", "0")),
            flash_attention=_env_bool("RERANK_FLASH_ATTENTION"),
            jina_context_size=max(
                1, int(os.getenv("RERANK_JINA_CONTEXT_SIZE", "20000"))
            ),
            jina_max_query_length=max(
                1, int(os.getenv("RERANK_JINA_MAX_QUERY_LENGTH", "512"))
            ),
            jina_max_doc_length=max(
                1, int(os.getenv("RERANK_JINA_MAX_DOC_LENGTH", "2048"))
            ),
            jina_dtype=os.getenv("RERANK_JINA_DTYPE", "auto"),
            jina_attention=os.getenv("RERANK_JINA_ATTENTION", "sdpa"),
            jina_require_cuda=_env_bool(
                "RERANK_JINA_REQUIRE_CUDA", "true"
            ),
            qwen_max_length=max(
                1, int(os.getenv("RERANK_QWEN_MAX_LENGTH", "8192"))
            ),
            qwen_dtype=os.getenv("RERANK_QWEN_DTYPE", "auto"),
            qwen_attention=os.getenv("RERANK_QWEN_ATTENTION", "sdpa"),
            bge_max_length=max(
                1, int(os.getenv("RERANK_BGE_MAX_LENGTH", "8192"))
            ),
            bge_dtype=os.getenv("RERANK_BGE_DTYPE", "auto"),
            bge_attention=os.getenv("RERANK_BGE_ATTENTION", "sdpa"),
            bge_batch_size=max(
                1, int(os.getenv("RERANK_BGE_BATCH_SIZE", "2"))
            ),
            mixedbread_max_length=max(
                1, int(os.getenv("RERANK_MIXEDBREAD_MAX_LENGTH", "8192"))
            ),
            mixedbread_dtype=os.getenv("RERANK_MIXEDBREAD_DTYPE", "auto"),
            mixedbread_attention=os.getenv(
                "RERANK_MIXEDBREAD_ATTENTION", "sdpa"
            ),
            mixedbread_batch_size=max(
                1, int(os.getenv("RERANK_MIXEDBREAD_BATCH_SIZE", "2"))
            ),
            nvidia_max_length=max(
                1, int(os.getenv("RERANK_NVIDIA_MAX_LENGTH", "8192"))
            ),
            nvidia_dtype=os.getenv("RERANK_NVIDIA_DTYPE", "auto"),
            nvidia_attention=os.getenv(
                "RERANK_NVIDIA_ATTENTION", "sdpa"
            ),
            nvidia_batch_size=max(
                1, int(os.getenv("RERANK_NVIDIA_BATCH_SIZE", "1"))
            ),
        )
        self.adapters = self._build_adapters()
        self._rerank_lock = threading.Lock()
        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        return self.rerank_with_metadata(query, documents, top_n, instruction).results

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> RerankExecution:
        lock = getattr(self, "_rerank_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._rerank_lock = lock
        with lock:
            return self._rerank_with_metadata_locked(
                query,
                documents,
                top_n,
                instruction,
            )

    def _rerank_with_metadata_locked(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> RerankExecution:
        if not self.enabled or not documents:
            return RerankExecution([], "", "")
        if not self.adapters:
            raise RuntimeError("No reranker models are configured")

        last_error: Optional[Exception] = None
        for adapter in self.adapters:
            self._release_other_adapters(adapter)
            try:
                adapter_instruction = (
                    None
                    if adapter.backend_name == "nvidia_cross_encoder"
                    else instruction
                )
                if adapter_instruction:
                    LOGGER.debug(
                        "Using configured reranker instruction for backend %s model %s"
                        % (adapter.backend_name, adapter.model_path)
                    )
                return RerankExecution(
                    results=adapter.rerank(
                        query=query,
                        documents=documents,
                        top_n=top_n,
                        instruction=adapter_instruction,
                    ),
                    backend_name=adapter.backend_name,
                    model_path=adapter.model_path,
                    runtime=dict(getattr(adapter, "runtime_metadata", {})),
                )
            except Exception as error:
                error_message = str(error)
                error_type = type(error).__name__
                LOGGER.error(
                    f"Reranker backend <{adapter.backend_name}> model "
                    f"<{adapter.model_path}> failed, trying next configured "
                    f"model if available: {error_message}"
                )
                error_traceback = error.__traceback__
                if error_traceback is not None:
                    traceback.clear_frames(error_traceback)
                error.__traceback__ = None
                self._release_adapter(adapter, reason="failure")
                last_error = RuntimeError(
                    f"{error_type}: {error_message}"
                )

        raise RuntimeError("All configured reranker models failed") from last_error

    def _release_other_adapters(self, active_adapter):
        for adapter in self.adapters:
            if adapter is active_adapter:
                continue
            if getattr(adapter, "model", None) is None:
                continue
            self._release_adapter(
                adapter,
                reason="before trying %s" % active_adapter.backend_name,
            )

    def _release_adapter(self, adapter, reason: str):
        release = getattr(adapter, "release", None)
        if release is None:
            return

        before = _cuda_memory_snapshot()
        release()
        after = _cuda_memory_snapshot()
        LOGGER.info(
            "Released reranker backend <%s> model <%s> after %s; "
            "GPU memory before=%s after=%s"
            % (
                adapter.backend_name,
                adapter.model_path,
                reason,
                before or "unavailable",
                after or "unavailable",
            )
        )

    def get_instruction(self, dataset_id: str) -> Optional[str]:
        return self.rerank_instructions.get(dataset_id)

    def _load_rerank_instructions(self) -> Dict[str, str]:
        instructions_file = os.getenv("RERANK_INSTRUCTIONS_FILE")
        if not instructions_file:
            return {}

        file_path = Path("/app/data") / instructions_file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Reranker instructions file does not exist: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in reranker instructions file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Reranker instructions file must be a JSON object.")

        instructions = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid reranker instruction key: {key}")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid reranker instruction text for {key}.")
            instructions[key] = value.strip()

        LOGGER.info(
            "Loaded reranker instructions for datasets: "
            + ", ".join(sorted(instructions.keys()))
        )
        return instructions

    def _build_adapters(self):
        adapters = []
        jina_model_index = 0
        for model_index, model_path in enumerate(self.model_paths):
            model_type = self._get_model_type(model_index, model_path)
            if model_type == "jina_gguf":
                projector_path = self._get_projector_path(jina_model_index)
                adapters.append(
                    JinaGgufRerankerAdapter(
                        model_path=model_path,
                        projector_path=projector_path,
                        config=self.config,
                    )
                )
                jina_model_index += 1
            elif model_type == "jina_transformers":
                adapters.append(
                    JinaTransformersRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "qwen_cross_encoder":
                adapters.append(
                    QwenCrossEncoderRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "sentence_transformers_cross_encoder":
                adapters.append(
                    SentenceTransformersCrossEncoderRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "bge_cross_encoder":
                adapters.append(
                    BgeCrossEncoderRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "mixedbread_cross_encoder":
                adapters.append(
                    MixedbreadCrossEncoderRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "nvidia_cross_encoder":
                adapters.append(
                    NvidiaCrossEncoderRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            else:
                LOGGER.error(
                    f"Unsupported reranker model type <{model_type}> for "
                    f"model <{model_path}>. Skipping this model."
                )
        return adapters

    def _get_model_type(self, model_index: int, model_path: str) -> str:
        if model_index < len(self.model_types):
            return self.model_types[model_index]
        return _infer_model_type(model_path)

    def _get_projector_path(self, jina_model_index: int) -> str:
        if len(self.projector_paths) == 1:
            return self.projector_paths[0]
        if jina_model_index < len(self.projector_paths):
            return self.projector_paths[jina_model_index]
        return self.projector_paths[-1]
