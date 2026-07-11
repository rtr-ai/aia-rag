import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import TypeAdapter

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _env_bool(name: str, default: str = "false") -> bool:
    return TypeAdapter(bool).validate_python(os.getenv(name, default))


def _env_list(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


@dataclass
class RerankResult:
    index: int
    relevance_score: float


@dataclass
class RerankExecution:
    results: List[RerankResult]
    backend_name: str
    model_path: str


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


class SentenceTransformersCrossEncoderRerankerAdapter:
    backend_name = "sentence_transformers_cross_encoder"
    supports_instruction = False

    def __init__(self, model_path: str, config: RerankerRuntimeConfig):
        self.model_path = model_path
        self.config = config
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
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformers reranker requires the sentence-transformers package"
            ) from e

        kwargs = {}
        if _env_bool("RERANK_TRUST_REMOTE_CODE"):
            kwargs["trust_remote_code"] = True
        try:
            import torch

            if torch.cuda.is_available():
                kwargs["device"] = "cuda"
        except ImportError:
            pass

        if normalized_instruction:
            kwargs["prompts"] = {"rerank": normalized_instruction}
            kwargs["default_prompt_name"] = "rerank"

        self.model = CrossEncoder(self.model_path, **kwargs)
        self.loaded_instruction = normalized_instruction


class QwenCrossEncoderRerankerAdapter(SentenceTransformersCrossEncoderRerankerAdapter):
    backend_name = "qwen_cross_encoder"
    supports_instruction = True


class TransformersSequenceClassificationRerankerAdapter:
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


def _infer_model_type(model_path: str) -> str:
    lowered_path = model_path.lower()
    if lowered_path.endswith(".gguf"):
        return "jina_gguf"
    if "qwen" in lowered_path or "quen3" in lowered_path:
        return "qwen_cross_encoder"
    if "bge" in lowered_path:
        return "bge_cross_encoder"
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
        )
        self.adapters = self._build_adapters()
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
        self, query: str, documents: List[str], top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> RerankExecution:
        if not self.enabled or not documents:
            return RerankExecution([], "", "")
        if not self.adapters:
            raise RuntimeError("No reranker models are configured")

        last_error: Optional[Exception] = None
        for adapter in self.adapters:
            try:
                adapter_instruction = instruction
                if adapter_instruction:
                    LOGGER.debug(
                        "Using configured reranker instruction for backend %s model %s"
                        % (adapter.backend_name, adapter.model_path)
                    )
                return RerankExecution(
                    results=adapter.rerank(query=query, documents=documents, top_n=top_n, instruction=adapter_instruction),
                    backend_name=adapter.backend_name,
                    model_path=adapter.model_path,
                )
            except Exception as e:
                last_error = e
                LOGGER.error(
                    f"Reranker backend <{adapter.backend_name}> model "
                    f"<{adapter.model_path}> failed, trying next configured "
                    f"model if available: {e}"
                )

        raise RuntimeError("All configured reranker models failed") from last_error

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
                    TransformersSequenceClassificationRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                    )
                )
            elif model_type == "nvidia_cross_encoder":
                adapters.append(
                    TransformersSequenceClassificationRerankerAdapter(
                        model_path=model_path,
                        config=self.config,
                        trust_remote_code=True,
                        single_text_template="question:{query} \n\n passage:{document}",
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
