import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pydantic import TypeAdapter

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _env_bool(name: str, default: str = "false") -> bool:
    return TypeAdapter(bool).validate_python(os.getenv(name, default))


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class MLPProjector:
    def __init__(self, linear1_weight, linear2_weight):
        self.linear1_weight = linear1_weight
        self.linear2_weight = linear2_weight

    def __call__(self, value):
        value = value @ self.linear1_weight.T
        value = np.maximum(0, value)
        return value @ self.linear2_weight.T


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
        self.model_path = os.getenv(
            "RERANK_MODEL_PATH",
            "/app/models/reranker/jina-reranker-v3-Q4_K_M.gguf",
        )
        self.projector_path = os.getenv(
            "RERANK_PROJECTOR_PATH",
            "/app/models/reranker/projector.safetensors",
        )
        self.llama_embedding_path = os.getenv(
            "RERANK_LLAMA_EMBEDDING_PATH", "/usr/local/bin/llama-embedding"
        )
        self.llama_tokenize_path = os.getenv(
            "RERANK_LLAMA_TOKENIZE_PATH", "/usr/local/bin/llama-tokenize"
        )
        self.timeout_seconds = int(os.getenv("RERANK_TIMEOUT_SECONDS", "60"))
        self.context_size = int(os.getenv("RERANK_CONTEXT_SIZE", "8192"))
        self.ubatch_size = int(os.getenv("RERANK_UBATCH_SIZE", "512"))
        self.gpu_layers = int(os.getenv("RERANK_GPU_LAYERS", "0"))
        self.flash_attention = _env_bool("RERANK_FLASH_ATTENTION")
        self.projector: Optional[MLPProjector] = None
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.query_embed_token_id = 151671
        self.doc_embed_token_id = 151670
        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        if not self.enabled:
            return []
        if not documents:
            return []

        self._ensure_ready()
        start = time.perf_counter()
        prompt = self._format_prompt(query, documents, instruction)
        embeddings = self._get_hidden_states(prompt)
        tokens = self._tokenize(prompt)
        scores = self._score_embeddings(embeddings, tokens)

        results = [
            RerankResult(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]

        LOGGER.debug(
            f"Reranked {len(documents)} candidate chunks to {len(results)} "
            f"chunks in {time.perf_counter() - start:.2f}s"
        )
        return results

    def _ensure_ready(self):
        required_paths = [
            self.model_path,
            self.projector_path,
            self.llama_embedding_path,
            self.llama_tokenize_path,
        ]
        missing_paths = [item for item in required_paths if not os.path.exists(item)]
        if missing_paths:
            raise FileNotFoundError(
                "Reranker required files are missing: " + ", ".join(missing_paths)
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
            self.llama_embedding_path,
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
            str(self.ubatch_size),
            "--ctx-size",
            str(self.context_size),
            "-ngl",
            str(self.gpu_layers),
        ]
        if self.flash_attention:
            command.append("--flash-attn")

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
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
            result = subprocess.run(
                [
                    self.llama_tokenize_path,
                    "-m",
                    self.model_path,
                    "-f",
                    prompt_file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
            tokens = []
            for line in result.stdout.strip().splitlines():
                if "->" in line:
                    tokens.append(int(line.split("->", 1)[0].strip()))
            return tokens
        finally:
            os.unlink(prompt_file_path)

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
