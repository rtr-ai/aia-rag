from typing import List, Dict, Tuple
import json
import os
from fastapi import HTTPException
import numpy as np
from services.embedding_service import EmbeddingService
from services.tokenizer_service import TokenizerService
from models.manual_index import ChunkNode, ManualIndex
from utils.logger import get_logger
from models.sources import RelevantChunk, Source
from utils import path_utils

LOGGER = get_logger(__name__)
STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
TOP_N_CHUNKS = int(os.getenv("TOP_N_CHUNKS", "15"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8000"))
PROMPT_BUFFER = int(os.getenv("PROMPT_BUFFER", "1500"))


class IndexService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the IndexService.

        """
        if self._initialized:
            return
        LOGGER.debug("Creating new instance of IndexService")
        # Dict mapping dataset_id -> vector data
        self.vector_store: Dict[str, dict] = {}
        self.embedding_service = EmbeddingService()
        self.tokenizer_service = TokenizerService()
        self.storage_path = STORAGE_PATH
        self.vector_store_path = os.path.join(self.storage_path, "vector_store.json")
        os.makedirs(self.storage_path, exist_ok=True)

        if os.path.exists(self.vector_store_path):
            with open(self.vector_store_path, "r") as f:
                self.vector_store = json.load(f)
        self._initialized = True

    async def create_index_for_dataset(
        self, dataset_id: str, chunks_path: str, force_recreate: bool = False
    ):
        """
        Create a vector store index for a specific dataset.

        Args:
            dataset_id: Unique identifier for the dataset
            chunks_path: Path to the chunks.json file for this dataset
            force_recreate: If True, recreate even if index exists
        """
        LOGGER.debug(f"Creating index for dataset <{dataset_id}> from <{chunks_path}>")

        if dataset_id in self.vector_store and not force_recreate:
            LOGGER.info(f"Index for dataset <{dataset_id}> already exists, skipping")
            return

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as file:
            manual_index_json = json.load(file)
            manual_index = ManualIndex.model_validate(manual_index_json)
            manual_index.id = dataset_id  # Use dataset_id as the index id

        await self._create_index(manual_index)
        LOGGER.info(f"Successfully created index for dataset <{dataset_id}>")

    async def _create_index(self, manual_index: ManualIndex):
        """
        Internal method to create and save an index from a ManualIndex object.
        """
        LOGGER.debug(f"Creating index with id <{manual_index.id}>")

        chunk_nodes = [
            chunk
            for chunk in manual_index.chunks
            if isinstance(chunk, ChunkNode) and chunk.content.strip()
        ]

        LOGGER.debug(f"Generating embeddings for <{len(chunk_nodes)}> chunks")
        embeddings = await self.embedding_service.generate_embeddings_batch(
            [chunk.content for chunk in chunk_nodes]
        )

        valid_chunk_ids = {chunk.id for chunk in chunk_nodes}
        vector_data = {
            "id": manual_index.id,
            "creation_date": manual_index.creation_date,
            "last_updated": manual_index.last_updated,
            "chunks": [
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "keywords": chunk.keywords,
                    "content": chunk.content,
                    "negativeKeywords": chunk.negativeKeywords,
                    "relevantChunksIds": [
                        cid for cid in chunk.relevantChunksIds if cid in valid_chunk_ids
                    ],
                    "parameters": chunk.parameters,
                    "vector": embedding,
                    "position": i,
                }
                for i, (chunk, embedding) in enumerate(zip(chunk_nodes, embeddings))
            ],
        }

        self.vector_store[manual_index.id] = vector_data
        self._save_vector_store()

    def list_datasets(self) -> List[str]:
        """Return list of all indexed dataset IDs."""
        return list(self.vector_store.keys())

    def has_dataset(self, dataset_id: str) -> bool:
        """Check if a dataset has been indexed."""
        return dataset_id in self.vector_store

    @staticmethod
    def cosine_similarity(query_vector, vectors):
        """
        Calculate cosine similarity between a query vector and a set of vectors.
        """
        query_vector = np.array(query_vector)
        vectors = np.array(vectors)
        return np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )

    def get_top_chunks(self, query_vector, chunks_vector, top_n=TOP_N_CHUNKS):
        """
        Get top-n chunks based on cosine similarity to a query vector.

        Args:
            query_vector (list): The embedding of the query.
            chunks_vector (list): List of chunks with embeddings.
            top_n (int): Number of top results to return.

        Returns:
            list: Top-n chunks with similarity scores.
        """
        vectors = [item["vector"] for item in chunks_vector]
        similarities = self.cosine_similarity(query_vector, vectors)
        top_n_indices = np.argsort(similarities)[-top_n:][::-1]

        def get_related_chunks(chunk):
            related_ids = chunk.get("relevantChunksIds", [])
            return [
                {
                    "id": related_id,
                    "title": next(
                        (c["title"] for c in chunks_vector if c["id"] == related_id), ""
                    ),
                    "content": next(
                        (c["content"] for c in chunks_vector if c["id"] == related_id),
                        "",
                    ),
                    "position": next(
                        (c["position"] for c in chunks_vector if c["id"] == related_id),
                        -1,
                    ),
                    "num_tokens": self.tokenizer_service.count_tokens(
                        next(
                            (
                                c["content"]
                                for c in chunks_vector
                                if c["id"] == related_id
                            ),
                            "",
                        )
                    ),
                }
                for related_id in related_ids
            ]

        return [
            {
                **chunks_vector[i],
                "score": similarities[i],
                "id": chunks_vector[i].get("id"),
                "title": chunks_vector[i].get("title"),
                "relevantChunks": get_related_chunks(chunks_vector[i]),
                "num_tokens": self.tokenizer_service.count_tokens(
                    chunks_vector[i].get("content")
                ),
            }
            for i in top_n_indices
        ]

    async def query_index(
        self, dataset_id: str, query: str, request_id: str
    ) -> Tuple[List[Source], float]:
        """
        Query a specific dataset's index.

        Args:
            dataset_id: ID of the dataset to query
            query: Query string
            request_id: Request ID for logging

        Returns:
            Tuple of (sources list, embedding duration)
        """
        if dataset_id not in self.vector_store:
            LOGGER.error(f"Dataset <{dataset_id}> not found in vector store")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_id}' not found. Available: {self.list_datasets()}",
            )

        embedding_response = await self.embedding_service.generate_embedding(query)
        duration = 0.0
        if "total_duration" in embedding_response:
            duration = embedding_response["total_duration"] / 1_000_000_000
            LOGGER.debug(f"Total duration for embedding from ollama: {str(duration)}")

        query_vector = embedding_response.embeddings[0]
        index_data = self.vector_store[dataset_id]
        top_chunks = self.get_top_chunks(query_vector, index_data["chunks"])

        total_tokens = 0
        LOGGER.debug(
            f"[{request_id}]    Prompt approximate token length: {self.tokenizer_service.count_tokens(query)}"
        )
        token_limit = CONTEXT_WINDOW - PROMPT_BUFFER
        sources: List[Source] = []
        added_chunks = set()
        context_window_reached = False

        for chunk in top_chunks:
            chunk_tokens = chunk["num_tokens"]

            if context_window_reached:
                skip = True
                skip_reason = "context_window"
            else:
                if chunk["id"] in added_chunks:
                    skip = True
                    skip_reason = "duplicate"
                elif total_tokens + chunk_tokens > token_limit:
                    skip = True
                    skip_reason = "context_window"
                    context_window_reached = True
                else:
                    skip = False
                    skip_reason = ""

            new_source = Source(
                content=chunk["content"],
                score=chunk["score"],
                title=chunk["title"],
                relevantChunks=[],
                num_tokens=chunk_tokens,
                skip=skip,
                skip_reason=skip_reason,
                position=chunk["position"],
            )

            if not skip:
                total_tokens += chunk_tokens
                added_chunks.add(chunk["id"])

                for relevant_chunk in chunk["relevantChunks"]:
                    relevant_chunk_tokens = relevant_chunk["num_tokens"]

                    if context_window_reached:
                        relevant_skipped = True
                        relevant_skip_reason = "context_window"
                    else:
                        if relevant_chunk["id"] in added_chunks:
                            relevant_skipped = True
                            relevant_skip_reason = "duplicate"
                        elif total_tokens + relevant_chunk_tokens > token_limit:
                            relevant_skipped = True
                            relevant_skip_reason = "context_window"
                            context_window_reached = True
                        else:
                            relevant_skipped = False
                            relevant_skip_reason = ""

                    new_relevant_chunk = RelevantChunk(
                        id=relevant_chunk["id"],
                        content=relevant_chunk["content"],
                        title=relevant_chunk.get("title") or relevant_chunk["id"],
                        num_tokens=relevant_chunk_tokens,
                        skip=relevant_skipped,
                        skip_reason=relevant_skip_reason,
                        position=relevant_chunk["position"],
                    )

                    if not relevant_skipped:
                        added_chunks.add(relevant_chunk["id"])
                        total_tokens += relevant_chunk_tokens

                    new_source.relevantChunks.append(new_relevant_chunk)

            sources.append(new_source)

        log_output_used_sources = f"[{request_id}]   Generated chunks for the query:\n"
        for source in sources:
            log_output_used_sources += f"(skip_reason:{source.skip_reason}) {source.title} ({str(source.num_tokens)} Token)\n"
            for relevant_source in source.relevantChunks:
                log_output_used_sources += f"_______(skip_reason:${relevant_source.skip_reason}) {relevant_source.title} ({str(relevant_source.num_tokens)} Token)\n"

        LOGGER.debug(log_output_used_sources)
        return sources, duration

    def _save_vector_store(self):
        """
        Save the vector store to disk.
        """
        LOGGER.debug(
            f"Saving vector store to the disk at path {self.vector_store_path}"
        )
        with open(self.vector_store_path, "w") as f:
            json.dump(self.vector_store, f)
