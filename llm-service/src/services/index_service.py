from typing import List, Dict
import json
import os
from fastapi import HTTPException
import numpy as np
from services.embedding_service import EmbeddingService
from services.tokenizer_service import TokenizerService
from services.power_meter_service import PowerMeterService
from models.manual_index import ChunkNode, ManualIndex
from utils.logger import get_logger
from models.sources import RelevantChunk, Source
from utils import path_utils
LOGGER = get_logger(__name__)
STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "indices")
TOP_N_CHUNKS = int(os.getenv("TOP_N_CHUNKS", "15"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8000"))


class IndexService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IndexService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the IndexService.

        """
        if self._initialized:
            return
        LOGGER.debug("Creating new instance of IndexService")
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

    @classmethod
    async def from_manual_annotations(
        cls,
        annotation: ManualIndex,
    ):
        """
        Initialize IndexService and create index from manual annotations.

        Args:
            annotation (ManualIndex): ManualIndex object containing index details and chunks.

        Returns:
            IndexService: An initialized IndexService instance.
        """
        LOGGER.debug("Creating manual index")
        service = cls()
        await service.clear_index()
        await service.create_index(annotation)
        return service

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

    async def clear_index(self):
        if os.path.exists(self.vector_store_path):
            os.remove(self.vector_store_path)

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
                        (c["content"] for c in chunks_vector if c["id"] == related_id), ""
                    ),
                    "num_tokens": self.tokenizer_service.count_tokens(next((c["content"] for c in chunks_vector if c["id"] == related_id), "")),
                }
                for related_id in related_ids
            ]
        return [
            {
                **chunks_vector[i],
                "score": similarities[i],
                "title": chunks_vector[i].get("title"),
                "relevantChunks": get_related_chunks(chunks_vector[i]),
                "num_tokens":self.tokenizer_service.count_tokens(chunks_vector[i].get("content"))
            }
            for i in top_n_indices
        ]

    async def create_index(self, manual_index: ManualIndex):
        """
        Create and save an index from a ManualIndex object.

        Args:
            manual_index (ManualIndex): Object containing chunk data.

        Raises:
            ValueError: If the index already exists in the vector store.
        """
        LOGGER.debug(f"Creating a new index with id <{manual_index.id}>")
        if manual_index.id in self.vector_store:
            LOGGER.error("Index already exists")
            raise ValueError(f"Index with ID {manual_index.id} already exists.")

        chunk_nodes = [
            chunk for chunk in manual_index.chunks if isinstance(chunk, ChunkNode)
        ]

        LOGGER.debug(f"Generating embedding for <{len(chunk_nodes)}> chunks")

        meter = PowerMeterService()
        meter.start()
        embeddings = await self.embedding_service.generate_embeddings_batch(
            [chunk.content for chunk in chunk_nodes]
        )
        measurement = meter.stop()
            
        LOGGER.debug(f"Power consumption over {measurement.duration_seconds:.2f} seconds:")
        LOGGER.debug(f"CPU: {measurement.cpu_watts:.2f} W")
        LOGGER.debug(f"GPU: {measurement.gpu_watts:.2f} W")
        LOGGER.debug(f"RAM: {measurement.ram_watts:.2f} W")
        LOGGER.debug(f"Total for generating initial index: {measurement.total_watts:.2f} W")

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
                    "relevantChunksIds": chunk.relevantChunksIds,
                    "parameters": chunk.parameters,
                    "vector": embedding,
                }
                for chunk, embedding in zip(chunk_nodes, embeddings)
            ],
        }

        # Save to vector store
        self.vector_store[manual_index.id] = vector_data
        self._save_vector_store()

    async def query_index(
        self,
        index_id: str,
        query: str,
    ) -> List[Source]:
        """
        Query an index and return all chunks, marking those exceeding token limits.

        Args:
            index_id (str): ID of the index to query.
            query (str): Query string.

        Returns:
            List[Source]: A list of all sources with skip property set based on token limit.
        """
        if index_id not in self.vector_store:
            LOGGER.debug(f"Unable to find index with id <{index_id}>")
            raise HTTPException(status_code=404, detail="Index not found")

        query_vector = (await self.embedding_service.generate_embedding(query))[0]
        index_data = self.vector_store[index_id]
        top_chunks = self.get_top_chunks(query_vector, index_data["chunks"])
        total_tokens = 0
        token_limit = CONTEXT_WINDOW - 1500
        sources: List[Source] = []

        for chunk in top_chunks:
            chunk_tokens = chunk["num_tokens"]
            skip = total_tokens + chunk_tokens > token_limit

            new_source = Source(
                content=chunk["content"],
                score=chunk["score"],
                title=chunk["title"],
                relevantChunks=[],
                num_tokens=chunk_tokens,
                skip=skip,
            )

            if not skip:
                total_tokens += chunk_tokens

            for relevant_chunk in chunk["relevantChunks"]:
                relevant_chunk_tokens = relevant_chunk["num_tokens"]
                relevant_skipped = total_tokens + relevant_chunk_tokens > token_limit

                new_relevant_chunk = RelevantChunk(
                    id=relevant_chunk["id"],
                    content=relevant_chunk["content"],
                    title=relevant_chunk.get("title") or relevant_chunk["id"],
                    num_tokens=relevant_chunk_tokens,
                    skip=relevant_skipped,
                )

                if not relevant_skipped:
                    new_source.relevantChunks.append(new_relevant_chunk)
                    total_tokens += relevant_chunk_tokens

            sources.append(new_source)

        LOGGER.debug(f"Generated chunks for the query {sources}")
        return sources

    def _save_vector_store(self):
        """
        Save the vector store to disk.
        """
        LOGGER.debug(
            f"Saving vector store to the disk at path {self.vector_store_path}"
        )
        with open(self.vector_store_path, "w") as f:
            json.dump(self.vector_store, f)
