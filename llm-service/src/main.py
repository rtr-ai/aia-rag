import asyncio
import json
import os
from fastapi import FastAPI
from httpx import AsyncClient
from utils.logger import get_logger
from contextlib import asynccontextmanager
from api.router import router as api_router
from services.index_service import IndexService
from models.manual_index import ManualIndex
from fastapi.middleware.cors import CORSMiddleware

LOGGER = get_logger(__name__)

EMBEDDING_MODELS = os.getenv("EMBEDDING_MODELS", "").split(",")
LLM_MODELS = os.getenv("LLM_MODELS", "").split(",")
ROOT_PATH = os.getenv("ROOT_PATH", "")


async def is_model_available(client: AsyncClient, model_name: str) -> bool:
    response = await client.post("/show", json={"model": model_name})
    return response.status_code == 200


async def pull_model(client: AsyncClient, model_name: str):
    response = await client.post("/pull", json={"model": model_name}, timeout=2400)
    if response.status_code != 200:
        raise Exception(f"Failed to pull model {model_name}: {response.text}")


async def create_vector_store(path: str):
    LOGGER.debug(f"Creating a vectore store index from {path}")
    with open(path, "r", encoding="utf-8") as file:
        manual_index_json = json.load(file)
        manual_index = ManualIndex.model_validate(manual_index_json)
        manual_index.id = "main"

    await IndexService.from_manual_annotations(manual_index)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_embedding_host = os.getenv("OLLAMA_EMBEDDING_HOST")

    if ollama_host:
        LOGGER.debug(f"Ollama host is: {ollama_host}")
    else:
        LOGGER.error("Ollama must be configured using OLLAMA_HOST environment")
        raise RuntimeError("Ollama must be configured using OLLAMA_HOST environment")

    if ollama_embedding_host:
        LOGGER.debug(f"Ollama embedding host is: {ollama_embedding_host}")
    else:
        LOGGER.error(
            "Ollama for embedding must be configured using OLLAMA_EMBEDDING_HOST environment"
        )
        raise RuntimeError(
            "Ollama for embedding must be configured using OLLAMA_EMBEDDING_HOST environment"
        )

    annotations_location = "/app/data/chunks.json"
    client = AsyncClient(base_url=f"http://{ollama_host}:11434/api")
    embedding_client = AsyncClient(base_url=f"http://{ollama_embedding_host}:11434/api")
    if not EMBEDDING_MODELS:
        raise RuntimeError("At least one embedding model has to be configured")
    if not LLM_MODELS:
        raise RuntimeError("At least one LLM model has to be configured")
    try:
        LOGGER.debug("Waiting 15 seconds to ensure all services are available...")
        await asyncio.sleep(15)
        LOGGER.debug("Checking embedding models...")
        for model in EMBEDDING_MODELS:
            if not await is_model_available(embedding_client, model):
                LOGGER.debug(f"Embedding Model {model} not found. Pulling...")
                await pull_model(embedding_client, model)
                LOGGER.debug(f"Embedding Model {model} pulled successfully.")
        LOGGER.debug("Checking models...")
        for model in LLM_MODELS:
            if not await is_model_available(client, model):
                LOGGER.debug(f"Model {model} not found. Pulling...")
                await pull_model(client, model)
                LOGGER.debug(f"Model {model} pulled successfully.")
        LOGGER.debug("All models are ready.")
        if os.path.exists(annotations_location):
            await create_vector_store(path=annotations_location)
        else:
            LOGGER.error(
                f"The annotated chunks should be accessible at path <{annotations_location}>"
            )
            raise RuntimeError(
                f"The annotated chunks should be accessible at path <{annotations_location}>"
            )
        yield
    finally:
        await client.aclose()


app = FastAPI(lifespan=lifespan, root_path=ROOT_PATH)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
