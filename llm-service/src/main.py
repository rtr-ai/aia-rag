import asyncio
import os
from fastapi import FastAPI
from httpx import AsyncClient
from utils.logger import get_logger
from contextlib import asynccontextmanager
from api.router import router as api_router
from services.index_service import IndexService
from fastapi.middleware.cors import CORSMiddleware
from services.dataset_configuration import DatasetConfiguration

LOGGER = get_logger(__name__)

EMBEDDING_MODELS = os.getenv("EMBEDDING_MODELS", "").split(",")
LLM_MODELS = os.getenv("LLM_MODELS", "").split(",")
ROOT_PATH = os.getenv("ROOT_PATH", "")
DATA_DIR = "/app/data"


async def is_model_available(client: AsyncClient, model_name: str) -> bool:
    response = await client.post("/show", json={"model": model_name})
    return response.status_code == 200


async def pull_model(client: AsyncClient, model_name: str):
    response = await client.post("/pull", json={"model": model_name}, timeout=2400)
    if response.status_code != 200:
        raise Exception(f"Failed to pull model {model_name}: {response.text}")


async def create_vector_stores(
    index_service: IndexService, dataset_config: DatasetConfiguration
):
    """
    Create vector store indices for all configured datasets.
    """
    LOGGER.info("Creating vector stores for all datasets...")

    for dataset_id, chunks_filename in dataset_config.datasets.items():
        chunks_path = os.path.join(DATA_DIR, chunks_filename)
        LOGGER.info(f"Indexing dataset <{dataset_id}> from <{chunks_path}>")

        try:
            await index_service.create_index_for_dataset(
                dataset_id=dataset_id,
                chunks_path=chunks_path,
                force_recreate=True,
            )
        except FileNotFoundError as e:
            LOGGER.error(f"Failed to index dataset <{dataset_id}>: {e}")
            raise RuntimeError(f"Dataset file not found: {chunks_path}")
        except Exception as e:
            LOGGER.error(f"Failed to index dataset <{dataset_id}>: {e}")
            raise

    LOGGER.info(
        f"Successfully indexed {len(dataset_config.datasets)} datasets: "
        f"{list(dataset_config.datasets.keys())}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama_host = os.getenv("OLLAMA_HOST")
    ollama_embedding_host = os.getenv("OLLAMA_EMBEDDING_HOST")

    if not ollama_host:
        LOGGER.error("OLLAMA_HOST environment variable is required")
        raise RuntimeError("OLLAMA_HOST environment variable is required")

    if not ollama_embedding_host:
        LOGGER.error("OLLAMA_EMBEDDING_HOST environment variable is required")
        raise RuntimeError("OLLAMA_EMBEDDING_HOST environment variable is required")

    LOGGER.debug(f"Ollama host: {ollama_host}")
    LOGGER.debug(f"Ollama embedding host: {ollama_embedding_host}")

    # Load dataset configuration
    try:
        dataset_config = DatasetConfiguration()
        app.state.dataset_config = dataset_config
    except Exception as e:
        LOGGER.error(f"Unable to load configuration: {e}")
        raise RuntimeError(f"Unable to load configuration: {e}")

    index_service = IndexService()
    app.state.index_service = index_service

    client = AsyncClient(base_url=f"http://{ollama_host}:11434/api")
    embedding_client = AsyncClient(base_url=f"http://{ollama_embedding_host}:11434/api")

    if not EMBEDDING_MODELS or EMBEDDING_MODELS == [""]:
        raise RuntimeError("At least one embedding model has to be configured")
    if not LLM_MODELS or LLM_MODELS == [""]:
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

        await create_vector_stores(index_service, dataset_config)
        yield
    finally:
        await client.aclose()
        await embedding_client.aclose()


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
