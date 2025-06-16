import os
from typing import List
from ollama import AsyncClient
from utils.logger import get_logger
from services.power_meter_service import PowerMeterService

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]
LOGGER = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.indices = {}
        self.client = AsyncClient(host=os.getenv("OLLAMA_EMBEDDING_HOST"))

    async def generate_embedding(self, input: str) -> dict:
        query_prefix = ""
        if "multilingual-e5" in DEFAULT_MODEL:
            query_prefix = "query: "
        response = await self.client.embed(
            model=DEFAULT_MODEL, input=f"{query_prefix}{input}"
        )

        return response["embeddings"]

    async def generate_embeddings_batch(
        self, input: List[str], batch_size: int = 10
    ) -> dict:
        meter = PowerMeterService()
        meter.start()
        power_samples = []
        passage_prefix = ""

        if "multilingual-e5" in DEFAULT_MODEL:
            passage_prefix = "passage: "
        prefixed_input = [f"{passage_prefix}{text}" for text in input]
        batches = [
            prefixed_input[i : i + batch_size]
            for i in range(0, len(prefixed_input), batch_size)
        ]
        all_embeddings = []
        for index, batch in enumerate(batches):
            power_samples.append(meter.sample_power())
            LOGGER.debug(f"Processing embeddings batch <{index+1}> of <{len(batches)}>")
            response = await self.client.embed(model=DEFAULT_MODEL, input=batch)
            all_embeddings.extend(response["embeddings"])
        measurement = meter.stop()
        median_measurement = meter.get_median_power(power_samples)
        LOGGER.debug(f"Final response: {response}")
        LOGGER.debug(
            f"Generating vector index: Median Power consumption over {measurement.duration_seconds:.2f} seconds:"
        )
        LOGGER.debug(
            f"CPU: {(median_measurement.cpu_watts * measurement.duration_seconds / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"GPU: {(median_measurement.gpu_watts * measurement.duration_seconds / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"RAM: {(median_measurement.ram_watts * measurement.duration_seconds / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"Total for generating response: {(median_measurement.total_watts * measurement.duration_seconds / 3600 / 1000):.8f} kWh"
        )
        meter.save_initial_power_consumption_data(
            median_measurement=median_measurement, measurement=measurement
        )
        return all_embeddings
