import os
from typing import List, Tuple
from ollama import AsyncClient, EmbedResponse
from utils.logger import get_logger
from services.power_meter_service import PowerMeasurement, PowerMeterService

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]
LOGGER = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.indices = {}
        self.client = AsyncClient(host=os.getenv("OLLAMA_EMBEDDING_HOST"))

    async def generate_embedding(self, input: str) -> EmbedResponse:
        query_prefix = ""
        if "multilingual-e5" in DEFAULT_MODEL:
            query_prefix = "query: "
        response = await self.client.embed(
            model=DEFAULT_MODEL, input=f"{query_prefix}{input}"
        )

        return response

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
        ollama_duration = 0
        all_embeddings = []
        for index, batch in enumerate(batches):
            power_samples.append(meter.sample_power())
            LOGGER.debug(f"Processing embeddings batch <{index+1}> of <{len(batches)}>")
            response = await self.client.embed(model=DEFAULT_MODEL, input=batch)
            if "total_duration" in response:
                ollama_duration += response["total_duration"] / 1_000_000_000

            all_embeddings.extend(response["embeddings"])
        measurement = meter.stop()
        median_measurement = meter.get_median_power(power_samples)

        duration = (
            ollama_duration if ollama_duration > 0 else measurement.duration_seconds
        )

        LOGGER.debug(f"Final response: {response}")
        LOGGER.debug(
            f"Generating vector index: Median Power consumption over {duration:.2f} seconds:"
        )
        LOGGER.debug(
            f"CPU: {(median_measurement.cpu_watts * duration / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"GPU: {(median_measurement.gpu_watts * duration / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"RAM: {(median_measurement.ram_watts * duration / 3600 / 1000):.8f} kWh"
        )
        LOGGER.debug(
            f"Total for generating response: {(median_measurement.total_watts * duration / 3600 / 1000):.8f} kWh"
        )
        adjusted_measurement = PowerMeasurement(
            cpu_watts=measurement.cpu_watts,
            gpu_watts=measurement.gpu_watts,
            ram_watts=measurement.ram_watts,
            duration_seconds=duration,
        )
        meter.save_initial_power_consumption_data(
            median_measurement=median_measurement, measurement=adjusted_measurement
        )
        return all_embeddings
