import json
import os
from typing import List, Tuple
from ollama import AsyncClient, EmbedResponse
from utils.logger import get_logger
from services.power_meter_service import PowerMeasurement, PowerMeterService

DEFAULT_MODEL = os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")[0]
LOGGER = get_logger(__name__)


def _json_mapping(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{name} must contain a valid JSON object") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def configured_embedding_models() -> list[str]:
    models = [
        item.strip()
        for item in os.getenv("EMBEDDING_MODELS", "bge-m3").split(",")
        if item.strip()
    ]
    models.extend(_json_mapping("EMBEDDING_MODELS_BY_DATASET").values())
    return list(dict.fromkeys(models))


def _with_prefix(text: str, prefix: str) -> str:
    if not prefix or text.startswith(prefix):
        return text
    return f"{prefix}{text}"


class EmbeddingService:
    def __init__(self):
        self.indices = {}
        self.model = DEFAULT_MODEL
        self.models_by_dataset = _json_mapping("EMBEDDING_MODELS_BY_DATASET")
        self.query_prefixes = _json_mapping("EMBEDDING_QUERY_PREFIXES")
        self.passage_prefixes = _json_mapping("EMBEDDING_PASSAGE_PREFIXES")
        self.client = AsyncClient(host=os.getenv("OLLAMA_EMBEDDING_HOST"))

    def model_for_dataset(self, dataset_id: str | None) -> str:
        return self.models_by_dataset.get(str(dataset_id or ""), self.model)

    def query_prefix_for_model(self, model: str) -> str:
        if model in self.query_prefixes:
            return self.query_prefixes[model]
        return "query: " if "multilingual-e5" in model else ""

    def query_input_for_model(self, text: str, model: str) -> str:
        return _with_prefix(text, self.query_prefix_for_model(model))

    def passage_prefix_for_model(self, model: str) -> str:
        if model in self.passage_prefixes:
            return self.passage_prefixes[model]
        return "passage: " if "multilingual-e5" in model else ""

    async def generate_embedding(
        self, input: str, dataset_id: str | None = None, model: str | None = None
    ) -> EmbedResponse:
        selected_model = model or self.model_for_dataset(dataset_id)
        query_input = self.query_input_for_model(input, selected_model)
        response = await self.client.embed(
            model=selected_model, input=query_input
        )

        return response

    async def generate_embeddings_batch(
        self,
        input: List[str],
        batch_size: int = 10,
        dataset_id: str | None = None,
        model: str | None = None,
    ) -> dict:
        meter = PowerMeterService()
        meter.start()
        power_samples = []
        selected_model = model or self.model_for_dataset(dataset_id)
        passage_prefix = self.passage_prefix_for_model(selected_model)
        prefixed_input = [_with_prefix(text, passage_prefix) for text in input]
        batches = [
            prefixed_input[i : i + batch_size]
            for i in range(0, len(prefixed_input), batch_size)
        ]
        ollama_duration = 0
        all_embeddings = []
        for index, batch in enumerate(batches):
            power_samples.append(meter.sample_power())
            LOGGER.debug(f"Processing embeddings batch <{index+1}> of <{len(batches)}>")
            response = await self.client.embed(model=selected_model, input=batch)
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
