import time
from typing import AsyncIterator, List, Optional, Union

from pydantic import BaseModel

from agent import llm_calls
from agent.llm_client import (
    ContentChunk,
    LlmBadResponse,
    LlmClient,
    Message,
    ReasoningChunk,
    TurnResult,
)
from models.sources import Source
from services.index_service import IndexService
from services.power_meter_service import PowerMeasurement, PowerMeterService
from utils.logger import get_logger
from utils.prompt_utils import render_prompt

LOGGER = get_logger(__name__)


class PhaseEnergy(BaseModel):
    cpu_kWh: float
    gpu_kWh: float
    ram_kWh: float
    total_kWh: float
    duration: float


class SourcesRetrieved(BaseModel):
    sources: List[Source]
    energy: PhaseEnergy


class PromptBuilt(BaseModel):
    messages: List[Message]


class GenerationFinished(BaseModel):
    result: TurnResult
    duration: float
    energy: PhaseEnergy


RagEvent = Union[
    SourcesRetrieved, PromptBuilt, ReasoningChunk, ContentChunk, GenerationFinished
]


def _phase_energy(measurement: PowerMeasurement, duration: float) -> PhaseEnergy:
    return PhaseEnergy(
        cpu_kWh=(measurement.cpu_watts * duration / 3600 / 1000),
        gpu_kWh=(measurement.gpu_watts * duration / 3600 / 1000),
        ram_kWh=(measurement.ram_watts * duration / 3600 / 1000),
        total_kWh=(measurement.total_watts * duration / 3600 / 1000),
        duration=duration,
    )


class RagPipeline:
    def __init__(self, index_service: IndexService, llm_client: LlmClient):
        self.index_service = index_service
        self.llm_client = llm_client

    async def run(
        self,
        dataset_id: str,
        query: str,
        prompt_template: str,
        request_id: str,
        system_prompt: Optional[str] = None,
        token_limit: Optional[int] = None,
    ) -> AsyncIterator[RagEvent]:
        LOGGER.debug(f"[{request_id}]   Prompting: <{query}>")
        meter = PowerMeterService()
        meter.start()

        chunks, duration = await self.index_service.query_index(
            dataset_id=dataset_id,
            query=query,
            request_id=request_id,
            token_limit=token_limit,
        )
        measurement = meter.stop()
        final_duration = duration if duration else measurement.duration_seconds
        yield SourcesRetrieved(
            sources=chunks,
            energy=_phase_energy(measurement, final_duration),
        )

        prompt = render_prompt(prompt_template, chunks, query)
        messages: List[Message] = []
        if system_prompt is not None:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        yield PromptBuilt(messages=messages)

        meter.start()
        power_samples = []
        turn_result: Optional[TurnResult] = None
        stream_started = time.monotonic()

        with llm_calls.track():
            async for delta in self.llm_client.stream_turn(messages, []):
                power_samples.append(meter.sample_power())
                if isinstance(delta, (ReasoningChunk, ContentChunk)):
                    yield delta
                elif isinstance(delta, TurnResult):
                    turn_result = delta
        wall_duration = time.monotonic() - stream_started

        if turn_result is None:
            raise LlmBadResponse("LLM stream ended without a turn result")

        measurement = meter.stop()
        timings = turn_result.timings
        provider_durations = [
            timings.load_duration,
            timings.prompt_eval_duration,
            timings.eval_duration,
        ]
        known = [d for d in provider_durations if d is not None]
        provider_duration = sum(known) if len(known) == len(provider_durations) else 0.0
        final_duration = provider_duration or measurement.duration_seconds
        median_measurement = meter.get_median_power(power_samples)
        yield GenerationFinished(
            result=turn_result,
            duration=wall_duration,
            energy=_phase_energy(median_measurement, final_duration),
        )
