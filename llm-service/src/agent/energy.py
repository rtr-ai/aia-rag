from typing import Iterable, Optional

from agent.llm_provider import LLM_PROVIDER
from models.agent_events import Energy
from services.rag_pipeline import PhaseEnergy, TimingSource

# the power meter only sees this host, so LLM calls are measurable only when the
# model runs locally
LLM_ENERGY_MEASURABLE = LLM_PROVIDER == "ollama"


def to_energy(
    phase: PhaseEnergy, timing_source: TimingSource, contended: bool
) -> Energy:
    return Energy(
        **phase.model_dump(), timing_source=timing_source, contended=contended
    )


def sum_energy(parts: Iterable[Optional[Energy]]) -> Optional[Energy]:
    measured = [part for part in parts if part is not None]
    if not measured:
        return None
    return Energy(
        cpu_kWh=sum(part.cpu_kWh for part in measured),
        gpu_kWh=sum(part.gpu_kWh for part in measured),
        ram_kWh=sum(part.ram_kWh for part in measured),
        total_kWh=sum(part.total_kWh for part in measured),
        duration=sum(part.duration for part in measured),
        timing_source=(
            "provider"
            if all(part.timing_source == "provider" for part in measured)
            else "wall_clock"
        ),
        contended=any(part.contended for part in measured),
    )
