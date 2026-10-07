import json
import os
import statistics
import time
import psutil
import pynvml as nvml
from dataclasses import dataclass
from typing import Optional
from utils.logger import get_logger
from utils import path_utils
from services.matomo_tracking_service import matomo_service


@dataclass
class PowerMeasurement:
    cpu_watts: float
    gpu_watts: float
    ram_watts: float
    duration_seconds: float

    @property
    def total_watts(self) -> float:
        return self.cpu_watts + self.gpu_watts + self.ram_watts


STORAGE_PATH = os.path.join(path_utils.get_project_root(), "data", "power")


class PowerMeterService:
    def __init__(self):
        self.gpu_available = False
        self.handle = None
        self.logger = get_logger(__name__)
        os.makedirs(STORAGE_PATH, exist_ok=True)
        self.storage_path = os.path.join(STORAGE_PATH, "index_power_consumption.json")

        try:
            nvml.nvmlInit()
            self.gpu_available = True
            self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            self.logger.warning(f"GPU monitoring not available: {e}")

        self._start_time: Optional[float] = None
        self._start_cpu_energy = None
        self._start_gpu_energy = None
        self._start_ram_usage = None

        # RAM power constants (based on DDR4 average consumption)
        self.RAM_POWER_FACTOR = 0.375  # Watts per GB

    def save_initial_power_consumption_data(self, median_measurement, measurement):
        data = {
            "cpu_kWh": (
                median_measurement.cpu_watts
                * measurement.duration_seconds
                / 3600
                / 1000
            ),
            "gpu_kWh": (
                median_measurement.gpu_watts
                * measurement.duration_seconds
                / 3600
                / 1000
            ),
            "ram_kWh": (
                median_measurement.ram_watts
                * measurement.duration_seconds
                / 3600
                / 1000
            ),
            "total_kWh": (
                median_measurement.total_watts
                * measurement.duration_seconds
                / 3600
                / 1000
            ),
            "duration": measurement.duration_seconds,
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=4)
        matomo_service.track_event(action="power_index", value=data)

    def get_initial_power_consumption(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            return data
        else:
            return {"error": "Power consumption data not available"}

    def sample_power(self) -> PowerMeasurement:
        current_cpu_energy = self._get_cpu_energy()
        current_gpu_energy = self._get_gpu_energy()
        current_ram_usage = self._get_ram_usage()

        cpu_watts = max(0, (current_cpu_energy - self._start_cpu_energy)) / (
            time.time() - self._start_time
        )
        gpu_watts = max(0, (current_gpu_energy - self._start_gpu_energy)) / (
            time.time() - self._start_time
        )
        avg_ram_usage = (self._start_ram_usage + current_ram_usage) / 2
        ram_watts = avg_ram_usage * self.RAM_POWER_FACTOR

        return PowerMeasurement(
            cpu_watts, gpu_watts, ram_watts, 0
        )  # Duration can be 0 for snapshot

    def get_median_power(self, measurements: list) -> PowerMeasurement:
        median_cpu = statistics.median([m.cpu_watts for m in measurements])
        median_gpu = statistics.median([m.gpu_watts for m in measurements])
        median_ram = statistics.median([m.ram_watts for m in measurements])
        return PowerMeasurement(median_cpu, median_gpu, median_ram, 0)

    def _get_cpu_energy(self) -> float:
        """Read CPU energy consumption from RAPL"""
        try:
            total_energy = 0
            socket_count = 0
            while True:
                try:
                    with open(
                        f"/sys/class/powercap/intel-rapl:{socket_count}/energy_uj", "r"
                    ) as f:
                        total_energy += int(f.read())
                    try:
                        with open(
                            f"/sys/class/powercap/intel-rapl:{socket_count}:0/energy_uj",
                            "r",
                        ) as f:
                            total_energy += int(f.read())
                    except FileNotFoundError:
                        pass
                    socket_count += 1
                except FileNotFoundError:
                    break
            return total_energy / 1_000_000  # Convert microjoules to joules
        except Exception as e:
            self.logger.warning(f"Failed to read CPU energy: {e}")
            return psutil.cpu_percent() * psutil.cpu_count() * 0.5  # Rough estimation

    def _get_gpu_energy(self) -> float:
        """Read GPU energy consumption"""
        if not self.gpu_available:
            return 0.0
        try:
            return (
                nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
            )  # Convert mJ to joules
        except nvml.NVMLError as e:
            self.logger.warning(f"Failed to read GPU energy: {e}")
            try:
                return (
                    nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                )  # Convert mW to W
            except:
                return 0.0

    def _get_ram_usage(self) -> float:
        """Get RAM usage in GB"""
        return psutil.virtual_memory().used / (1024 * 1024 * 1024)  # Convert to GB

    def start(self):
        """Start power measurement"""
        if self._start_time is not None:
            raise RuntimeError("Measurement already in progress")

        self._start_time = time.time()
        self._start_cpu_energy = self._get_cpu_energy()
        self._start_gpu_energy = self._get_gpu_energy()
        self._start_ram_usage = self._get_ram_usage()

    def stop(self) -> PowerMeasurement:
        """Stop measurement and return power consumption"""
        if self._start_time is None:
            raise RuntimeError("No measurement in progress")

        end_time = time.time()
        duration = end_time - self._start_time

        end_cpu_energy = self._get_cpu_energy()
        end_gpu_energy = self._get_gpu_energy()
        end_ram_usage = self._get_ram_usage()

        cpu_watts = max(0, (end_cpu_energy - self._start_cpu_energy)) / duration
        gpu_watts = max(0, (end_gpu_energy - self._start_gpu_energy)) / duration

        avg_ram_usage = (self._start_ram_usage + end_ram_usage) / 2
        ram_watts = avg_ram_usage * self.RAM_POWER_FACTOR

        self._start_time = None
        self._start_cpu_energy = None
        self._start_gpu_energy = None
        self._start_ram_usage = None

        return PowerMeasurement(
            cpu_watts=cpu_watts,
            gpu_watts=gpu_watts,
            ram_watts=ram_watts,
            duration_seconds=duration,
        )

    def phase_snapshot(self):
        """Read cumulative joules, RAM usage and a monotonic boundary clock.

        Missing counters stay unavailable; CPU packages include core subdomains.
        """
        cpu_energy = None
        try:
            energies = []
            socket = 0
            while True:
                try:
                    with open(f"/sys/class/powercap/intel-rapl:{socket}/energy_uj") as handle:
                        energies.append(int(handle.read()) / 1_000_000)
                except FileNotFoundError:
                    break
                socket += 1
            if energies:
                cpu_energy = sum(energies)
        except (OSError, ValueError):
            pass
        gpu_energy = None
        if self.gpu_available:
            try:
                gpu_energy = nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000
            except Exception:
                pass
        try:
            ram_usage = self._get_ram_usage()
        except Exception:
            ram_usage = None
        return time.perf_counter(), cpu_energy, gpu_energy, ram_usage

    def __del__(self):
        if getattr(self, "gpu_available", False):
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


class PhasePowerMeter:
    """Request-local, non-overlapping intervals; no shared start/stop state."""

    def __init__(self, meter):
        self.meter = meter
        self.active_stage = None
        self.boundary = None
        self.stages = {}

    def switch(self, stage):
        if stage == self.active_stage:
            return
        boundary = self.meter.phase_snapshot()
        self._finish_interval(boundary)
        self.active_stage = stage
        self.boundary = boundary if stage is not None else None

    def stop(self):
        if self.active_stage is not None:
            self.switch(None)

    def _finish_interval(self, end):
        if self.active_stage is None:
            return
        start = self.boundary
        duration = max(0.0, end[0] - start[0])
        row = self.stages.setdefault(self.active_stage, {
            "cpu_kWh": 0.0, "gpu_kWh": 0.0, "ram_kWh": 0.0, "duration": 0.0,
        })
        row["duration"] += duration
        for field, index in (("cpu_kWh", 1), ("gpu_kWh", 2)):
            delta = None
            if start[index] is not None and end[index] is not None:
                if end[index] >= start[index]:
                    delta = (end[index] - start[index]) / 3_600_000
            row[field] = None if row[field] is None or delta is None else row[field] + delta
        ram = None
        if start[3] is not None and end[3] is not None:
            ram = (start[3] + end[3]) / 2 * self.meter.RAM_POWER_FACTOR * duration / 3_600_000
        row["ram_kWh"] = None if row["ram_kWh"] is None or ram is None else row["ram_kWh"] + ram

    def payload(self, stage, status="completed"):
        empty_value = 0.0 if status in ("not_run", "skipped") else None
        row = dict(self.stages.get(stage, {
            "cpu_kWh": empty_value, "gpu_kWh": empty_value,
            "ram_kWh": empty_value, "duration": empty_value,
        }))
        values = [row[field] for field in ("cpu_kWh", "gpu_kWh", "ram_kWh")]
        row["total_kWh"] = sum(values) if all(value is not None for value in values) else None
        row.update({
            "measurement_version": 2,
            "status": status,
            "measurement_scope": "host_cpu_gpu0_ram; concurrent workloads may be included",
            "measurement_methods": {
                "cpu_kWh": "package_energy_counter" if row["cpu_kWh"] is not None else "unavailable",
                "gpu_kWh": "gpu0_energy_counter" if row["gpu_kWh"] is not None else "unavailable",
                "ram_kWh": "estimated_from_ram_usage" if row["ram_kWh"] is not None else "unavailable",
            },
        })
        if status in ("not_run", "skipped"):
            row["measurement_methods"] = {field: "not_run" for field in ("cpu_kWh", "gpu_kWh", "ram_kWh")}
        return row
