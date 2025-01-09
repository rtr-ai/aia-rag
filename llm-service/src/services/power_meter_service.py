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
        self.logger = get_logger(__name__)
        self.storage_path = os.path.join(STORAGE_PATH, "index_power_consumption.json")
        
        try:
            nvml.nvmlInit()
            self.gpu_available = True
            self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            self.logger.warning(f"GPU monitoring not available: {e}")
            self.gpu_available = False
        
        self._start_time: Optional[float] = None
        self._start_cpu_energy = None
        self._start_gpu_energy = None
        self._start_ram_usage = None
        
        # RAM power constants (based on DDR4 average consumption)
        self.RAM_POWER_FACTOR = 0.375  # Watts per GB

    def save_initial_power_consumption_data(self, median_measurement, measurement):
        data = {
            "cpu_kWh": (median_measurement.cpu_watts * measurement.duration_seconds / 3600 / 1000),
            "gpu_kWh": (median_measurement.gpu_watts * measurement.duration_seconds / 3600 / 1000),
            "ram_kWh": (median_measurement.ram_watts * measurement.duration_seconds / 3600 / 1000),
            "total_kWh": (median_measurement.total_watts * measurement.duration_seconds / 3600 / 1000),
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_power_consumption(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            return {"error": "Power consumption data not available"}
        
    def sample_power(self) -> PowerMeasurement:
        current_cpu_energy = self._get_cpu_energy()
        current_gpu_energy = self._get_gpu_energy()
        current_ram_usage = self._get_ram_usage()

        cpu_watts = max(0, (current_cpu_energy - self._start_cpu_energy)) / (time.time() - self._start_time)
        gpu_watts = max(0, (current_gpu_energy - self._start_gpu_energy)) / (time.time() - self._start_time)
        avg_ram_usage = (self._start_ram_usage + current_ram_usage) / 2
        ram_watts = avg_ram_usage * self.RAM_POWER_FACTOR

        return PowerMeasurement(cpu_watts, gpu_watts, ram_watts, 0)  # Duration can be 0 for snapshot
    
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
                    with open(f'/sys/class/powercap/intel-rapl:{socket_count}/energy_uj', 'r') as f:
                        total_energy += int(f.read())
                    try:
                        with open(f'/sys/class/powercap/intel-rapl:{socket_count}:0/energy_uj', 'r') as f:
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
            return nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # Convert mJ to joules
        except nvml.NVMLError as e:
            self.logger.warning(f"Failed to read GPU energy: {e}")
            try:
                return nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert mW to W
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
            duration_seconds=duration
        )
    
    def __del__(self):
        if self.gpu_available:
            try:
                nvml.nvmlShutdown()
            except:
                pass
