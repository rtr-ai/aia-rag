import { PowerDataDisplayed, missingPower, requestPowerTotal, validPowerValue } from './models';

function row(name: string, duration: number, version = 2): PowerDataDisplayed {
  return { name, label: name, cpu_kWh: 1, gpu_kWh: 2, ram_kWh: 3, total_kWh: 6, duration,
    measurement_version: version, status: 'completed' };
}

describe('Request phase totals', () => {
  it('sums exclusive phases and excludes startup indexing', () => {
    const total = requestPowerTotal([row('power_index', 100), row('power_prompt', 2), row('power_rerank', 3), row('power_response', 5)]);
    expect(total.duration).toBe(10);
    expect(total.total_kWh).toBe(18);
    expect(total.cpu_kWh).toBe(3);
  });
  it('does not add a rerank breakdown twice for a legacy combined prompt', () => {
    const total = requestPowerTotal([row('power_prompt', 7, 1), row('power_response', 5, 1), row('power_rerank', 3)]);
    expect(total.duration).toBe(12);
    expect(total.total_kWh).toBe(12);
  });
  it('keeps absent phase and hardware values unavailable', () => {
    let total = requestPowerTotal([row('power_prompt', 2), row('power_response', 5)]);
    expect(total.duration).toBeNull();
    const rerank = row('power_rerank', 3);
    rerank.gpu_kWh = null;
    rerank.total_kWh = null;
    total = requestPowerTotal([row('power_prompt', 2), rerank, row('power_response', 5)]);
    expect(total.gpu_kWh).toBeNull();
    expect(total.total_kWh).toBeNull();
    expect(total.duration).toBe(10);
  });
  it('rejects invalid measurements and starts unavailable, not zero', () => {
    for (const value of [undefined, null, true, NaN, Infinity, -1, '1']) expect(validPowerValue(value)).toBeNull();
    expect(validPowerValue(0)).toBe(0);
    expect(missingPower().duration).toBeNull();
  });
  it('uses the latest event rather than double counting repeated events', () => {
    const total = requestPowerTotal([row('power_prompt', 100), row('power_prompt', 2), row('power_rerank', 3), row('power_response', 5)]);
    expect(total.duration).toBe(10);
  });
});
