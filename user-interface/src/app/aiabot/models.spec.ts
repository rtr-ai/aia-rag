import { PowerDataDisplayed, missingPower, requestPowerTotal, validPowerValue } from './models';

function row(name: string, duration: number, version = 2): PowerDataDisplayed {
  return { name, label: name, cpu_kWh: 1, gpu_kWh: 2, ram_kWh: 3, total_kWh: 6, duration,
    measurement_version: version, status: 'completed' };
}

describe('Public chatbot totals including startup indexing', () => {
  it('sums indexing and exclusive request phases', () => {
    const total = requestPowerTotal([row('power_index', 100), row('power_prompt', 2), row('power_rerank', 3), row('power_response', 5)]);
    expect(total.duration).toBe(110);
    expect(total.total_kWh).toBe(24);
    expect(total.cpu_kWh).toBe(4);
    expect(total.gpu_kWh).toBe(8);
    expect(total.ram_kWh).toBe(12);
  });
  it('includes indexing but does not add a rerank breakdown twice for a legacy combined prompt', () => {
    const total = requestPowerTotal([row('power_index', 100, 1), row('power_prompt', 7, 1), row('power_response', 5, 1), row('power_rerank', 3)]);
    expect(total.duration).toBe(112);
    expect(total.total_kWh).toBe(18);
  });
  it('keeps absent phase and hardware values unavailable', () => {
    let total = requestPowerTotal([row('power_index', 100), row('power_prompt', 2), row('power_response', 5)]);
    expect(total.duration).toBeNull();
    const rerank = row('power_rerank', 3);
    rerank.gpu_kWh = null;
    rerank.total_kWh = null;
    total = requestPowerTotal([row('power_index', 100), row('power_prompt', 2), rerank, row('power_response', 5)]);
    expect(total.gpu_kWh).toBeNull();
    expect(total.total_kWh).toBeNull();
    expect(total.duration).toBe(110);
  });
  it('does not invent zero indexing when its measurement is absent or partial', () => {
    const phases = [row('power_prompt', 2), row('power_rerank', 3), row('power_response', 5)];
    expect(requestPowerTotal(phases).duration).toBeNull();
    const index = row('power_index', 100);
    index.cpu_kWh = null;
    index.total_kWh = null;
    const total = requestPowerTotal([index, ...phases]);
    expect(total.duration).toBe(110);
    expect(total.cpu_kWh).toBeNull();
    expect(total.total_kWh).toBeNull();
    expect(total.gpu_kWh).toBe(8);
  });
  it('adds no cost for reranking and generation explicitly not run', () => {
    const notRun = (name: string): PowerDataDisplayed => ({...row(name, 0),
      cpu_kWh: 0, gpu_kWh: 0, ram_kWh: 0, total_kWh: 0, status: 'not_run'});
    const total = requestPowerTotal([row('power_index', 100), row('power_prompt', 2), notRun('power_rerank'), notRun('power_response')]);
    expect(total.duration).toBe(102);
    expect(total.total_kWh).toBe(12);
  });
  it('rejects invalid measurements and starts unavailable, not zero', () => {
    for (const value of [undefined, null, true, NaN, Infinity, -1, '1']) expect(validPowerValue(value)).toBeNull();
    expect(validPowerValue(0)).toBe(0);
    expect(missingPower().duration).toBeNull();
  });
  it('uses the latest event rather than double counting repeated indexing or phases', () => {
    const total = requestPowerTotal([row('power_index', 200), row('power_index', 100), row('power_prompt', 100), row('power_prompt', 2), row('power_rerank', 3), row('power_response', 5)]);
    expect(total.duration).toBe(110);
    expect(total.total_kWh).toBe(24);
  });
});
