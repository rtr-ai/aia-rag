export interface LLMMessageParams {
  content: string;
  type:
    | "error"
    | "sources"
    | "user"
    | "assistant"
    | "power_prompt"
    | "power_rerank"
    | "power_index"
    | "power_response"
    | "queue_position"
    | "metadata";
}

interface RelevantSource {
  id: string;
  title: string;
  content: string;
  num_tokens: number;
  score?: number;
  retrieval_score?: number;
  rerank_score?: number;
  retrieval_position?: number;
  rerank_position?: number;
  vector_retrieval_position?: number;
  position?: number;
  skip: boolean;
  skip_reason: "context_window" | "duplicate";
}
export interface Source {
  score: number;
  retrieval_score?: number;
  rerank_score?: number;
  retrieval_position?: number;
  rerank_position?: number;
  vector_retrieval_position?: number;
  position?: number;
  content: string;
  title?: string;
  relevantChunks: RelevantSource[];
  num_tokens: number;
  skip: boolean;
  skip_reason: "context_window" | "duplicate";
}

export interface PowerUsageData {
  cpu_kWh: number | null;
  gpu_kWh: number | null;
  ram_kWh: number | null;
  total_kWh: number | null;
  duration: number | null;
  measurement_version?: number;
  status?: string;
  measurement_scope?: string;
  measurement_methods?: Record<string, string>;
}

export interface PowerDataDisplayed extends PowerUsageData {
  label: string;
  name: string;
}

export const POWER_FIELDS = ["cpu_kWh", "gpu_kWh", "ram_kWh", "total_kWh", "duration"] as const;

export function missingPower(): PowerUsageData {
  return { cpu_kWh: null, gpu_kWh: null, ram_kWh: null, total_kWh: null, duration: null };
}

export function validPowerValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

export function requestPowerTotal(rows: PowerDataDisplayed[]): PowerUsageData {
  const byName = new Map(rows.map(row => [row.name, row]));
  const separate = byName.get("power_prompt")?.measurement_version === 2;
  // The public chatbot includes saved startup indexing; testbench totals are separate.
  const names = separate ? ["power_index", "power_prompt", "power_rerank", "power_response"] : ["power_index", "power_prompt", "power_response"];
  const total = missingPower();
  for (const field of POWER_FIELDS) {
    const values = names.map(name => validPowerValue(byName.get(name)?.[field]));
    total[field] = values.every(value => value !== null)
      ? values.reduce<number>((sum, value) => sum + (value as number), 0) : null;
  }
  return total;
}

export type Step = "initial" | "research" | "prompt" | "output" | "done";
