export interface LLMMessageParams {
  content: string;
  type:
    | "error"
    | "sources"
    | "user"
    | "assistant"
    | "power_prompt"
    | "power_index"
    | "power_response";
}

interface RelevantSource {
  id: string;
  title: string;
  content: string;
  num_tokens: number;
  skip: boolean;
  skip_reason: "context_window" | "duplicate";
}
export interface Source {
  score: number;
  content: string;
  title?: string;
  relevantChunks: RelevantSource[];
  num_tokens: number;
  skip: boolean;
  skip_reason: "context_window" | "duplicate";
}

export interface PowerUsageData {
  cpu_kWh: number;
  gpu_kWh: number;
  ram_kWh: number;
  total_kWh: number;
  duration: number;
}

export interface PowerDataDisplayed {
  cpu_kWh: number;
  gpu_kWh: number;
  ram_kWh: number;
  total_kWh: number;
  duration: number;
  label: string;
  name: string;
}

export type Step = "initial" | "research" | "prompt" | "output" | "done";
