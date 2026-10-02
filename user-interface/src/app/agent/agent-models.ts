export interface AgentEvent {
  v: number;
  seq: number;
  run_id: string;
  step_id: string | null;
  parent_step_id: string | null;
  type: string;
  payload: any;
}
export interface EnergyData {
  cpu_kWh: number;
  gpu_kWh: number;
  ram_kWh: number;
  total_kWh: number;
  duration: number;
}
export interface ToolSpec {
  name: string;
  description: string;
  parameters: { properties?: Record<string, { type?: string }>; required?: string[] };
}
export interface ToolCall {
  tool_call_id: string;
  name: string;
  arguments: string;
}
export interface LlmMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning: string | null;
  tool_calls: { id: string; name: string; raw_arguments: string }[];
  tool_call_id: string | null;
  tool_name: string | null;
}
export interface RagRelevantChunk {
  id: string;
  title: string;
  content: string;
  num_tokens: number;
  skip: boolean;
  position: number;
  skip_reason: string;
}
export interface RagSource {
  content: string;
  score: number;
  title: string;
  relevantChunks: RagRelevantChunk[];
  num_tokens: number;
  skip: boolean;
  position: number;
  skip_reason: string;
}
export interface NestedLlmCall {
  stepId: string;
  systemText?: string;
  messages?: LlmMessage[];
  reasoning: string;
  content: string;
  status: "running" | "done";
  duration?: number;
  ttft?: number | null;
  generationDuration?: number | null;
  energy?: EnergyData | null;
}
export interface Limits {
  max_turns: number;
  max_calls_per_tool: Record<string, number>;
}
export interface Entry {
  id: number;
  stepId: string | null;
  kind: "system" | "user" | "llm" | "tool";
  status: "running" | "done";
  collapsed?: boolean;
  systemText?: string;
  userText?: string;
  contextMessages?: number;
  contextTokens?: number;
  messages?: LlmMessage[];
  reasoning?: string;
  content?: string;
  toolCalls?: ToolCall[];
  toolName?: string;
  argsJson?: string;
  args?: Record<string, unknown> | null;
  callIndex?: number;
  resultText?: string;
  isError?: boolean;
  duration?: number;
  ttft?: number | null;
  generationDuration?: number | null;
  energy?: EnergyData | null;
  sources?: RagSource[];
  sourcesDuration?: number | null;
  nestedCalls?: NestedLlmCall[];
  subSteps?: { label: string; detail: string }[];
}
