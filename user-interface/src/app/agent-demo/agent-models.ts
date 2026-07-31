// Data model of the agentic UI demo.
// All content is deterministic (a hard-coded replay) – no real backend call
// takes place. The interface reproduces the actual course of an agentic
// tool-calling run: a growing message history (system prompt → question →
// assistant turns with tool calls → tool results) that is passed again in
// full on every LLM call.

export type ToolName =
  | "today"
  | "definition"
  | "applicability"
  | "lookup_article"
  | "search_ai_act"
  | "search_guidelines"
  | "ask_user";

export interface EnergyData {
  cpu_kWh: number;
  gpu_kWh: number;
  ram_kWh: number;
  total_kWh: number;
  duration: number; // seconds
}

export type SourceKind =
  | "ai_act" // binding text of the regulation (articles)
  | "erwaegungsgrund" // recital
  | "ki-servicestelle" // material of the AI Service Desk
  | "querverweis" // manually curated cross-reference
  | "leitlinie" // guideline of the European Commission
  | "cop"; // Code of Practice / practical guide

// A source returned by a search tool.
// Important: cosine/rankBefore/rankAfter/rerankScore are purely
// transparency/visualisation data for the viewer. The LLM only receives the
// final matching text (title + content) in the ranked order – it “knows”
// nothing about similarity scores or rank changes.
export interface RetrievedSource {
  id: string;
  title: string;
  snippet: string;
  kind: SourceKind;
  cosine: number; // semantic similarity (0..1)
  rankBefore?: number; // rank by pure similarity
  rankAfter?: number; // rank after the reranker
  rerankScore?: number; // 0..1, relevance score of the reranker
  used: boolean;
  skipReason?: "duplicate" | "context_window" | "low_relevance";
}

// A traceable sub-step within a tool
// (e.g. the hybrid search of the guidelines lookup).
export interface SubStep {
  label: string;
  detail: string;
}

export interface ToolCall {
  name: ToolName;
  argsJson: string; // arguments exactly as the model emits them as JSON
}

// An entry in the message history (= one row of the timeline).
export interface Entry {
  id: number;
  kind: "system" | "user" | "llm" | "tool";
  status: "pending" | "running" | "done";
  // UI state: finished cards are automatically reduced to their header line as
  // soon as the next step begins (a click opens them again).
  collapsed?: boolean;

  // kind === "system"
  systemText?: string;

  // kind === "user"
  userText?: string;

  // kind === "llm": one LLM call over the entire history so far
  contextMessages?: number; // number of messages passed along
  contextTokens?: number; // approximate token count of the history passed
  reasoning?: string; // raw chain-of-thought (part of the assistant turn)
  reasoningDone?: boolean; // true as soon as the reasoning has finished streaming
  toolCall?: ToolCall; // emitted function call (if any)
  answer?: string; // final answer (HTML) if no further tool call follows
  citations?: { label: string; kind: SourceKind }[];

  // kind === "tool": result of a tool execution
  toolName?: ToolName;
  argsJson?: string; // echo of the executed arguments
  callIndex?: number; // which call of this tool it is (max. 3 allowed)
  resultText?: string; // RAW output, exactly as it is passed to the model
  isError?: boolean; // the tool reports an error (e.g. unknown term)

  // ask_user: the tool pauses the run and waits for a real input. The answer
  // (chosen suggestion or free text) becomes resultText unchanged.
  askQuestion?: string;
  askOptions?: string[];
  awaitingUser?: boolean;
  sources?: RetrievedSource[]; // structured list of matches (for the visualisation)
  rerankerApplied?: boolean;
  subSteps?: SubStep[];

  // Tools that internally call an LLM themselves (search_ai_act,
  // search_guidelines): they have a prompt of their own and their result is –
  // like any LLM output – streamed token by token.
  llmGenerates?: boolean;
  ragSystemPrompt?: string; // system prompt of the tool
  ragUserPrompt?: string; // search/question input used as the user prompt

  // search_ai_act additionally shows the original RAG visualisation
  // retrieve → augment → generate.
  ragView?: boolean;

  // shared by llm/tool
  energy?: EnergyData;
}

export interface ToolSpec {
  name: ToolName;
  signature: string;
  description: string;
}
