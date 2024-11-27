export interface LLMMessageParams {
  content: string;
  type: 'error' | 'sources' | 'user' | 'assistant';
}
export interface Source {
  score: number;
  content: string;
  title?: string;
}

export type Step = 'initial' | 'research' | 'prompt' | 'output' | 'done';
