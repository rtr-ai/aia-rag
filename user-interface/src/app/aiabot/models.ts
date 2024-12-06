export interface LLMMessageParams {
  content: string;
  type: 'error' | 'sources' | 'user' | 'assistant';
}

interface RelevantSource {
  id:string;
  title:string;
}
export interface Source {
  score: number;
  content: string;
  title?: string;
  relevantChunks: RelevantSource[]
}

export type Step = 'initial' | 'research' | 'prompt' | 'output' | 'done';
