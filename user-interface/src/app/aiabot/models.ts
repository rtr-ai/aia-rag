export interface LLMMessageParams {
  content: string;
  type: 'error' | 'sources' | 'user' | 'assistant';
}

interface RelevantSource {
  id:string;
  title:string;
  content:string;
  num_tokens:number;
  skip: boolean;
}
export interface Source {
  score: number;
  content: string;
  title?: string;
  relevantChunks: RelevantSource[]
  num_tokens: number;
  skip: boolean;
}

export type Step = 'initial' | 'research' | 'prompt' | 'output' | 'done';
