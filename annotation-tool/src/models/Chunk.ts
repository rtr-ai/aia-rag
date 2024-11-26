export interface Chunk extends ChunkItemDTO {
  availableKeywords: { value: string; label: string }[];
}
export interface ChunkItemDTO {
  id: string;
  title: string;
  content: string;
  keywords: string[];
  availableKeywords: { value: string; label: string }[];
  negativeKeywords: string[];
  relevantChunksIds: string[];
  parameters: string[];
}
export interface TextItem {
  id: string;
  content: string;
}

export interface ManualIndex {
  id: string;
  chunks: Array<Chunk | TextItem>;
  date_created: number;
  last_modified: number;
}
