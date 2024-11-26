import { Chunk, ManualIndex, TextItem } from "../models/Chunk";

export const isManualIndex = (data: any): data is ManualIndex => {
  return (
    typeof data === "object" &&
    typeof data.id === "string" &&
    (typeof data.date_created === "number" ||
      data.date_created === undefined) &&
    (typeof data.last_modified === "number" ||
      data.last_modified === undefined) &&
    Array.isArray(data.chunks) &&
    data.chunks.every(
      (item: any) =>
        typeof item === "string" || isTextItem(item) || isManualChunk(item)
    )
  );
};

export const isTextItem = (data: any): data is TextItem => {
  return (
    typeof data.id === "string" &&
    typeof data.content === "string" &&
    !isManualChunk(data)
  );
};

export const isManualChunk = (data: any): data is Chunk => {
  return (
    typeof data === "object" &&
    typeof data.id === "string" &&
    typeof data.title === "string" &&
    typeof data.content === "string" &&
    Array.isArray(data.keywords) &&
    data.keywords.every((item: any) => typeof item === "string") &&
    Array.isArray(data.negativeKeywords) &&
    data.negativeKeywords.every((item: any) => typeof item === "string") &&
    Array.isArray(data.relevantChunksIds) &&
    data.relevantChunksIds.every((item: any) => typeof item === "string") &&
    Array.isArray(data.parameters) &&
    data.parameters.every((item: any) => typeof item === "string")
  );
};
