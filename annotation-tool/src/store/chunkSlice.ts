import { createSelector, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ManualIndex, Chunk, TextItem } from '../models/Chunk';
import { v4 as uuidv4 } from 'uuid';
import { isManualChunk, isTextItem } from '../utilities';
interface ChunkState {
  currentIndex: ManualIndex;
  loading: boolean;
  selectionState: {
    start: number;
    end: number;
    id: string;
  };
}

const initialState: ChunkState = {
  currentIndex: {
    id: '',
    chunks: [],
    date_created: Date.now(),
    last_modified: Date.now(),
  },
  loading: false,
  selectionState: { start: 0, end: 0, id: '' },
};

const chunkSlice = createSlice({
  name: 'chunk',
  initialState,
  reducers: {
    setManualIndex(state, action: PayloadAction<ManualIndex>) {
      state.currentIndex = action.payload;
      state.loading = false;
    },
    updateChunkProperty(
      state,
      action: PayloadAction<{
        chunkId: string;
        property: keyof Chunk;
        value: any;
      }>
    ) {
      state.currentIndex.chunks = state.currentIndex.chunks.map((chunk) =>
        chunk.id === action.payload.chunkId
          ? { ...chunk, [action.payload.property]: action.payload.value }
          : chunk
      );
    },
    deleteChunk(state, action: PayloadAction<string>) {
      const chunkId = action.payload;
      if (chunkId.length === 0) return;
      const newElements: Array<TextItem | Chunk> = [];
      const chunks = [...state.currentIndex.chunks];
      chunks.forEach((item, index) => {
        if (item.id === chunkId) {
          if (isManualChunk(item)) {
            item.relevantChunksIds = item.relevantChunksIds.filter(
              (p) => p !== chunkId
            );
          }
          const chunkContent = (item as Chunk).content;
          if (
            newElements[newElements.length - 1] &&
            isTextItem(newElements[newElements.length - 1])
          ) {
            newElements[newElements.length - 1].content += chunkContent;
          } else {
            newElements.push({ id: uuidv4(), content: chunkContent });
          }
          if (chunks[index + 1] && isTextItem(chunks[index + 1])) {
            newElements[newElements.length - 1].content +=
              chunks[index + 1].content;
            chunks.splice(index + 1, 1);
          }
        } else {
          newElements.push(item);
        }
      });

      state.currentIndex.chunks = newElements;
      resetSelectionState();
    },
    resetSelectionState(state) {
      state.selectionState = { start: 0, end: 0, id: '' };
    },
    createChunk(
      state,
      action: PayloadAction<{ start: number; end: number; id: string }>
    ) {
      const { start, end, id } = action.payload;
      if (id.length <= 0 || start >= end) return;

      const chunks = [...state.currentIndex.chunks];
      const chunkIndex = chunks.findIndex((p) => p.id === id);

      if (chunkIndex < 0) {
        return;
      }

      const chunk = chunks[chunkIndex];
      if (!isTextItem(chunk)) {
        return;
      }

      const before = chunk.content.slice(0, start).trim();
      const selected = chunk.content.slice(start, end).trim();
      const after = chunk.content.slice(end).trim();

      const newChunks: Array<Chunk | TextItem> = [];
      if (before) newChunks.push({ id: uuidv4(), content: before });
      newChunks.push({
        id: uuidv4(),
        title: '',
        content: selected,
        keywords: [],
        availableKeywords: [],
        negativeKeywords: [],
        relevantChunksIds: [],
        parameters: [],
      });
      if (after) newChunks.push({ id: uuidv4(), content: after });

      state.currentIndex.chunks = [
        ...chunks.slice(0, chunkIndex),
        ...newChunks,
        ...chunks.slice(chunkIndex + 1),
      ];
      resetSelectionState();
    },
    setSelectionState(
      state,
      action: PayloadAction<{ start: number; end: number; id: string }>
    ) {
      state.selectionState = action.payload;
    },
  },
});
export const relevantChunksOptions = createSelector(
  [(state: ChunkState) => state.currentIndex.chunks],
  (chunks) =>
    chunks.filter(isManualChunk).map((chunk) => ({
      label: chunk.title || chunk.id,
      value: chunk.id,
    }))
);

export const {
  setManualIndex,
  updateChunkProperty,
  deleteChunk,
  createChunk,
  setSelectionState,
  resetSelectionState,
} = chunkSlice.actions;
export default chunkSlice.reducer;
