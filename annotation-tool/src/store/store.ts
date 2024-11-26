import { configureStore } from '@reduxjs/toolkit';
import chunkReducer from './chunkSlice';

export const store = configureStore({
  reducer: {
    chunk: chunkReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
