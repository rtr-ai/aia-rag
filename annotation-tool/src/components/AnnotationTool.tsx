import React, { useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import {
  setManualIndex,
  createChunk,
  setSelectionState,
} from '../store/chunkSlice';
import { ChunkItem } from './ChunkItem';
import { Header } from './Header';
import { ManualIndex } from '../models/Chunk';
import { v4 as uuidv4 } from 'uuid';

const AnnotationTool: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const [inputText, setInputText] = useState('');

  const { currentIndex, loading, selectionState } = useSelector(
    (state: RootState) => state.chunk
  );

  const confirmInput = () => {
    dispatch(
      setManualIndex({
        id: '1',
        date_created: Date.now(),
        last_modified: Date.now(),
        chunks: [{ id: uuidv4(), content: inputText }],
      })
    );
  };

  const onDownload = async () => {
    const jsonData = JSON.stringify(currentIndex, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.setAttribute('download', `${currentIndex.id}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const onFileImported = async (jsonData: ManualIndex) => {
    dispatch(setManualIndex(jsonData));
  };

  const contentRef = useRef<HTMLDivElement>(null);

  function isValidSelection() {
    return (
      selectionState.start !== selectionState.end &&
      selectionState.end > 0 &&
      selectionState.start >= 0 &&
      selectionState.id.length > 0
    );
  }

  const handleSelection = (
    id: string,
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount <= 0) return;

    const range = selection.getRangeAt(0);
    const start = range.startOffset;
    const end = range.endOffset;

    dispatch(setSelectionState({ start, end, id }));
  };

  return (
    <div className="uk-container">
      {loading ? (
        <div>Loading...</div>
      ) : (
        <>
          <Header
            onDownload={onDownload}
            onFileImported={onFileImported}
            onCut={() =>
              dispatch(
                createChunk({
                  start: selectionState.start,
                  end: selectionState.end,
                  id: selectionState.id,
                })
              )
            }
            hasValidSelection={isValidSelection()}
          />
          {currentIndex.chunks.length === 0 && (
            <div className="uk-margin ">
              <textarea
                className="uk-textarea"
                rows={5}
                placeholder="Bitte den Text hineinkopieren"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              ></textarea>
              <button
                className="uk-button uk-button-default uk-margin-top"
                onClick={confirmInput}
              >
                Weiter
              </button>
            </div>
          )}
          <div ref={contentRef} className=" uk-margin-large-top">
            {currentIndex.chunks.map((chunk) => (
              <ChunkItem
                key={chunk.id}
                chunk={chunk}
                onGenerateKeywords={(chunk) => {}}
                onMouseUp={(id, e) => handleSelection(id, e)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default AnnotationTool;
