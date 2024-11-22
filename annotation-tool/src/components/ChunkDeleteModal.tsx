import { useState } from 'react';
import { Chunk } from '../models/Chunk';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { isManualChunk } from '../utilities';
import React from 'react';

interface ChunkDeleteProps {
  chunk: Chunk;
  onCancel: () => void;
  deleteChunk: () => void;
}

const ChunkDeleteModal: React.FC<ChunkDeleteProps> = ({
  chunk,
  deleteChunk,
  onCancel,
}) => {
  const { currentIndex } = useSelector((state: RootState) => state.chunk);
  const [relatedChunks, setRelatedChunks] = useState<Chunk[]>(
    () =>
      currentIndex.chunks.filter(
        (item) =>
          isManualChunk(item) && item.relevantChunksIds.includes(chunk.id)
      ) as Chunk[]
  );

  const handleDelete = () => {
    deleteChunk();
    onCancel();
  };

  const handleCancel = () => {
    onCancel();
  };

  return (
    <div id={`modal-${chunk.id}`}>
      <div
        id={`delete-modal-${chunk.id}`}
        data-container={`modal-${chunk.id}`}
        uk-modal="true"
        className="uk-open"
        style={{ display: 'block' }}
      >
        <div className="uk-modal-dialog uk-modal-body">
          <h2 className="uk-modal-title">
            Möchten Sie diese Textpassage wirklich löschen?
          </h2>
          {relatedChunks.length > 0 ? (
            <p>
              Es gibt Querverweise auf diese Textpassage in folgenden
              Textpassagen:
            </p>
          ) : (
            <p>Keine Querverweise gefunden.</p>
          )}
          <ul>
            {relatedChunks.map((chunk) => (
              <li key={chunk.id}>{`${chunk.id} | ${chunk.title}`}</li>
            ))}
          </ul>
          <div className="uk-modal-footer uk-text-right">
            <button
              className="uk-button uk-button-default"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                handleCancel();
              }}
            >
              Abbrechen
            </button>
            <button
              className="uk-button uk-button-primary"
              onClick={handleDelete}
            >
              Löschen
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export { ChunkDeleteModal };
