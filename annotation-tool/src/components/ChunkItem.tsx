import React, { useState } from "react";
import { Chunk, TextItem } from "../models/Chunk";
import { Select } from "./Select";
import { isTextItem } from "./../utilities/index";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../store/store";
import {
  deleteChunk,
  relevantChunksOptions,
  updateChunkProperty,
} from "../store/chunkSlice";
import { ChunkDeleteModal } from "./ChunkDeleteModal";

interface ChunkItemProps {
  chunk: Chunk | TextItem;
  keywordsLoading?: boolean;
  onGenerateKeywords: (chunk: Chunk) => void;
  onMouseUp: (id: string, event: React.MouseEvent<HTMLDivElement>) => void;
}

const TextItemComponent: React.FC<{
  chunk: TextItem;
  onMouseUp: (event: React.MouseEvent<HTMLDivElement>) => void;
}> = ({ chunk, onMouseUp }) => (
  <div onMouseUp={onMouseUp} className="content-row content-text">
    <div className="left-column">
      <span>{chunk.content}</span>
    </div>
  </div>
);

const ChunkComponent: React.FC<
  Omit<ChunkItemProps, "onMouseUp"> & { chunk: Chunk }
> = ({ chunk, keywordsLoading = false, onGenerateKeywords }) => {
  const [tempTitle, setTempTitle] = useState(chunk.title);
  const [isModalVisible, setIsModalVisible] = useState(false);

  const dispatch = useDispatch<AppDispatch>();
  const options = useSelector((state: RootState) =>
    relevantChunksOptions(state.chunk)
  ).filter((p) => p.value !== chunk.id);

  const handleContentChange = (e: React.ChangeEvent<HTMLDivElement>) => {
    dispatch(
      updateChunkProperty({
        chunkId: chunk.id,
        property: "content",
        value: e.currentTarget.textContent || "",
      })
    );
  };

  const handleTitleBlur = () => {
    dispatch(
      updateChunkProperty({
        chunkId: chunk.id,
        property: "title",
        value: tempTitle,
      })
    );
  };

  return (
    <div className="content-row content-chunk">
      <div className="left-column">
        <div>
          <div className="uk-card-title">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span style={{ fontSize: "14px", fontWeight: "bold" }}>
                {chunk.id}
              </span>
              <button
                className="uk-button uk-button-danger"
                onClick={() => setIsModalVisible(true)}
              >
                Löschen
              </button>
            </div>
            <div
              style={{
                marginTop: "8px",
                display: "flex",
                fontSize: "14px",
                fontWeight: "bold",
              }}
            >
              <div
                style={{ display: "flex" }}
                className="uk-grid-small uk-width-full"
              >
                <div className="uk-width-1-4">
                  <label className="uk-form-label">Titel</label>
                </div>
                <div className="uk-width-2-3">
                  <input
                    className="uk-input"
                    id="form-horizontal-text"
                    type="text"
                    value={tempTitle}
                    onChange={(e) => setTempTitle(e.target.value)}
                    onBlur={handleTitleBlur}
                    placeholder="Bitte eingeben"
                  />
                </div>
              </div>
            </div>
          </div>
          <div
            className="chunk-content"
            contentEditable
            suppressContentEditableWarning
            onBlur={handleContentChange}
          >
            {chunk.content}
          </div>
        </div>
      </div>
      <div className="right-column">
        <div className="chunk-settings">
          <button
            className={`uk-button uk-margin-small-bottom ${keywordsLoading ? "uk-button-disabled" : ""}`}
            disabled
          >
            Keywords via LLM generieren
          </button>
          <label>Keywords:</label>
          <Select
            value={chunk.keywords}
            mode="tags"
            onChange={(e) =>
              dispatch(
                updateChunkProperty({
                  chunkId: chunk.id,
                  property: "keywords",
                  value: e,
                })
              )
            }
            options={[]}
          />
          <label>Negative Keywords:</label>
          <Select
            value={chunk.negativeKeywords}
            mode="tags"
            onChange={(e) =>
              dispatch(
                updateChunkProperty({
                  chunkId: chunk.id,
                  property: "negativeKeywords",
                  value: e,
                })
              )
            }
            options={[]}
          />
          <label>Querverweis:</label>
          <Select
            value={chunk.relevantChunksIds}
            mode="multiple"
            onChange={(e) =>
              dispatch(
                updateChunkProperty({
                  chunkId: chunk.id,
                  property: "relevantChunksIds",
                  value: e,
                })
              )
            }
            options={options}
          />
          <label>URL:</label>
          <Select
            value={chunk.parameters}
            mode="tags"
            onChange={(e) =>
              dispatch(
                updateChunkProperty({
                  chunkId: chunk.id,
                  property: "parameters",
                  value: e,
                })
              )
            }
            options={[]}
          />
        </div>
      </div>
      {isModalVisible && (
        <ChunkDeleteModal
          onCancel={() => setIsModalVisible(false)}
          chunk={chunk}
          deleteChunk={() => dispatch(deleteChunk(chunk.id))}
        />
      )}
    </div>
  );
};

const ChunkItem: React.FC<ChunkItemProps> = (props) => {
  const { chunk } = props;

  if (isTextItem(chunk)) {
    return (
      <TextItemComponent
        chunk={chunk}
        onMouseUp={(e) => props.onMouseUp(chunk.id, e)}
      />
    );
  }

  return (
    <ChunkComponent
      chunk={chunk}
      keywordsLoading={props.keywordsLoading}
      onGenerateKeywords={props.onGenerateKeywords}
    />
  );
};

export { ChunkItem };
