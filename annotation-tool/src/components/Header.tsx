import UIkit from 'uikit';
import { isManualIndex } from '../utilities';
import { ManualIndex } from '../models/Chunk';
import { v4 as uuidv4 } from 'uuid';

interface HeaderProps {
  onFileImported: (data: ManualIndex) => void;
  onDownload: () => void;
  onCut: () => void;
  hasValidSelection: boolean;
}

const Header: React.FC<HeaderProps> = ({
  onFileImported,
  onDownload,
  onCut,
  hasValidSelection,
}) => {
  const selectFile = (): Promise<File | null> => {
    return new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json';
      input.onchange = (event: Event) => {
        const file = (event.target as HTMLInputElement).files?.[0] ?? null;
        resolve(file);
      };
      input.click();
    });
  };

  const parseFile = (file: File): Promise<any> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result as string);
          resolve(data);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsText(file);
    });
  };

  const handleSave = () => {
    UIkit.notification('Diese Funktion ist derzeit nicht verfügbar', {
      status: 'danger',
    });
  };

  const onSelectFile = async () => {
    try {
      const file = await selectFile();
      if (!file) return;

      const jsonData = await parseFile(file);

      if (isManualIndex(jsonData)) {
        for (let i = 0; i < jsonData.chunks.length; i++) {
          if (typeof jsonData.chunks[i] == 'string') {
            jsonData.chunks[i] = {
              id: uuidv4(),
              content: jsonData.chunks[i] as unknown as string,
            };
          }
        }
        onFileImported(jsonData);
      } else {
        UIkit.notification(
          'Importieren fehlgeschlagen. Bitte stellen Sie sicher, dass die JSON Struktur unverändert ist und versuchen Sie erneut.',
          { status: 'danger' }
        );
      }
    } catch (error) {
      UIkit.notification('Diese Datei kann nicht importiert werden', {
        status: 'danger',
      });
    }
  };

  return (
    <div
      className="uk-flex uk-flex-between uk-sticky"
      style={{
        backgroundColor: '#ffffff',
      }}
      data-uk-sticky
    >
      <button
        className="uk-button uk-button-primary"
        onClick={onCut}
        disabled={!hasValidSelection}
      >
        Textpassage schneiden
      </button>
      <button
        className="uk-button uk-button-primary"
        disabled
        onClick={handleSave}
      >
        Index Speichern
      </button>

      <button className="uk-button uk-button-primary" onClick={onDownload}>
        Index herunterladen (JSON)
      </button>

      <button className="uk-button uk-button-primary" onClick={onSelectFile}>
        Index Importieren (JSON){' '}
      </button>
    </div>
  );
};

export { Header };
