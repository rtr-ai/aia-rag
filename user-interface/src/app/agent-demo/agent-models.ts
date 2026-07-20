// Datenmodell für die agentische UI-Demo.
// Alle Inhalte sind deterministisch (fest hinterlegtes Replay) – es findet
// kein echter Backend-Aufruf statt. Die Oberfläche bildet den tatsächlichen
// Ablauf eines agentischen Tool-Calling-Laufs nach: ein wachsender
// Nachrichten-Verlauf (System-Prompt → Frage → Assistenz-Turns mit Tool-Calls
// → Tool-Ergebnisse), der bei jedem LLM-Aufruf vollständig erneut übergeben
// wird.

export type ToolName =
  | "heute"
  | "definition"
  | "anwendbarkeit"
  | "artikel_nachschlagen"
  | "suche_ai_act"
  | "suche_leitlinien_praxisleitfaeden"
  | "rueckfrage_nutzer";

export interface EnergyData {
  cpu_kWh: number;
  gpu_kWh: number;
  ram_kWh: number;
  total_kWh: number;
  duration: number; // Sekunden
}

export type SourceKind =
  | "ai_act" // verbindlicher Verordnungstext (Artikel)
  | "erwaegungsgrund" // Erwägungsgrund
  | "ki-servicestelle" // Material der KI-Servicestelle
  | "querverweis" // manuell gepflegter Querverweis
  | "leitlinie" // Leitlinie der EU-Kommission
  | "cop"; // Code of Practice / Praxisleitfaden

// Eine durch ein Such-Tool zurückgegebene Quelle.
// Wichtig: cosine/rankBefore/rankAfter/rerankScore sind ausschließlich
// Transparenz-/Visualisierungsdaten für die Betrachter:innen. Dem LLM wird
// nur der finale Treffertext (Titel + Inhalt) in der gerankten Reihenfolge
// übergeben – es „weiß“ nichts über Ähnlichkeitswerte oder Rangänderungen.
export interface RetrievedSource {
  id: string;
  title: string;
  snippet: string;
  kind: SourceKind;
  cosine: number; // semantische Ähnlichkeit (0..1)
  rankBefore?: number; // Rang nach reiner Ähnlichkeit
  rankAfter?: number; // Rang nach dem Reranker
  rerankScore?: number; // 0..1, Relevanzbewertung des Rerankers
  used: boolean;
  skipReason?: "duplicate" | "context_window" | "low_relevance";
}

// Ein nachvollziehbarer Teilschritt innerhalb eines Tools
// (z. B. die hybride Suche der Leitlinien-Recherche).
export interface SubStep {
  label: string;
  detail: string;
}

export interface ToolCall {
  name: ToolName;
  argsJson: string; // Argumente, wie das Modell sie als JSON ausgibt
}

// Ein Eintrag im Nachrichten-Verlauf (= eine Zeile der Timeline).
export interface Entry {
  id: number;
  kind: "system" | "user" | "llm" | "tool";
  status: "pending" | "running" | "done";
  // UI-Zustand: Abgeschlossene Karten werden automatisch auf ihre Kopfzeile
  // reduziert, sobald der nächste Schritt beginnt (per Klick wieder öffenbar).
  collapsed?: boolean;

  // kind === "system"
  systemText?: string;

  // kind === "user"
  userText?: string;

  // kind === "llm": ein LLM-Aufruf über den gesamten bisherigen Verlauf
  contextMessages?: number; // Anzahl der übergebenen Nachrichten
  contextTokens?: number; // ungefähre Tokenzahl des übergebenen Verlaufs
  reasoning?: string; // rohe Chain-of-Thought (Teil des Assistenz-Turns)
  reasoningDone?: boolean; // true, sobald das Reasoning fertig gestreamt ist
  toolCall?: ToolCall; // emittierter Funktionsaufruf (falls vorhanden)
  answer?: string; // finale Antwort (HTML), wenn kein Tool-Call mehr folgt
  citations?: { label: string; kind: SourceKind }[];

  // kind === "tool": Ergebnis einer Werkzeug-Ausführung
  toolName?: ToolName;
  argsJson?: string; // Echo der ausgeführten Argumente
  callIndex?: number; // Wievielter Aufruf dieses Tools (max. 3 erlaubt)
  resultText?: string; // ROH-Ausgabe, exakt so wie sie dem Modell übergeben wird
  isError?: boolean; // Werkzeug meldet einen Fehler (z. B. unbekannter Begriff)

  // rueckfrage_nutzer: Das Werkzeug pausiert den Lauf und wartet auf eine
  // echte Eingabe. Die Antwort (gewählter Vorschlag oder Freitext) wird
  // unverändert zu resultText.
  askQuestion?: string;
  askOptions?: string[];
  awaitingUser?: boolean;
  sources?: RetrievedSource[]; // strukturierte Trefferliste (für Visualisierung)
  rerankerApplied?: boolean;
  subSteps?: SubStep[];

  // Werkzeuge, die intern selbst ein LLM aufrufen (suche_ai_act,
  // suche_leitlinien_praxisleitfaeden): Sie besitzen ein eigenes Prompt und
  // ihre Rückgabe wird – wie jede LLM-Ausgabe – Token für Token gestreamt.
  llmGenerates?: boolean;
  ragSystemPrompt?: string; // System-Prompt des Werkzeugs
  ragUserPrompt?: string; // als User-Prompt eingesetzte Such-/Frageeingabe

  // suche_ai_act zeigt zusätzlich die Original-RAG-Visualisierung
  // Retrieve → Augment → Generate.
  ragView?: boolean;

  // gemeinsam für llm/tool
  energy?: EnergyData;
}

export interface ToolSpec {
  name: ToolName;
  signature: string;
  description: string;
}
