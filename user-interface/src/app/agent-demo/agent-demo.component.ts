import { Component, NgZone } from "@angular/core";
import { NgClass } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { EnergyData, Entry, RetrievedSource, ToolSpec } from "./agent-models";

@Component({
  selector: "app-agent-demo",
  imports: [NgClass, FormsModule],
  templateUrl: "./agent-demo.component.html",
  styleUrl: "./agent-demo.component.scss",
})
export class AgentDemoComponent {
  // --- Eingabe ---
  readonly exampleQuestion =
    "Ich möchte die Arbeitsaktivität meiner Mitarbeiter mit einem KI-System " +
    "überwachen. Insbesondere möchte ich über die Webcam erkennen, ob sie " +
    "schlecht gelaunt sind. Darf ich das?";
  userPrompt = "";
  submittedPrompt = "";
  maxLength = 500;

  // --- Ablaufsteuerung ---
  running = false;
  done = false;
  entries: Entry[] = [];

  // --- Energie ---
  totalEnergy: EnergyData = this.emptyEnergy();
  totalProQuery = 0;

  readonly model = {
    llm: "mistral-small-3.2 (Q5_K_M)",
    embedding: "snowflake-arctic-embed-l-v2.0",
    reranker: "bge-reranker-v2-m3",
    host: "Self-Hosting · Hetzner · NVIDIA RTX 4000 Ada (20 GB VRAM) · ollama",
    maxCallsPerTool: 3,
  };

  // Feste Liste der im AI Act (v. a. Art. 3) definierten Begriffe. Wird
  // definition() mit einem nicht enthaltenen Begriff aufgerufen, gibt das
  // Werkzeug eine Fehlermeldung samt dieser Liste zurück.
  readonly definedTerms: string[] = [
    "KI-System", "Risiko", "Anbieter", "Betreiber", "Bevollmächtigter",
    "Einführer", "Händler", "Akteur", "Inverkehrbringen",
    "Bereitstellung auf dem Markt", "Inbetriebnahme", "Zweckbestimmung",
    "vernünftigerweise vorhersehbare Fehlanwendung", "Sicherheitsbauteil",
    "Betriebsanleitungen", "Rückruf eines KI-Systems", "Rücknahme eines KI-Systems",
    "Leistung eines KI-Systems", "notifizierende Behörde", "Konformitätsbewertung",
    "Konformitätsbewertungsstelle", "notifizierte Stelle", "wesentliche Veränderung",
    "CE-Kennzeichnung", "System zur Beobachtung nach dem Inverkehrbringen",
    "Marktüberwachungsbehörde", "harmonisierte Norm", "gemeinsame Spezifikation",
    "Trainingsdaten", "Validierungsdaten", "Validierungsdatensatz", "Testdaten",
    "Eingabedaten", "biometrische Daten", "biometrische Identifizierung",
    "biometrische Verifizierung", "besondere Kategorien personenbezogener Daten",
    "sensible operative Daten", "Emotionserkennungssystem",
    "System zur biometrischen Kategorisierung", "biometrisches Fernidentifizierungssystem",
    "biometrisches Echtzeit-Fernidentifizierungssystem",
    "System zur nachträglichen biometrischen Fernidentifizierung",
    "öffentlich zugänglicher Raum", "Strafverfolgungsbehörde", "Strafverfolgung",
    "Büro für Künstliche Intelligenz", "zuständige nationale Behörde",
    "schwerwiegender Vorfall", "personenbezogene Daten", "nicht personenbezogene Daten",
    "Profiling", "Plan für einen Test unter Realbedingungen", "Plan für das Reallabor",
    "KI-Reallabor", "KI-Kompetenz", "Test unter Realbedingungen", "Testteilnehmer",
    "informierte Einwilligung", "Deepfake", "weitverbreiteter Verstoß",
    "KI-Modell mit allgemeinem Verwendungszweck", "Fähigkeiten mit hoher Wirkkraft",
    "systemisches Risiko", "KI-System mit allgemeinem Verwendungszweck",
    "Gleitkommaoperation", "nachgelagerter Anbieter",
  ];

  // Die Tool-Definitionen werden bei jedem LLM-Aufruf als Teil des Kontexts
  // mitgeschickt – sie sind hier deshalb bewusst als Bestandteil der
  // System-Nachricht dargestellt.
  readonly tools: ToolSpec[] = [
    { name: "heute", signature: "heute()", description: "Gibt das heutige Datum zurück." },
    {
      name: "definition",
      signature: "definition(begriff: string)",
      description:
        "Liefert die Legaldefinition eines juristischen Begriffs (meist Art. 3 " +
        "KI-VO), ergänzt um einfache Klarstellungen aus den Leitlinien. Definiert " +
        "ist eine feste Liste von Begriffen; bei einem nicht enthaltenen Begriff " +
        "gibt das Werkzeug eine Fehlermeldung samt der Liste aller verfügbaren " +
        "Begriffe zurück.",
    },
    {
      name: "anwendbarkeit",
      signature: "anwendbarkeit(artikel_nummer: string)",
      description:
        "Erläutert die zeitliche Geltung eines Artikels als Prosa (inkl. " +
        "Ausnahme- und Grandfathering-Klauseln).",
    },
    {
      name: "artikel_nachschlagen",
      signature: "artikel_nachschlagen(artikel_nummer: string)",
      description:
        "Gibt den Reintext eines Artikels zurück, granular bis zur ersten " +
        "Ebene (z. B. „5-1“ für Art. 5 Abs. 1).",
    },
    {
      name: "suche_ai_act",
      signature: "suche_ai_act(suchbegriff: string)",
      description:
        "Semantische Suche in AI Act, Erwägungsgründen und Materialien der " +
        "KI-Servicestelle – inklusive manuell gepflegter Querverweise.",
    },
    {
      name: "suche_leitlinien_praxisleitfaeden",
      signature: "suche_leitlinien_praxisleitfaeden(bezugsartikel?: string, frage: string)",
      description:
        "Hybride (lexikalisch + semantische) Suche in den Leitlinien und " +
        "Praxisleitfäden der EU-Kommission zu einem Bezugsartikel.",
    },
  ];

  showToolDefs = false;

  // Sobald der/die Betrachter:in während eines Laufs selbst scrollt, wird das
  // automatische Mitscrollen deaktiviert (nur echte Nutzer-Interaktionen zählen,
  // nicht das programmatische scrollIntoView).
  private userScrolled = false;

  constructor(private zone: NgZone) {
    const stop = () => (this.userScrolled = true);
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    window.addEventListener("keydown", (ev) => {
      if (
        [
          "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ",
        ].includes(ev.key)
      ) {
        this.userScrolled = true;
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Energie
  // ---------------------------------------------------------------------------
  private emptyEnergy(): EnergyData {
    return { cpu_kWh: 0, gpu_kWh: 0, ram_kWh: 0, total_kWh: 0, duration: 0 };
  }

  private accumulateEnergy() {
    this.totalEnergy = this.emptyEnergy();
    this.totalProQuery = 0;
    for (const e of this.entries) {
      if (!e.energy) continue;
      this.totalEnergy.cpu_kWh += e.energy.cpu_kWh;
      this.totalEnergy.gpu_kWh += e.energy.gpu_kWh;
      this.totalEnergy.ram_kWh += e.energy.ram_kWh;
      this.totalEnergy.total_kWh += e.energy.total_kWh;
      this.totalEnergy.duration += e.energy.duration;
      this.totalProQuery += e.energy.total_kWh;
    }
  }

  toolCallCount(tool: string): number {
    return this.entries.filter((e) => e.kind === "tool" && e.toolName === tool)
      .length;
  }

  // ---------------------------------------------------------------------------
  // Anzeige-Helfer
  // ---------------------------------------------------------------------------
  formatScore(score: number): string {
    return (score * 100).toFixed(1).replace(".", ",") + " %";
  }
  formatKWh(value: number): string {
    return value.toFixed(6).replace(".", ",");
  }
  formatTokens(n: number | undefined): string {
    return n === undefined ? "" : new Intl.NumberFormat("de-DE").format(n);
  }
  sourceKindLabel(kind: string): string {
    switch (kind) {
      case "ai_act": return "AI Act (verbindlich)";
      case "erwaegungsgrund": return "Erwägungsgrund";
      case "ki-servicestelle": return "KI-Servicestelle";
      case "querverweis": return "Querverweis";
      case "leitlinie": return "Leitlinie der EU-Kommission";
      case "cop": return "Code of Practice";
      default: return kind;
    }
  }
  toggle(id: string) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("uk-hidden");
  }
  loadExample() {
    this.userPrompt = this.exampleQuestion;
  }

  // Der Verlauf, der einem LLM-Aufruf übergeben wird = alle vorherigen Einträge.
  contextOf(idx: number): Entry[] {
    return this.entries.slice(0, idx);
  }
  roleLabel(e: Entry): string {
    if (e.kind === "system") return "system";
    if (e.kind === "user") return "user";
    if (e.kind === "tool") return "tool";
    return "assistant";
  }
  private truncate(s: string, n = 90): string {
    const t = s.replace(/\s+/g, " ").trim();
    return t.length > n ? t.slice(0, n) + " …" : t;
  }
  entrySummary(e: Entry): string {
    switch (e.kind) {
      case "system":
        return "System-Prompt (+ Tool-Definitionen)";
      case "user":
        return this.truncate(e.userText || "");
      case "tool":
        return `${e.toolName}() → ${this.truncate(e.resultText || "", 80)}`;
      default:
        return e.toolCall
          ? `${this.truncate(e.reasoning || "", 64)} → ${e.toolCall.name}(…)`
          : "finale Antwort";
    }
  }
  // Vollständiger Inhalt einer Nachricht (für das Ausklappen im Verlauf).
  entryFullText(e: Entry): string {
    switch (e.kind) {
      case "system":
        return e.systemText || "";
      case "user":
        return e.userText || "";
      case "tool":
        return e.resultText || "";
      default: {
        const tail = e.toolCall
          ? `\n\n→ tool_call: ${e.toolCall.name}(${e.toolCall.argsJson})`
          : "\n\n→ finale Antwort";
        return (e.reasoning || "") + tail;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Loop-Visualisierung (Kreislauf-Diagramm + sticky Verlaufs-Leiste)
  // ---------------------------------------------------------------------------
  // Wievielte Runde des Agenten-Loops läuft gerade (= Anzahl LLM-Aufrufe).
  get roundCount(): number {
    return this.entries.filter((e) => e.kind === "llm").length;
  }
  // Die Schritte des Laufs als Chips (nur LLM- und Tool-Einträge).
  get chips(): Entry[] {
    return this.entries.filter((e) => e.kind === "llm" || e.kind === "tool");
  }
  // Welcher Knoten/welche Kante des Kreislauf-Diagramms gerade aktiv ist –
  // vollständig aus dem Nachrichten-Verlauf abgeleitet.
  get loopActive(): { node: string; edge: string } {
    if (this.done) return { node: "antwort", edge: "nein" };
    const e = this.entries[this.entries.length - 1];
    if (!e || !this.running || e.status !== "running") return { node: "", edge: "" };
    if (e.kind === "tool") return { node: "tool", edge: "zurueck" };
    // System-/User-Nachricht: Der erste LLM-Aufruf wird gerade vorbereitet.
    if (e.kind !== "llm") return { node: "llm", edge: "" };
    if (e.reasoningDone) {
      return e.toolCall
        ? { node: "llm", edge: "ja" }
        : { node: "antwort", edge: "nein" };
    }
    return { node: "llm", edge: "" };
  }
  chipLabel(e: Entry): string {
    if (e.kind === "llm") return e.toolCall ? "LLM" : "✓ Antwort";
    const short: Record<string, string> = {
      suche_leitlinien_praxisleitfaeden: "suche_leitlinien…",
    };
    const name = short[e.toolName!] ?? e.toolName;
    return name + "()" + (e.isError ? " ✗" : "");
  }
  chipClass(e: Entry): string {
    const cls: string[] = [];
    if (e.kind === "llm") cls.push(e.toolCall ? "chip-llm" : "chip-final");
    else cls.push("chip-tool");
    if (e.isError) cls.push("chip-error");
    if (e.status === "running") cls.push("chip-running");
    return cls.join(" ");
  }
  // Klick auf einen Chip: zum Eintrag springen. Das ist eine bewusste
  // Navigation – danach kein automatisches Mitscrollen mehr.
  jumpTo(id: number) {
    this.userScrolled = true;
    document
      .getElementById("entry-" + id)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  private scrollToEntry(id: number, _block: ScrollLogicalPosition = "nearest") {
    if (this.userScrolled) return;
    setTimeout(() => {
      if (this.userScrolled) return;
      const el = document.getElementById("entry-" + id);
      if (!el) return;
      // Unterkante des Eintrags 200px über dem Fensterrand halten, damit
      // nachlaufender (gestreamter) Text nicht am unteren Rand abgeschnitten
      // wird. Nur nach unten scrollen – nie nach oben zurückspringen.
      const delta = el.getBoundingClientRect().bottom - (window.innerHeight - 200);
      if (delta > 0) window.scrollBy({ top: delta, behavior: "smooth" });
    }, 60);
  }

  // Streamt einen Text Wort für Wort in ein Feld (simuliert die Token-Ausgabe
  // eines LLM) und scrollt dabei mit.
  private async streamInto(
    entry: Entry,
    set: (v: string) => void,
    full: string,
    perWord = 40
  ) {
    this.zone.run(() => set(""));
    const words = full.split(" ");
    let acc = "";
    for (let i = 0; i < words.length; i++) {
      acc += (i === 0 ? "" : " ") + words[i];
      const v = acc;
      this.zone.run(() => set(v));
      if (i % 8 === 0) this.scrollToEntry(entry.id, "end");
      await this.delay(perWord + Math.random() * 26);
    }
  }

  // Baut das augmentierte Prompt des Sub-RAG (suche_ai_act) zusammen:
  // System-Prompt + verwendete Quellen im Volltext + Suchanfrage als User-Prompt.
  buildAugmentedPrompt(e: Entry): string {
    const used = (e.sources || [])
      .filter((s) => s.used)
      .sort((a, b) => (a.rankAfter ?? 99) - (b.rankAfter ?? 99));
    const context = used
      .map((s, i) => `[${i + 1}] ${s.title}\n${s.snippet}`)
      .join("\n\n");
    return (
      "[SYSTEM_PROMPT]\n" +
      (e.ragSystemPrompt || "") +
      "\n[/SYSTEM_PROMPT]\n\n" +
      `### Kontext (${used.length} Quellen, gerankte Reihenfolge)\n` +
      context +
      "\n\n[INST]\nSuchanfrage: " +
      (e.ragUserPrompt || "") +
      "\n[/INST]"
    );
  }

  // ---------------------------------------------------------------------------
  // Deterministisches Replay
  // ---------------------------------------------------------------------------
  private delay = (ms: number) =>
    new Promise<void>((resolve) => setTimeout(resolve, ms));

  async run() {
    if (this.running) return;
    this.entries = [];
    this.done = false;
    this.running = true;
    this.userScrolled = false;
    this.submittedPrompt = this.userPrompt;
    this.totalEnergy = this.emptyEnergy();
    this.totalProQuery = 0;

    setTimeout(
      () =>
        document
          .getElementById("agentRun")
          ?.scrollIntoView({ behavior: "smooth" }),
      150
    );

    const script = this.buildScript();
    let nextId = 0;

    for (const partial of script) {
      const entry: Entry = { id: nextId++, status: "running", ...partial } as Entry;

      // Zu streamende (LLM-)Felder vorab merken und leeren, damit während der
      // „Denkzeit“ nicht kurz der volle Text aufblitzt.
      const fullReasoning = entry.reasoning;
      const streamAnswer = entry.kind === "llm" && !entry.toolCall && !!entry.answer;
      const fullAnswer = streamAnswer ? entry.answer : undefined;
      const streamResult =
        entry.kind === "tool" && !!entry.llmGenerates && !!entry.resultText;
      const fullResult = streamResult ? entry.resultText : undefined;
      if (entry.kind === "llm" && fullReasoning) entry.reasoning = "";
      if (streamAnswer) entry.answer = "";
      if (streamResult) entry.resultText = "";
      // Plain-Tools (ohne eigenen LLM-Aufruf): Die Rückgabe erscheint erst nach
      // der Ausführungszeit – nicht schon, während der Loader noch läuft.
      const plainResult =
        entry.kind === "tool" && !entry.llmGenerates && !!entry.resultText;
      const fullPlainResult = plainResult ? entry.resultText : undefined;
      if (plainResult) entry.resultText = "";
      // Interne Teilschritte (z. B. Retrieve-Pipeline) werden nacheinander
      // aufgedeckt, damit die Reihenfolge der Pipeline erlebbar wird.
      const fullSubSteps =
        entry.kind === "tool" && entry.subSteps && entry.subSteps.length > 0
          ? entry.subSteps
          : undefined;
      if (fullSubSteps) entry.subSteps = [];

      this.zone.run(() => this.entries.push(entry));
      this.scrollToEntry(entry.id);

      if (entry.kind === "system" || entry.kind === "user") {
        await this.delay(800);
      } else if (entry.kind === "tool") {
        if (fullSubSteps) {
          // Pipeline-Teilschritte einzeln aufdecken (Retrieve → … → Reranking).
          await this.delay(600);
          for (const sub of fullSubSteps) {
            this.zone.run(() => entry.subSteps!.push(sub));
            this.scrollToEntry(entry.id, "end");
            await this.delay(750);
          }
          await this.delay(400);
        } else {
          await this.delay(1100); // Ausführungszeit des Werkzeugs
        }
      } else if (this.roundCount === 1) {
        // Vor der allerersten LLM-Runde bewusst länger warten, damit
        // Anwender:innen Zeit haben, das Kreislauf-Diagramm zu lesen.
        await this.delay(4500);
      } else {
        await this.delay(900); // „Denkzeit“ vor dem Reasoning
      }

      if (entry.kind === "llm") {
        // Reasoning ist LLM-Ausgabe → Token für Token streamen.
        if (fullReasoning) {
          await this.streamInto(entry, (v) => (entry.reasoning = v), fullReasoning);
        }
        this.zone.run(() => (entry.reasoningDone = true));
        this.scrollToEntry(entry.id, "end");

        // Finale Antwort streamen (nach dem Reasoning, falls kein Tool-Call).
        if (fullAnswer !== undefined) {
          await this.delay(300);
          await this.streamInto(entry, (v) => (entry.answer = v), fullAnswer, 44);
        } else if (entry.toolCall) {
          await this.delay(350); // kurze Pause, dann wird der Tool-Call emittiert
        }
      } else if (fullResult !== undefined) {
        // Werkzeuge mit eigenem LLM-Aufruf: Rückgabe Token für Token streamen.
        await this.streamInto(entry, (v) => (entry.resultText = v), fullResult);
      } else if (fullPlainResult !== undefined) {
        // Kein LLM beteiligt: Die Rückgabe erscheint auf einen Schlag.
        this.zone.run(() => (entry.resultText = fullPlainResult));
        this.scrollToEntry(entry.id, "end");
      }

      this.zone.run(() => {
        entry.status = "done";
        this.accumulateEnergy();
      });
    }

    this.zone.run(() => {
      this.running = false;
      this.done = true;
    });
  }

  // ---------------------------------------------------------------------------
  // Fest hinterlegter Nachrichten-Verlauf für die Beispielfrage.
  // 8 LLM-Aufrufe (7 mit Tool-Call, 1 mit finaler Antwort) + 7 Tool-Ergebnisse.
  // Die Energie der LLM-Aufrufe steigt mit dem wachsenden Kontext (Prefill).
  // ---------------------------------------------------------------------------
  private buildScript(): Partial<Entry>[] {
    // heute() liefert das echte aktuelle Datum – auch im deterministischen Replay.
    const heuteDatum = new Intl.DateTimeFormat("de-AT", {
      dateStyle: "full",
    }).format(new Date());
    return [
      // System-Nachricht (inkl. mitgeschickter Tool-Definitionen)
      {
        kind: "system",
        systemText:
          "Du bist der AI-Act-Assistent der KI-Servicestelle (RTR). Beantworte " +
          "Fragen zum EU AI Act ausschließlich auf Basis der über die Werkzeuge " +
          "abgerufenen Quellen.\n\n" +
          "Regeln:\n" +
          "1. Plane, welche Werkzeuge du benötigst, und rufe sie schrittweise " +
          "auf (max. 3× je Werkzeug).\n" +
          "2. Stütze jede Aussage auf einen Beleg. Erfinde keine Artikel, " +
          "Randnummern oder Inhalte.\n" +
          "3. Verbindliche Bestimmungen (AI Act) haben Vorrang; Leitlinien/" +
          "Praxisleitfäden dürfen nur ergänzen.\n" +
          "4. Bei Fragen zur zeitlichen Geltung verwende immer anwendbarkeit().\n" +
          "5. Lege deine Überlegungen offen und nenne am Ende alle Quellen.",
      },

      // Nutzerfrage
      { kind: "user", userText: this.exampleQuestion },

      // --- LLM-Aufruf 1: startet stets mit der breiten semantischen Suche ---
      {
        kind: "llm",
        contextMessages: 2,
        contextTokens: 712,
        reasoning:
          "Ich kenne die einschlägigen Bestimmungen noch nicht. Wie bei " +
          "praktisch jeder Frage beginne ich mit einer breiten semantischen " +
          "Suche im AI Act, um zu sehen, welche Artikel und Erwägungsgründe " +
          "überhaupt relevant sind. (Nur wenn jemand ausdrücklich nach einem " +
          "bestimmten Artikel fragt, würde ich diesen direkt nachschlagen.) Als " +
          "Suchbegriff nehme ich die Kernbegriffe der Frage.",
        toolCall: {
          name: "suche_ai_act",
          argsJson:
            '{ "suchbegriff": "Mitarbeiter mit KI über Webcam überwachen Stimmung erkennen Arbeitsplatz" }',
        },
        energy: this.energy(0.000016, 0.000058, 0.000004, 1.5),
      },
      {
        kind: "tool",
        toolName: "suche_ai_act",
        callIndex: 1,
        argsJson:
          '{ "suchbegriff": "Mitarbeiter mit KI über Webcam überwachen Stimmung erkennen Arbeitsplatz" }',
        rerankerApplied: true,
        ragView: true,
        llmGenerates: true,
        ragUserPrompt:
          "Mitarbeiter mit KI über Webcam überwachen Stimmung erkennen Arbeitsplatz",
        ragSystemPrompt:
          "Du bist die semantische Suche der KI-Servicestelle über AI Act, " +
          "Erwägungsgründe und KI-Servicestellen-Materialien. Beantworte die " +
          "Suchanfrage ausschließlich aus den untenstehenden Quellen und gib zu " +
          "jeder Aussage die Fundstelle an. Triff keine eigene rechtliche " +
          "Bewertung – fasse nur die einschlägigen Bestimmungen zusammen.",
        subSteps: [
          {
            label: "Retrieve · ① Embedding der Suchanfrage",
            detail:
              "Der Suchbegriff wird mit snowflake-arctic-embed-l-v2.0 in einen " +
              "Vektor überführt.",
          },
          {
            label: "Retrieve · ② Semantische Vektorsuche",
            detail:
              "Abgleich gegen den Korpus → 38 Kandidaten nach Kosinus-Ähnlichkeit, " +
              "begrenzt auf die 25 stärksten bzw. 15.000 Token.",
          },
          {
            label: "Retrieve · ③ Querverweise ergänzen",
            detail:
              "Manuell gepflegte Querverweise zwischen Bestimmungen werden " +
              "hinzugeladen (hier: Art. 26 aus Anhang III Nr. 4).",
          },
          {
            label: "Retrieve · ④ Reranking",
            detail:
              "bge-reranker-v2-m3 bewertet jeden Kandidaten gegen die Anfrage " +
              "neu und reiht final; Dubletten und Treffer geringer Relevanz " +
              "werden verworfen.",
          },
        ],
        sources: this.searchAiActSources(),
        // „Generate“ des Sub-RAG: quellengebundene Passage, die an den Agenten
        // zurückgeht (= roher Tool-Output, keine nachträgliche UI-Zusammenfassung).
        resultText:
          "Am Arbeitsplatz ist die Ableitung von Emotionen durch KI-Systeme " +
          "verboten (Art. 5 Abs. 1 lit. f KI-VO); Ausnahmen bestehen nur aus " +
          "medizinischen oder Sicherheitsgründen. Erwägungsgrund 44 begründet " +
          "das Verbot mit dem Machtungleichgewicht und der begrenzten " +
          "wissenschaftlichen Verlässlichkeit. Daneben können KI-Systeme zur " +
          "Überwachung/Bewertung der Arbeitsleistung – soweit nicht verboten – " +
          "nach Anhang III Nr. 4 als Hochrisiko-KI-Systeme gelten. " +
          "[Fundstellen: Art. 5 Abs. 1 lit. f; ErwG 44; Anhang III Nr. 4]",
        energy: this.energy(0.000071, 0.001034, 0.000021, 6.2),
      },

      // --- LLM-Aufruf 2: erkennt aus den Treffern das Emotions-Thema, holt die
      // Definition – zunächst mit dem umgangssprachlichen (falschen) Begriff ---
      {
        kind: "llm",
        contextMessages: 4,
        contextTokens: 936,
        reasoning:
          "Die Suche bringt es an die Oberfläche: Im Zentrum steht Art. 5 Abs. 1 " +
          "lit. f – das Ableiten von Emotionen am Arbeitsplatz. Mein Vorhaben " +
          "(die „Laune“ über die Webcam erkennen) klingt nach " +
          "„Emotionserkennung“. Ob es unter das Verbot fällt, hängt am genauen " +
          "Rechtsbegriff – ich hole die Definition.",
        toolCall: {
          name: "definition",
          argsJson: '{ "begriff": "Emotionserkennung" }',
        },
        energy: this.energy(0.000017, 0.000064, 0.000005, 1.7),
      },
      // Erster Aufruf schlägt fehl: Der Begriff ist nicht in der Liste. Das
      // Werkzeug antwortet – wie spezifiziert – mit Fehler + Begriffsliste,
      // sodass sich das Modell selbst korrigieren kann.
      {
        kind: "tool",
        toolName: "definition",
        callIndex: 1,
        isError: true,
        argsJson: '{ "begriff": "Emotionserkennung" }',
        resultText:
          "FEHLER: Für „Emotionserkennung“ ist keine Legaldefinition " +
          "hinterlegt.\n\nVerfügbare Begriffe (" +
          this.definedTerms.length +
          "): " +
          this.definedTerms.join(", "),
        energy: this.energy(0.000005, 0.0, 0.000002, 0.2),
      },

      // --- LLM-Aufruf 3: Selbstkorrektur anhand der Fehlermeldung ---
      {
        kind: "llm",
        contextMessages: 6,
        contextTokens: 1290,
        reasoning:
          "Der Aufruf ist fehlgeschlagen – „Emotionserkennung“ ist nicht in der " +
          "Begriffsliste. Die Fehlermeldung liefert aber alle definierten " +
          "Begriffe mit: Der zutreffende Terminus lautet " +
          "„Emotionserkennungssystem“ (Art. 3). Ich wiederhole den Aufruf mit " +
          "dem korrekten Begriff.",
        toolCall: {
          name: "definition",
          argsJson: '{ "begriff": "Emotionserkennungssystem" }',
        },
        energy: this.energy(0.000019, 0.000073, 0.000005, 1.9),
      },
      {
        kind: "tool",
        toolName: "definition",
        callIndex: 2,
        argsJson: '{ "begriff": "Emotionserkennungssystem" }',
        resultText:
          "Art. 3 Nr. 39 KI-VO: „Emotionserkennungssystem“ bezeichnet ein " +
          "KI-System, das dem Zweck dient, Emotionen oder Absichten " +
          "natürlicher Personen auf der Grundlage ihrer biometrischen Daten " +
          "festzustellen oder daraus abzuleiten.\n\n" +
          "Ergänzung (Leitlinien zu verbotenen Praktiken): Das bloße Erkennen " +
          "offenkundiger Ausdrucksformen (z. B. ein Lächeln) oder physischer " +
          "Zustände ist NICHT erfasst. Erfasst ist das Ableiten innerer " +
          "Gefühlszustände wie Zufriedenheit, Stress, Müdigkeit oder " +
          "„schlechte Laune“.",
        energy: this.energy(0.000009, 0.000005, 0.000002, 0.6),
      },

      // --- LLM-Aufruf 4: liest den genauen Wortlaut von Art. 5 Abs. 1 lit. f nach ---
      {
        kind: "llm",
        contextMessages: 8,
        contextTokens: 1428,
        reasoning:
          "Die Definition (Art. 3 Nr. 39) bestätigt: Das Ableiten innerer " +
          "Zustände wie „schlechte Laune“ ist erfasst, reines Mimik-Erkennen " +
          "nicht. Mein Fall ist also erfasst. Jetzt brauche ich den genauen " +
          "Wortlaut von Art. 5 Abs. 1 lit. f – vor allem die Ausnahmen.",
        toolCall: {
          name: "artikel_nachschlagen",
          argsJson: '{ "artikel_nummer": "5-1" }',
        },
        energy: this.energy(0.00002, 0.000079, 0.000006, 2.1),
      },
      {
        kind: "tool",
        toolName: "artikel_nachschlagen",
        callIndex: 1,
        argsJson: '{ "artikel_nummer": "5-1" }',
        resultText:
          "Art. 5 Abs. 1 KI-VO – Folgende Praktiken im KI-Bereich sind verboten: " +
          "[…]\n\n" +
          "lit. f) das Inverkehrbringen, die Inbetriebnahme zu diesem " +
          "spezifischen Zweck oder die Verwendung von KI-Systemen zur Ableitung " +
          "von Emotionen einer natürlichen Person am Arbeitsplatz und in " +
          "Einrichtungen der allgemeinen und beruflichen Bildung, es sei denn, " +
          "die Verwendung des KI-Systems soll aus medizinischen Gründen oder " +
          "Sicherheitsgründen eingeführt oder in den Verkehr gebracht werden.",
        energy: this.energy(0.000006, 0.0, 0.000002, 0.3),
      },

      // --- LLM-Aufruf 5: ruft suche_leitlinien_praxisleitfaeden() auf ---
      {
        kind: "llm",
        contextMessages: 10,
        contextTokens: 1641,
        reasoning:
          "Es gibt nur die enge Ausnahme „medizinische Gründe oder " +
          "Sicherheitsgründe“. Wie diese auszulegen ist, klären die Leitlinien " +
          "der Kommission – die liegen nicht im allgemeinen Suchpool, sondern " +
          "werden über ein eigenes Tool durchsucht.",
        toolCall: {
          name: "suche_leitlinien_praxisleitfaeden",
          argsJson:
            '{ "bezugsartikel": "Artikel 5", "frage": "Ist das Ableiten der ' +
            'Stimmung von Beschäftigten über eine Webcam am Arbeitsplatz ' +
            'verboten und greift eine Ausnahme?" }',
        },
        energy: this.energy(0.000022, 0.000086, 0.000006, 2.3),
      },
      {
        kind: "tool",
        toolName: "suche_leitlinien_praxisleitfaeden",
        callIndex: 1,
        argsJson:
          '{ "bezugsartikel": "Artikel 5", "frage": "…Stimmung über Webcam am Arbeitsplatz…" }',
        rerankerApplied: true,
        llmGenerates: true,
        ragUserPrompt:
          "Ist das Ableiten der Stimmung von Beschäftigten über eine Webcam am " +
          "Arbeitsplatz verboten und greift eine Ausnahme?",
        ragSystemPrompt:
          "Du durchsuchst die Leitlinien und Praxisleitfäden der EU-Kommission " +
          "zum Bezugsartikel und beantwortest die Frage ausschließlich aus den " +
          "untenstehenden Fundstellen. Nenne die Quelle und triff keine eigene " +
          "rechtliche Bewertung – referiere nur, was die Leitlinien sagen.",
        subSteps: [
          {
            label: "1. Lexikalische Suche (BM25)",
            detail:
              "Die Suchterme stammen nicht vom Modell, sondern werden aus dem " +
              "Parameter „frage“ extrahiert (Tokenisierung, Stoppwort-Entfernung, " +
              "Lemmatisierung) → „Arbeitsplatz“, „Emotion“, „Stimmung“, " +
              "„Ausnahme“, „Sicherheit“. Damit wird der Leitlinien-Index gematcht; " +
              "stärkster Treffer: Abschnitt „Emotion recognition“.",
          },
          {
            label: "2. Semantische Suche (Embedding)",
            detail:
              "snowflake-arctic-embed-l-v2.0 bestätigt dieselbe Quelle und " +
              "findet zusätzlich den „CoP zu Transparenzpflichten“ (geringer Score).",
          },
          {
            label: "3. Hybride Fusion + Reranker",
            detail:
              "Reciprocal Rank Fusion kombiniert beide Listen, bge-reranker-v2-m3 " +
              "reiht final; der CoP-Treffer fällt wegen geringer Relevanz heraus.",
          },
        ],
        sources: this.searchLeitlinienSources(),
        // „Generate“ des Werkzeugs: quellengebundene Passage aus den Leitlinien.
        resultText:
          "Die Ausnahmen „medizinische Gründe“ und „Sicherheitsgründe“ sind eng " +
          "auszulegen. Als Sicherheitsgrund nennt die Kommission etwa die " +
          "Müdigkeitserkennung bei Pilot:innen oder Berufskraftfahrer:innen. " +
          "Ausdrücklich NICHT gedeckt ist das Monitoring von Zufriedenheit, " +
          "Stimmung, Engagement oder Produktivität von Beschäftigten – es bleibt " +
          "nach Art. 5 Abs. 1 lit. f verboten, auch mit Einwilligung. Reines " +
          "Erkennen offenkundiger Ausdrucksformen ist hingegen kein " +
          "Verbotstatbestand. [Quelle: Leitlinien zu verbotenen Praktiken, " +
          "Abschnitt Emotionserkennung]",
        energy: this.energy(0.000061, 0.000874, 0.000018, 5.4),
      },

      // --- LLM-Aufruf 6: holt das heutige Datum (LLMs kennen es nicht zuverlässig) ---
      {
        kind: "llm",
        contextMessages: 12,
        contextTokens: 1868,
        reasoning:
          "Die Leitlinien bestätigen: Stimmungs-/Launenüberwachung zur " +
          "Mitarbeiterkontrolle ist nicht von der Ausnahme gedeckt. Bleibt die " +
          "zeitliche Geltung – „Darf ich das?“ heißt auch: Gilt das Verbot " +
          "überhaupt schon? Mein Trainingsstand liegt in der Vergangenheit, " +
          "das heutige Datum kenne ich nicht zuverlässig – ich rufe heute() auf.",
        toolCall: {
          name: "heute",
          argsJson: "{}",
        },
        energy: this.energy(0.000023, 0.000092, 0.000007, 2.4),
      },
      // heute() ist ein triviales Werkzeug ohne LLM: Rückgabe instantan,
      // Energieverbrauch praktisch null.
      {
        kind: "tool",
        toolName: "heute",
        callIndex: 1,
        argsJson: "{}",
        resultText: "Heute ist " + heuteDatum + ".",
        energy: this.energy(0.000001, 0.0, 0.000001, 0.1),
      },

      // --- LLM-Aufruf 7: ruft anwendbarkeit() auf ---
      {
        kind: "llm",
        contextMessages: 14,
        contextTokens: 1926,
        reasoning:
          "heute() liefert: " +
          heuteDatum +
          ". Jetzt prüfe ich, seit wann das Verbot des Art. 5 gilt – für " +
          "Fragen zur zeitlichen Geltung nutze ich laut System-Prompt immer " +
          "anwendbarkeit().",
        toolCall: {
          name: "anwendbarkeit",
          argsJson: '{ "artikel_nummer": "Artikel 5" }',
        },
        energy: this.energy(0.000024, 0.000096, 0.000007, 2.5),
      },
      {
        kind: "tool",
        toolName: "anwendbarkeit",
        callIndex: 1,
        argsJson: '{ "artikel_nummer": "Artikel 5" }',
        resultText:
          "Das Verbot bestimmter Praktiken nach Artikel 5 gilt gemäß Art. 113 " +
          "lit. a KI-VO bereits seit dem 2. Februar 2025 – früher als der " +
          "Großteil der Verordnung.\n\n" +
          "Für Art. 5 besteht KEINE Übergangs- oder Bestandsschutzregelung " +
          "(kein „Grandfathering“): Verbotene Systeme dürfen seit diesem Datum " +
          "weder in Verkehr gebracht noch in Betrieb genommen oder verwendet " +
          "werden – auch dann nicht, wenn sie zuvor bereits im Einsatz waren.",
        energy: this.energy(0.000007, 0.0, 0.000002, 0.4),
      },

      // --- LLM-Aufruf 8: finale Antwort (kein Tool-Call mehr) ---
      {
        kind: "llm",
        contextMessages: 16,
        contextTokens: 2134,
        reasoning:
          "Alle Bausteine liegen vor: Emotionserkennungssystem (Art. 3 Nr. 39), " +
          "am Arbeitsplatz nach Art. 5 Abs. 1 lit. f verboten, keine Ausnahme " +
          "greift, anwendbar seit 2.2.2025 – das Verbot gilt heute also " +
          "längst. Ich formuliere die Antwort streng quellengebunden.",
        answer: this.finalAnswer(),
        citations: [
          { label: "Art. 5 Abs. 1 lit. f KI-VO", kind: "ai_act" },
          { label: "Art. 3 Nr. 39 KI-VO", kind: "ai_act" },
          { label: "Erwägungsgrund 44 KI-VO", kind: "erwaegungsgrund" },
          { label: "Art. 113 lit. a KI-VO", kind: "ai_act" },
          {
            label: "Leitlinien zu verbotenen Praktiken (Emotionserkennung)",
            kind: "leitlinie",
          },
        ],
        energy: this.energy(0.000156, 0.002104, 0.000047, 12.8),
      },
    ];
  }

  private energy(
    cpu: number,
    gpu: number,
    ram: number,
    duration: number
  ): EnergyData {
    return {
      cpu_kWh: cpu,
      gpu_kWh: gpu,
      ram_kWh: ram,
      total_kWh: cpu + gpu + ram,
      duration,
    };
  }

  private searchAiActSources(): RetrievedSource[] {
    return [
      {
        id: "art5-1-f",
        title: "Art. 5 Abs. 1 lit. f KI-VO – Verbot Emotionserkennung am Arbeitsplatz",
        kind: "ai_act",
        snippet:
          "Verboten ist die Ableitung von Emotionen am Arbeitsplatz und in Bildungseinrichtungen, außer aus medizinischen oder Sicherheitsgründen.",
        cosine: 0.712,
        rankBefore: 3,
        rankAfter: 1,
        rerankScore: 0.94,
        used: true,
      },
      {
        id: "erwg-44",
        title: "Erwägungsgrund 44 KI-VO – Begründung des Verbots",
        kind: "erwaegungsgrund",
        snippet:
          "Ernste Bedenken gegen Emotionserkennung am Arbeitsplatz wegen des Machtungleichgewichts und der begrenzten wissenschaftlichen Verlässlichkeit.",
        cosine: 0.685,
        rankBefore: 2,
        rankAfter: 2,
        rerankScore: 0.88,
        used: true,
      },
      {
        id: "art5-1-a",
        title: "Art. 5 Abs. 1 lit. a KI-VO – manipulative/täuschende Techniken",
        kind: "ai_act",
        snippet:
          "Verbot von KI-Systemen, die das Verhalten durch unterschwellige Techniken wesentlich beeinflussen.",
        cosine: 0.731,
        rankBefore: 1,
        rankAfter: 3,
        rerankScore: 0.41,
        used: true,
      },
      {
        id: "annex3-4",
        title: "Anhang III Nr. 4 KI-VO – Beschäftigung & Personalmanagement (Hochrisiko)",
        kind: "ai_act",
        snippet:
          "KI-Systeme zur Überwachung und Bewertung der Arbeitsleistung gelten – soweit nicht verboten – als Hochrisiko-KI-Systeme.",
        cosine: 0.664,
        rankBefore: 4,
        rankAfter: 4,
        rerankScore: 0.57,
        used: true,
      },
      {
        id: "art26",
        title: "Querverweis: Art. 26 KI-VO – Pflichten der Betreiber von Hochrisiko-KI",
        kind: "querverweis",
        snippet:
          "Manuell gepflegter Querverweis aus Anhang III Nr. 4. Relevant nur für nicht verbotene Überwachungssysteme.",
        cosine: 0.602,
        rankBefore: 6,
        rankAfter: 5,
        rerankScore: 0.33,
        used: false,
        skipReason: "low_relevance",
      },
      {
        id: "erwg-44-dup",
        title: "Erwägungsgrund 44 KI-VO (Materialien KI-Servicestelle, Paraphrase)",
        kind: "ki-servicestelle",
        snippet: "Inhaltlich identisch zu Erwägungsgrund 44 – als Dublette aussortiert.",
        cosine: 0.659,
        rankBefore: 5,
        rankAfter: 6,
        rerankScore: 0.86,
        used: false,
        skipReason: "duplicate",
      },
    ];
  }

  private searchLeitlinienSources(): RetrievedSource[] {
    return [
      {
        id: "ll-prohibited-emotion",
        title:
          "Leitlinien zu verbotenen Praktiken – Abschnitt „Emotion recognition“ (Arbeitsplatz)",
        kind: "leitlinie",
        snippet:
          "Enge Auslegung der Ausnahmen; Müdigkeitserkennung (Sicherheit) zulässig, Stimmungsmonitoring von Beschäftigten verboten.",
        cosine: 0.821,
        rankBefore: 1,
        rankAfter: 1,
        rerankScore: 0.96,
        used: true,
      },
      {
        id: "ll-prohibited-scope",
        title:
          "Leitlinien zu verbotenen Praktiken – Abgrenzung Emotion vs. reines Mimik-Erkennen",
        kind: "leitlinie",
        snippet:
          "Erkennen offenkundiger Ausdrucksformen ist kein Verbotstatbestand; das Ableiten innerer Zustände schon.",
        cosine: 0.768,
        rankBefore: 3,
        rankAfter: 2,
        rerankScore: 0.79,
        used: true,
      },
      {
        id: "cop-transparency",
        title: "CoP zu Transparenzpflichten – Kennzeichnung von Emotionserkennung",
        kind: "cop",
        snippet:
          "Betrifft Offenlegungspflichten nach Art. 50, nicht das Verbot selbst – für diese Frage nicht einschlägig.",
        cosine: 0.642,
        rankBefore: 2,
        rankAfter: 3,
        rerankScore: 0.22,
        used: false,
        skipReason: "low_relevance",
      },
    ];
  }

  private finalAnswer(): string {
    return (
      "<strong>Nein – das ist nach dem EU AI Act verboten.</strong> Ein " +
      "KI-System, das über die Webcam die Stimmung Ihrer Mitarbeiter:innen " +
      "(„schlecht gelaunt“) erkennt, fällt unter die <strong>verbotenen " +
      "Praktiken</strong> und darf nicht eingesetzt werden.\n\n" +
      "<strong>Warum?</strong>\n\n" +
      "<ul>" +
      "<li><strong>Es ist ein Emotionserkennungssystem</strong> (Art. 3 Nr. 39 " +
      "KI-VO): Aus dem Gesichtsausdruck wird ein innerer Gefühlszustand " +
      "abgeleitet. Das bloße Erkennen einer Mimik wäre unkritisch – das " +
      "Schließen auf die „Laune“ ist es nicht.</li>" +
      "<li><strong>Der Einsatz am Arbeitsplatz ist ausdrücklich verboten</strong> " +
      "(Art. 5 Abs. 1 lit. f KI-VO). Erwägungsgrund 44 begründet dies u. a. mit " +
      "dem Machtgefälle und der zweifelhaften wissenschaftlichen Verlässlichkeit " +
      "solcher Systeme.</li>" +
      "<li><strong>Keine Ausnahme greift</strong>: Erlaubt wären nur " +
      "<em>medizinische Gründe</em> oder <em>Sicherheitsgründe</em> (z. B. " +
      "Müdigkeitserkennung bei Pilot:innen). Eine Launen-/Stimmungsüberwachung " +
      "zur Leistungs- bzw. Verhaltenskontrolle ist davon laut den Leitlinien " +
      "der EU-Kommission nicht gedeckt – auch nicht mit Einwilligung.</li>" +
      "<li><strong>Das Verbot gilt bereits</strong>: Art. 5 ist gemäß Art. 113 " +
      "lit. a KI-VO seit dem <strong>2. Februar 2025</strong> anwendbar, ohne " +
      "Bestandsschutz für ältere Systeme.</li>" +
      "</ul>\n\n" +
      "<strong>Wichtige Abgrenzung:</strong> Verboten ist die " +
      "<em>Emotions-/Stimmungserkennung</em>. Eine reine, datenschutzkonforme " +
      "Erfassung der Arbeitstätigkeit <em>ohne</em> Ableitung von Emotionen " +
      "kann hingegen als <strong>Hochrisiko-KI-System</strong> nach Anhang III " +
      "Nr. 4 gelten und löst dann eigene Pflichten aus.\n\n" +
      "Für eine rechtssichere Einschätzung Ihres konkreten Vorhabens wenden Sie " +
      "sich bitte an das (menschliche) Team der <strong>KI-Servicestelle</strong>."
    );
  }
}
