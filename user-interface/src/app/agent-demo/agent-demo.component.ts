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
  // --- Input ---
  readonly exampleQuestion =
    "I would like to monitor the work activity of my employees with an AI " +
    "system. In particular, I would like to detect via webcam whether they " +
    "are in a bad mood. Am I allowed to do that?";
  userPrompt = "";
  submittedPrompt = "";
  maxLength = 500;

  // --- Run control ---
  running = false;
  done = false;
  entries: Entry[] = [];

  // --- Energy ---
  totalEnergy: EnergyData = this.emptyEnergy();
  totalProQuery = 0;

  readonly model = {
    llm: "mistral-small-3.2 (Q5_K_M)",
    embedding: "snowflake-arctic-embed-l-v2.0",
    reranker: "bge-reranker-v2-m3",
    host: "Self-hosting · Hetzner · NVIDIA RTX 4000 Ada (20 GB VRAM) · ollama",
    maxCallsPerTool: 3,
  };

  // Fixed list of the terms defined in the AI Act (above all Art. 3). If
  // definition() is called with a term that is not on the list, the tool
  // returns an error message together with this list.
  readonly definedTerms: string[] = [
    "AI system", "risk", "provider", "deployer", "authorised representative",
    "importer", "distributor", "operator", "placing on the market",
    "making available on the market", "putting into service", "intended purpose",
    "reasonably foreseeable misuse", "safety component",
    "instructions for use", "recall of an AI system", "withdrawal of an AI system",
    "performance of an AI system", "notifying authority", "conformity assessment",
    "conformity assessment body", "notified body", "substantial modification",
    "CE marking", "post-market monitoring system",
    "market surveillance authority", "harmonised standard", "common specification",
    "training data", "validation data", "validation data set", "testing data",
    "input data", "biometric data", "biometric identification",
    "biometric verification", "special categories of personal data",
    "sensitive operational data", "emotion recognition system",
    "biometric categorisation system", "remote biometric identification system",
    "real-time remote biometric identification system",
    "post-remote biometric identification system",
    "publicly accessible space", "law enforcement authority", "law enforcement",
    "AI Office", "national competent authority",
    "serious incident", "personal data", "non-personal data",
    "profiling", "real-world testing plan", "sandbox plan",
    "AI regulatory sandbox", "AI literacy", "testing in real-world conditions", "subject",
    "informed consent", "deep fake", "widespread infringement",
    "general-purpose AI model", "high-impact capabilities",
    "systemic risk", "general-purpose AI system",
    "floating-point operation", "downstream provider",
  ];

  // The tool definitions are passed along as part of the context on every LLM
  // call – which is why they are deliberately shown here as a component of the
  // system message.
  readonly tools: ToolSpec[] = [
    { name: "today", signature: "today()", description: "Returns today's date." },
    {
      name: "definition",
      signature: "definition(term: string)",
      description:
        "Returns the legal definition of a legal term (usually Art. 3 AI " +
        "Act), supplemented by plain-language clarifications from the " +
        "guidelines. A fixed list of terms is defined; for a term that is not " +
        "on that list the tool returns an error message together with the " +
        "list of all available terms.",
    },
    {
      name: "applicability",
      signature: "applicability(article_number: string)",
      description:
        "Explains in prose from when an article applies (including " +
        "exemptions and grandfathering clauses).",
    },
    {
      name: "lookup_article",
      signature: "lookup_article(article_number: string)",
      description:
        "Returns the plain text of an article, granular down to the first " +
        "level (e.g. “5-1” for Art. 5(1)).",
    },
    {
      name: "search_ai_act",
      signature: "search_ai_act(search_term: string)",
      description:
        "Semantic search across the AI Act, its recitals and the materials of " +
        "the AI Service Desk – including manually curated cross-references.",
    },
    {
      name: "search_guidelines",
      signature: "search_guidelines(reference_article?: string, question: string)",
      description:
        "Hybrid (lexical + semantic) search across the guidelines and " +
        "practical guides of the European Commission for a given article.",
    },
    {
      name: "ask_user",
      signature: "ask_user(question: string, suggested_answers?: string[])",
      description:
        "Asks the user a follow-up question and pauses the run. Suggested " +
        "answers can optionally be provided; the user picks one or types an " +
        "answer of their own. The return value is the selected or freely " +
        "entered answer as a string.",
    },
  ];

  // The page scrolls as a whole (only ONE scrollbar). As soon as viewers
  // scroll themselves, the automatic follow-along scrolling pauses. Scrolling
  // all the way back down resumes it automatically (usual chat behaviour).
  private userScrolled = false;

  // --- ask_user(): real interaction in the middle of the replay ---
  // The tool pauses the run until answerRueckfrage() resolves the promise.
  rueckfrageInput = "";
  private rueckfrageResolve: ((reply: string) => void) | null = null;
  private rueckfrageReply = "";

  answerRueckfrage(reply: string) {
    const r = reply.trim();
    if (!r || !this.rueckfrageResolve) return;
    this.rueckfrageResolve(r);
    this.rueckfrageResolve = null;
    this.rueckfrageInput = "";
  }

  constructor(private zone: NgZone) {
    // Only genuine user interactions count as “scrolled by hand” – the
    // programmatic scrollBy/scrollIntoView does not fire wheel/touchmove/keydown.
    const stop = () => (this.userScrolled = true);
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    window.addEventListener("keydown", (ev) => {
      if (
        ["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(ev.key)
      ) {
        this.userScrolled = true;
      }
    });
    // Back at the very bottom → follow along automatically again.
    window.addEventListener(
      "scroll",
      () => {
        if (
          window.innerHeight + window.scrollY >=
          document.documentElement.scrollHeight - 60
        ) {
          this.userScrolled = false;
        }
      },
      { passive: true }
    );
  }

  // ---------------------------------------------------------------------------
  // Energy
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
  // Display helpers
  // ---------------------------------------------------------------------------
  formatScore(score: number): string {
    return (score * 100).toFixed(1) + " %";
  }
  formatKWh(value: number): string {
    return value.toFixed(6);
  }
  formatTokens(n: number | undefined): string {
    return n === undefined ? "" : new Intl.NumberFormat("en-GB").format(n);
  }
  sourceKindLabel(kind: string): string {
    switch (kind) {
      case "ai_act": return "AI Act (binding)";
      case "erwaegungsgrund": return "Recital";
      case "ki-servicestelle": return "AI Service Desk";
      case "querverweis": return "Cross-reference";
      case "leitlinie": return "Guideline of the EU Commission";
      case "cop": return "Code of Practice";
      default: return kind;
    }
  }
  toggle(id: string) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("uk-hidden");
  }

  // The final answer card (LLM call without a further tool call) stays open
  // permanently and cannot be collapsed.
  isFinal(e: Entry): boolean {
    return e.kind === "llm" && !e.toolCall;
  }
  toggleCollapse(e: Entry) {
    if (!this.isFinal(e)) e.collapsed = !e.collapsed;
  }

  // Click on “reads XX messages”: open the history – if the card is collapsed,
  // expand it first (the target element only exists once rendered).
  openCtx(e: Entry) {
    const wasCollapsed = e.collapsed;
    e.collapsed = false;
    if (wasCollapsed) {
      setTimeout(() =>
        document.getElementById("ctx-" + e.id)?.classList.remove("uk-hidden")
      );
    } else {
      this.toggle("ctx-" + e.id);
    }
  }

  // Long error results (definition() with all 67 terms) are shown clipped in
  // the card – the model receives the full text unchanged (see the expandable
  // prompt history).
  resultClipped(e: Entry): boolean {
    return !!e.isError && (e.resultText || "").length > 300;
  }
  resultPreview(e: Entry): string {
    const t = e.resultText || "";
    return this.resultClipped(e) ? t.slice(0, 260) + " …" : t;
  }
  loadExample() {
    this.userPrompt = this.exampleQuestion;
  }

  // Icon per tool (Bootstrap Icons class, single colour) – makes the tool
  // cards distinguishable at a glance.
  toolIcon(name?: string): string {
    switch (name) {
      case "today": return "bi-calendar3";
      case "definition": return "bi-book";
      case "applicability": return "bi-hourglass-split";
      case "lookup_article": return "bi-journal-text";
      case "search_ai_act": return "bi-search";
      case "search_guidelines": return "bi-compass";
      case "ask_user": return "bi-chat-dots";
      default: return "bi-gear";
    }
  }
  // Speaking display name per tool – the technical function name
  // (e.g. search_ai_act()) only appears in small print as meta information.
  toolDisplayName(name?: string): string {
    switch (name) {
      case "today": return "Today's date";
      case "definition": return "Term definition";
      case "applicability": return "Temporal applicability";
      case "lookup_article": return "Look up article";
      case "search_ai_act": return "AI Act search";
      case "search_guidelines": return "Guidelines search";
      case "ask_user": return "Question for you";
      default: return name || "Tool";
    }
  }
  // Label of the central argument in the card header.
  toolArgLabel(name?: string): string {
    switch (name) {
      case "definition": return "Term";
      case "applicability": return "Article";
      case "lookup_article": return "Article";
      case "search_ai_act": return "Search term";
      case "search_guidelines": return "Search query";
      default: return "Argument";
    }
  }
  // Short official designation for article numbers – so that the card header
  // does not just read “5-1”. Deliberately terse (no title sentence as the
  // “argument”).
  private readonly articleTitles: Record<string, string> = {
    "5-1": "Art. 5(1) AI Act",
    "Article 5": "Art. 5 AI Act",
  };
  private articleTitle(nr: string): string {
    return this.articleTitles[nr] ?? nr;
  }
  // The central argument of a tool call (search term, term, article …) for the
  // prominent display in the card header. Article numbers are supplemented by
  // their official heading; several arguments are joined with “·”; if the JSON
  // is invalid the raw string is shown. ask_user shows its question
  // prominently in the interactive box – not a second time in the header.
  toolArg(e: Entry): string {
    if (!e.argsJson || e.toolName === "ask_user") return "";
    try {
      const args = JSON.parse(e.argsJson);
      return Object.entries(args)
        .filter(([, v]) => v !== null && v !== undefined && v !== "")
        .map(([k, v]) =>
          k === "article_number" ? this.articleTitle(String(v)) : String(v)
        )
        .join(" · ");
    } catch {
      return e.argsJson;
    }
  }

  // ---------------------------------------------------------------------------
  // The prompt history passed along (expandable on every LLM call)
  // ---------------------------------------------------------------------------
  // The history passed to an LLM call = all preceding entries.
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
        return "System prompt (+ tool definitions)";
      case "user":
        return this.truncate(e.userText || "");
      case "tool":
        return `${e.toolName}() → ${this.truncate(e.resultText || "", 80)}`;
      default:
        return e.toolCall
          ? `${this.truncate(e.reasoning || "", 64)} → ${e.toolCall.name}(…)`
          : "final answer";
    }
  }
  // Full content of a message (for expanding it in the history).
  entryFullText(e: Entry): string {
    switch (e.kind) {
      case "system":
        return e.systemText || "";
      case "user":
        return e.userText || "";
      case "tool":
        // Also show the arguments the tool was called with.
        return `→ called with: ${e.toolName}(${e.argsJson || ""})\n\n${e.resultText || ""}`;
      default: {
        const tail = e.toolCall
          ? `\n\n→ tool_call: ${e.toolCall.name}(${e.toolCall.argsJson})`
          : "\n\n→ final answer";
        return (e.reasoning || "") + tail;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Loop visualisation (loop diagram + sticky history bar)
  // ---------------------------------------------------------------------------
  // Which round of the agent loop is currently running (= number of LLM calls).
  get roundCount(): number {
    return this.entries.filter((e) => e.kind === "llm").length;
  }
  // The steps of the run as chips (only LLM and tool entries).
  get chips(): Entry[] {
    return this.entries.filter((e) => e.kind === "llm" || e.kind === "tool");
  }
  // Which node/edge of the loop diagram is currently active – derived entirely
  // from the message history.
  get loopActive(): { node: string; edge: string } {
    if (this.done) return { node: "antwort", edge: "nein" };
    const e = this.entries[this.entries.length - 1];
    if (!e || !this.running || e.status !== "running") return { node: "", edge: "" };
    if (e.kind === "tool") return { node: "tool", edge: "zurueck" };
    // System/user message: the first LLM call is currently being prepared.
    if (e.kind !== "llm") return { node: "llm", edge: "" };
    if (e.reasoningDone) {
      return e.toolCall
        ? { node: "llm", edge: "ja" }
        : { node: "antwort", edge: "nein" };
    }
    return { node: "llm", edge: "" };
  }
  chipLabel(e: Entry): string {
    if (e.kind === "llm") return e.toolCall ? "LLM" : "✓ Answer";
    return this.toolDisplayName(e.toolName) + (e.isError ? " ✗" : "");
  }
  chipClass(e: Entry): string {
    const cls: string[] = [];
    if (e.kind === "llm") cls.push(e.toolCall ? "chip-llm" : "chip-final");
    else cls.push("chip-tool");
    if (e.isError) cls.push("chip-error");
    if (e.status === "running") cls.push("chip-running");
    return cls.join(" ");
  }
  // Click on a chip: jump to the entry and expand it. This is a deliberate
  // navigation – after it, no more automatic follow-along scrolling.
  jumpTo(id: number) {
    this.userScrolled = true;
    const entry = this.entries.find((e) => e.id === id);
    if (entry) entry.collapsed = false;
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
      // Keep the bottom edge of the entry ~200px above the window edge so that
      // trailing (streamed) text is not cut off. Only scroll down – never jump
      // back up.
      const delta = el.getBoundingClientRect().bottom - (window.innerHeight - 200);
      if (delta > 0) window.scrollBy({ top: delta, behavior: "smooth" });
    }, 60);
  }

  // Streams a text word by word into a field (simulating the token output of
  // an LLM) and scrolls along with it.
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

  // Assembles the augmented prompt of the sub-RAG (search_ai_act): system
  // prompt + the sources used in full text + the search query as user prompt.
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
      `### Context (${used.length} sources, ranked order)\n` +
      context +
      "\n\n[INST]\nSearch query: " +
      (e.ragUserPrompt || "") +
      "\n[/INST]"
    );
  }

  // ---------------------------------------------------------------------------
  // Deterministic replay
  // ---------------------------------------------------------------------------
  private delay = (ms: number) =>
    new Promise<void>((resolve) => setTimeout(resolve, ms));

  async run() {
    if (this.running) return;
    this.entries = [];
    this.done = false;
    this.running = true;
    this.userScrolled = false;
    this.rueckfrageReply = "";
    this.rueckfrageInput = "";
    this.rueckfrageResolve = null;
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

      // Note and clear the (LLM) fields to be streamed up front so that the
      // full text does not flash up briefly during the “thinking time”. The
      // real answer to the follow-up question is only known at runtime and is
      // inserted here.
      const fullReasoning = entry.reasoning?.replace(
        "%USERANSWER%",
        this.rueckfrageReply
      );
      const streamAnswer = entry.kind === "llm" && !entry.toolCall && !!entry.answer;
      const fullAnswer = streamAnswer ? entry.answer : undefined;
      const streamResult =
        entry.kind === "tool" && !!entry.llmGenerates && !!entry.resultText;
      const fullResult = streamResult ? entry.resultText : undefined;
      if (entry.kind === "llm" && fullReasoning) entry.reasoning = "";
      if (streamAnswer) entry.answer = "";
      if (streamResult) entry.resultText = "";
      // Plain tools (without their own LLM call): the result only appears after
      // the execution time – not already while the loader is still running.
      const plainResult =
        entry.kind === "tool" && !entry.llmGenerates && !!entry.resultText;
      const fullPlainResult = plainResult ? entry.resultText : undefined;
      if (plainResult) entry.resultText = "";
      // Internal sub-steps (e.g. the retrieve pipeline) are revealed one after
      // another so that the order of the pipeline can be experienced.
      const fullSubSteps =
        entry.kind === "tool" && entry.subSteps && entry.subSteps.length > 0
          ? entry.subSteps
          : undefined;
      if (fullSubSteps) entry.subSteps = [];

      this.zone.run(() => this.entries.push(entry));
      this.scrollToEntry(entry.id);

      if (entry.kind === "system" || entry.kind === "user") {
        await this.delay(800);
      } else if (entry.kind === "tool" && entry.toolName === "ask_user") {
        // The run pauses until the user really answers – the duration of the
        // “tool” is the real waiting time for the human.
        await this.delay(500);
        const waitStart = Date.now();
        this.zone.run(() => (entry.awaitingUser = true));
        this.scrollToEntry(entry.id, "end");
        const reply = await new Promise<string>(
          (resolve) => (this.rueckfrageResolve = resolve)
        );
        this.rueckfrageReply = reply;
        this.zone.run(() => {
          entry.awaitingUser = false;
          entry.resultText = reply;
          if (entry.energy) {
            entry.energy.duration =
              Math.round((Date.now() - waitStart) / 100) / 10;
          }
        });
        this.scrollToEntry(entry.id, "end");
        await this.delay(600);
      } else if (entry.kind === "tool") {
        if (fullSubSteps) {
          // Reveal the pipeline sub-steps one by one (retrieve → … → reranking).
          await this.delay(600);
          for (const sub of fullSubSteps) {
            this.zone.run(() => entry.subSteps!.push(sub));
            this.scrollToEntry(entry.id, "end");
            await this.delay(750);
          }
          await this.delay(400);
        } else {
          await this.delay(1100); // execution time of the tool
        }
      } else if (this.roundCount === 1) {
        // Deliberately wait longer before the very first LLM round so that
        // viewers have time to read the loop diagram.
        await this.delay(4500);
      } else {
        await this.delay(900); // “thinking time” before the reasoning
      }

      if (entry.kind === "llm") {
        // The reasoning is LLM output → stream it token by token
        // (with a short pause beforehand: “time to first token”).
        if (fullReasoning) {
          await this.delay(700);
          await this.streamInto(entry, (v) => (entry.reasoning = v), fullReasoning);
        }
        this.zone.run(() => (entry.reasoningDone = true));
        this.scrollToEntry(entry.id, "end");

        // Stream the final answer (after the reasoning, if no tool call follows).
        if (fullAnswer !== undefined) {
          await this.delay(300);
          await this.streamInto(entry, (v) => (entry.answer = v), fullAnswer, 44);
        } else if (entry.toolCall) {
          await this.delay(350); // short pause, then the tool call is emitted
        }
      } else if (fullResult !== undefined) {
        // Tools with their own LLM call: stream the result token by token.
        await this.streamInto(entry, (v) => (entry.resultText = v), fullResult);
      } else if (fullPlainResult !== undefined) {
        // No LLM involved: the result appears all at once.
        this.zone.run(() => (entry.resultText = fullPlainResult));
        this.scrollToEntry(entry.id, "end");
      }

      this.zone.run(() => {
        entry.status = "done";
        // Calmer scrolling behaviour: cards are not collapsed immediately after
        // they finish themselves, but only once the NEXT assistant step has
        // finished. (The system prompt has its own toggle; the assistant step
        // that finished last stays open.)
        if (entry.kind === "llm") {
          for (const p of this.entries) {
            if (p.id < entry.id && p.kind !== "system") p.collapsed = true;
          }
        }
        this.accumulateEnergy();
      });
    }

    this.zone.run(() => {
      this.running = false;
      this.done = true;
    });
  }

  // ---------------------------------------------------------------------------
  // Hard-coded message history for the example question.
  // 9 LLM calls (8 with a tool call, 1 with the final answer) + 8 tool results.
  // The energy of the LLM calls grows with the growing context (prefill).
  // ask_user() is the only non-deterministic step: the run pauses there for a
  // real input; the answer is inserted into the following reasoning texts via
  // the placeholder %USERANSWER%.
  // ---------------------------------------------------------------------------
  private buildScript(): Partial<Entry>[] {
    // today() returns the real current date – even in the deterministic replay.
    const todaysDate = new Intl.DateTimeFormat("en-GB", {
      dateStyle: "full",
    }).format(new Date());
    return [
      // System message (including the tool definitions sent along)
      {
        kind: "system",
        systemText:
          "You are the AI Act assistant of the AI Service Desk (RTR). Answer " +
          "questions on the EU AI Act exclusively on the basis of the sources " +
          "retrieved via the tools.\n\n" +
          "Rules:\n" +
          "1. Plan which tools you need and call them step by step " +
          "(max. 3× per tool).\n" +
          "2. Base every statement on a source. Do not invent articles, " +
          "paragraph numbers or content.\n" +
          "3. Binding provisions (AI Act) take precedence; guidelines/" +
          "practical guides may only supplement them.\n" +
          "4. For questions on temporal applicability always use applicability().\n" +
          "5. If information that is material to the answer is missing, ask " +
          "exactly one follow-up question with ask_user() – with suggested " +
          "answers where possible.\n" +
          "6. Make your reasoning explicit and list all sources at the end.",
      },

      // User question
      { kind: "user", userText: this.exampleQuestion },

      // --- LLM call 1: always starts with the broad semantic search ---
      {
        kind: "llm",
        contextMessages: 2,
        contextTokens: 712,
        reasoning:
          "I do not yet know the relevant provisions. As with practically " +
          "every question, I start with a broad semantic search in the AI Act " +
          "in order to see which articles and recitals are relevant at all. " +
          "(Only if someone explicitly asks about a particular article would I " +
          "look that one up directly.) As the search term I take the core " +
          "concepts of the question.",
        toolCall: {
          name: "search_ai_act",
          argsJson:
            '{ "search_term": "monitor employees with AI via webcam detect mood workplace" }',
        },
        energy: this.energy(0.000016, 0.000058, 0.000004, 1.5),
      },
      {
        kind: "tool",
        toolName: "search_ai_act",
        callIndex: 1,
        argsJson:
          '{ "search_term": "monitor employees with AI via webcam detect mood workplace" }',
        rerankerApplied: true,
        ragView: true,
        llmGenerates: true,
        ragUserPrompt:
          "monitor employees with AI via webcam detect mood workplace",
        ragSystemPrompt:
          "You are the semantic search of the AI Service Desk across the AI " +
          "Act, its recitals and the materials of the AI Service Desk. Answer " +
          "the search query exclusively from the sources below and cite the " +
          "source for every statement. Do not make a legal assessment of your " +
          "own – only summarise the relevant provisions.",
        subSteps: [
          {
            label: "Retrieve · ① Embedding of the search query",
            detail:
              "The search term is turned into a vector with " +
              "snowflake-arctic-embed-l-v2.0.",
          },
          {
            label: "Retrieve · ② Semantic vector search",
            detail:
              "Matching against the corpus → 38 candidates by cosine similarity, " +
              "capped at the 25 strongest or 15,000 tokens.",
          },
          {
            label: "Retrieve · ③ Add cross-references",
            detail:
              "Manually curated cross-references between provisions are loaded " +
              "in addition (here: Art. 26 from Annex III point 4).",
          },
          {
            label: "Retrieve · ④ Reranking",
            detail:
              "bge-reranker-v2-m3 re-scores every candidate against the query " +
              "and produces the final order; duplicates and matches of low " +
              "relevance are discarded.",
          },
        ],
        sources: this.searchAiActSources(),
        // “Generate” of the sub-RAG: source-bound passage that goes back to the
        // agent (= raw tool output, not a summary added later by the UI).
        resultText:
          "In the workplace, inferring emotions by means of AI systems is " +
          "prohibited (Art. 5(1)(f) AI Act); exceptions exist only for medical " +
          "or safety reasons. Recital 44 justifies the prohibition with the " +
          "imbalance of power and the limited scientific reliability of such " +
          "systems. In addition, AI systems for monitoring/evaluating work " +
          "performance may – as far as they are not prohibited – qualify as " +
          "high-risk AI systems under Annex III point 4. " +
          "[Sources: Art. 5(1)(f); Recital 44; Annex III point 4]",
        energy: this.energy(0.000071, 0.001034, 0.000021, 6.2),
      },

      // --- LLM call 2: recognises the emotion topic from the matches and
      // fetches the definition – at first with the colloquial (wrong) term ---
      {
        kind: "llm",
        contextMessages: 4,
        contextTokens: 936,
        reasoning:
          "The search brings it to the surface: at the centre is Art. 5(1)(f) " +
          "– the inference of emotions in the workplace. What I am planning " +
          "(detecting the “mood” via the webcam) sounds like “emotion " +
          "recognition”. Whether it falls under the prohibition depends on the " +
          "exact legal term – I will fetch the definition.",
        toolCall: {
          name: "definition",
          argsJson: '{ "term": "emotion recognition" }',
        },
        energy: this.energy(0.000017, 0.000064, 0.000005, 1.7),
      },
      // The first call fails: the term is not on the list. As specified, the
      // tool responds with an error plus the list of terms, so that the model
      // can correct itself.
      {
        kind: "tool",
        toolName: "definition",
        callIndex: 1,
        isError: true,
        argsJson: '{ "term": "emotion recognition" }',
        resultText:
          "ERROR: No legal definition is on file for “emotion " +
          "recognition”.\n\nAvailable terms (" +
          this.definedTerms.length +
          "): " +
          this.definedTerms.join(", "),
        energy: this.energy(0.000005, 0.0, 0.000002, 0.2),
      },

      // --- LLM call 3: self-correction based on the error message ---
      {
        kind: "llm",
        contextMessages: 6,
        contextTokens: 1290,
        reasoning:
          "The call failed – “emotion recognition” is not on the list of " +
          "terms. The error message does, however, come with all defined " +
          "terms: the applicable term is “emotion recognition system” " +
          "(Art. 3). I repeat the call with the correct term.",
        toolCall: {
          name: "definition",
          argsJson: '{ "term": "emotion recognition system" }',
        },
        energy: this.energy(0.000019, 0.000073, 0.000005, 1.9),
      },
      {
        kind: "tool",
        toolName: "definition",
        callIndex: 2,
        argsJson: '{ "term": "emotion recognition system" }',
        resultText:
          "Art. 3(39) AI Act: “emotion recognition system” means an AI system " +
          "for the purpose of identifying or inferring emotions or intentions " +
          "of natural persons on the basis of their biometric data.\n\n" +
          "Supplement (guidelines on prohibited practices): The mere detection " +
          "of readily apparent expressions (e.g. a smile) or physical states " +
          "is NOT covered. What is covered is the inference of inner emotional " +
          "states such as satisfaction, stress, fatigue or “being in a bad " +
          "mood”.",
        energy: this.energy(0.000009, 0.000005, 0.000002, 0.6),
      },

      // --- LLM call 4: looks up the exact wording of Art. 5(1)(f) ---
      {
        kind: "llm",
        contextMessages: 8,
        contextTokens: 1428,
        reasoning:
          "The definition (Art. 3(39)) confirms it: inferring inner states " +
          "such as “a bad mood” is covered, mere detection of facial " +
          "expressions is not. So my case is covered. Now I need the exact " +
          "wording of Art. 5(1)(f) – above all the exceptions.",
        toolCall: {
          name: "lookup_article",
          argsJson: '{ "article_number": "5-1" }',
        },
        energy: this.energy(0.00002, 0.000079, 0.000006, 2.1),
      },
      {
        kind: "tool",
        toolName: "lookup_article",
        callIndex: 1,
        argsJson: '{ "article_number": "5-1" }',
        resultText:
          "Art. 5(1) AI Act – The following AI practices shall be prohibited: " +
          "[…]\n\n" +
          "(f) the placing on the market, the putting into service for this " +
          "specific purpose, or the use of AI systems to infer emotions of a " +
          "natural person in the areas of workplace and education " +
          "institutions, except where the use of the AI system is intended to " +
          "be put in place or into the market for medical or safety reasons.",
        energy: this.energy(0.000006, 0.0, 0.000002, 0.3),
      },

      // --- LLM call 5: calls search_guidelines() ---
      {
        kind: "llm",
        contextMessages: 10,
        contextTokens: 1641,
        reasoning:
          "There is only the narrow exception of “medical or safety reasons”. " +
          "How it is to be interpreted is clarified by the Commission's " +
          "guidelines – these are not in the general search pool but are " +
          "searched via a tool of their own.",
        toolCall: {
          name: "search_guidelines",
          argsJson:
            '{ "reference_article": "Article 5", "question": "Is inferring the ' +
            'mood of employees via a webcam at the workplace prohibited and ' +
            'does an exception apply?" }',
        },
        energy: this.energy(0.000022, 0.000086, 0.000006, 2.3),
      },
      {
        kind: "tool",
        toolName: "search_guidelines",
        callIndex: 1,
        argsJson:
          '{ "reference_article": "Article 5", "question": "Is inferring the ' +
          'mood of employees via a webcam at the workplace prohibited and ' +
          'does an exception apply?" }',
        rerankerApplied: true,
        llmGenerates: true,
        ragUserPrompt:
          "Is inferring the mood of employees via a webcam at the workplace " +
          "prohibited and does an exception apply?",
        ragSystemPrompt:
          "You search the guidelines and practical guides of the European " +
          "Commission on the reference article and answer the question " +
          "exclusively from the passages below. Name the source and do not " +
          "make a legal assessment of your own – only report what the " +
          "guidelines say.",
        subSteps: [
          {
            label: "1. Lexical search (BM25)",
            detail:
              "The search terms do not come from the model but are extracted " +
              "from the “question” parameter (tokenisation, stop-word removal, " +
              "lemmatisation) → “workplace”, “emotion”, “mood”, “exception”, " +
              "“safety”. These are matched against the guidelines index; " +
              "strongest match: section “Emotion recognition”.",
          },
          {
            label: "2. Semantic search (embedding)",
            detail:
              "snowflake-arctic-embed-l-v2.0 confirms the same source and " +
              "additionally finds the “CoP on transparency obligations” (low score).",
          },
          {
            label: "3. Hybrid fusion + reranker",
            detail:
              "Reciprocal rank fusion combines both lists, bge-reranker-v2-m3 " +
              "produces the final order; the CoP match drops out due to low relevance.",
          },
        ],
        sources: this.searchLeitlinienSources(),
        // “Generate” of the tool: source-bound passage from the guidelines.
        resultText:
          "The exceptions of “medical reasons” and “safety reasons” are to be " +
          "interpreted narrowly. As a safety reason the Commission names, for " +
          "example, fatigue detection for pilots or professional drivers. " +
          "Explicitly NOT covered is the monitoring of satisfaction, mood, " +
          "engagement or productivity of employees – it remains prohibited " +
          "under Art. 5(1)(f), even with consent. The mere detection of " +
          "readily apparent expressions, on the other hand, does not fall " +
          "under the prohibition. [Source: guidelines on prohibited practices, " +
          "section on emotion recognition]",
        energy: this.energy(0.000061, 0.000874, 0.000018, 5.4),
      },

      // --- LLM call 6: asks the user a genuine follow-up question ---
      {
        kind: "llm",
        contextMessages: 12,
        contextTokens: 1868,
        reasoning:
          "The guidelines confirm it: monitoring mood or temper in order to " +
          "control employees is not covered by the exception. That leaves the " +
          "temporal applicability – and for that it matters whether the system " +
          "is already running or is only about to be introduced (keyword: " +
          "transitional/grandfathering rules). That is not stated in the " +
          "question. According to the system prompt I may ask for missing, " +
          "material information with ask_user() – I will provide two suggested " +
          "answers.",
        toolCall: {
          name: "ask_user",
          argsJson:
            '{ "question": "Is the AI system already in use or are you only ' +
            'planning to introduce it?", "suggested_answers": ["Already in ' +
            'use", "Introduction only planned"] }',
        },
        energy: this.energy(0.000023, 0.00009, 0.000007, 2.4),
      },
      // ask_user() pauses the run: the result (resultText) is the real answer
      // of the user, the duration is the real waiting time.
      {
        kind: "tool",
        toolName: "ask_user",
        callIndex: 1,
        argsJson:
          '{ "question": "Is the AI system already in use or are you only ' +
          'planning to introduce it?", "suggested_answers": ["Already in ' +
          'use", "Introduction only planned"] }',
        askQuestion:
          "Is the AI system already in use or are you only planning to " +
          "introduce it?",
        askOptions: ["Already in use", "Introduction only planned"],
        energy: this.energy(0.000001, 0.0, 0.000001, 0),
      },

      // --- LLM call 7: fetches today's date (LLMs do not know it reliably) ---
      {
        kind: "llm",
        contextMessages: 14,
        contextTokens: 1943,
        reasoning:
          "The user replies: “%USERANSWER%”. I will now check the temporal " +
          "applicability of the prohibition – including the question of " +
          "whether there are transitional rules for systems already running. " +
          "First I need today's date: my training cut-off lies in the past, I " +
          "do not know it reliably – I call today().",
        toolCall: {
          name: "today",
          argsJson: "{}",
        },
        energy: this.energy(0.000024, 0.000092, 0.000007, 2.5),
      },
      // today() is a trivial tool without an LLM: instant result, energy
      // consumption practically zero.
      {
        kind: "tool",
        toolName: "today",
        callIndex: 1,
        argsJson: "{}",
        resultText: "Today is " + todaysDate + ".",
        energy: this.energy(0.000001, 0.0, 0.000001, 0.1),
      },

      // --- LLM call 8: calls applicability() ---
      {
        kind: "llm",
        contextMessages: 16,
        contextTokens: 2001,
        reasoning:
          "today() returns: " +
          todaysDate +
          ". Now I check since when the prohibition in Art. 5 applies – for " +
          "questions on temporal applicability the system prompt tells me to " +
          "always use applicability().",
        toolCall: {
          name: "applicability",
          argsJson: '{ "article_number": "Article 5" }',
        },
        energy: this.energy(0.000025, 0.000096, 0.000007, 2.6),
      },
      {
        kind: "tool",
        toolName: "applicability",
        callIndex: 1,
        argsJson: '{ "article_number": "Article 5" }',
        resultText:
          "Under Art. 113(a) AI Act the prohibition of certain practices in " +
          "Article 5 has applied since 2 February 2025 already – earlier than " +
          "the bulk of the regulation.\n\n" +
          "For Art. 5 there is NO transitional or grandfathering rule: since " +
          "that date, prohibited systems may neither be placed on the market " +
          "nor put into service nor used – not even if they were already in " +
          "use beforehand.",
        energy: this.energy(0.000007, 0.0, 0.000002, 0.4),
      },

      // --- LLM call 9: final answer (no further tool call) ---
      {
        kind: "llm",
        contextMessages: 18,
        contextTokens: 2209,
        reasoning:
          "All the building blocks are in place: emotion recognition system " +
          "(Art. 3(39)), prohibited in the workplace under Art. 5(1)(f), no " +
          "exception applies, applicable since 2 February 2025 without " +
          "grandfathering – so in the end the answer to the follow-up question " +
          "(“%USERANSWER%”) does not matter: the prohibition applies today in " +
          "both cases. I will phrase the answer strictly source-bound.",
        answer: this.finalAnswer(),
        citations: [
          { label: "Art. 5(1)(f) AI Act", kind: "ai_act" },
          { label: "Art. 3(39) AI Act", kind: "ai_act" },
          { label: "Recital 44 AI Act", kind: "erwaegungsgrund" },
          { label: "Art. 113(a) AI Act", kind: "ai_act" },
          {
            label: "Guidelines on prohibited practices (emotion recognition)",
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
        title: "Art. 5(1)(f) AI Act – prohibition of emotion recognition in the workplace",
        kind: "ai_act",
        snippet:
          "Inferring emotions in the workplace and in education institutions is prohibited, except for medical or safety reasons.",
        cosine: 0.712,
        rankBefore: 3,
        rankAfter: 1,
        rerankScore: 0.94,
        used: true,
      },
      {
        id: "erwg-44",
        title: "Recital 44 AI Act – rationale of the prohibition",
        kind: "erwaegungsgrund",
        snippet:
          "Serious concerns about emotion recognition in the workplace because of the imbalance of power and the limited scientific reliability.",
        cosine: 0.685,
        rankBefore: 2,
        rankAfter: 2,
        rerankScore: 0.88,
        used: true,
      },
      {
        id: "art5-1-a",
        title: "Art. 5(1)(a) AI Act – manipulative/deceptive techniques",
        kind: "ai_act",
        snippet:
          "Prohibition of AI systems that materially distort behaviour through subliminal techniques.",
        cosine: 0.731,
        rankBefore: 1,
        rankAfter: 3,
        rerankScore: 0.41,
        used: true,
      },
      {
        id: "annex3-4",
        title: "Annex III point 4 AI Act – employment & workers management (high risk)",
        kind: "ai_act",
        snippet:
          "AI systems for monitoring and evaluating work performance qualify – as far as they are not prohibited – as high-risk AI systems.",
        cosine: 0.664,
        rankBefore: 4,
        rankAfter: 4,
        rerankScore: 0.57,
        used: true,
      },
      {
        id: "art26",
        title: "Cross-reference: Art. 26 AI Act – obligations of deployers of high-risk AI",
        kind: "querverweis",
        snippet:
          "Manually curated cross-reference from Annex III point 4. Relevant only for monitoring systems that are not prohibited.",
        cosine: 0.602,
        rankBefore: 6,
        rankAfter: 5,
        rerankScore: 0.33,
        used: false,
        skipReason: "low_relevance",
      },
      {
        id: "erwg-44-dup",
        title: "Recital 44 AI Act (AI Service Desk materials, paraphrase)",
        kind: "ki-servicestelle",
        snippet: "Identical in substance to Recital 44 – discarded as a duplicate.",
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
          "Guidelines on prohibited practices – section “Emotion recognition” (workplace)",
        kind: "leitlinie",
        snippet:
          "Narrow interpretation of the exceptions; fatigue detection (safety) permitted, mood monitoring of employees prohibited.",
        cosine: 0.821,
        rankBefore: 1,
        rankAfter: 1,
        rerankScore: 0.96,
        used: true,
      },
      {
        id: "ll-prohibited-scope",
        title:
          "Guidelines on prohibited practices – emotion vs. mere detection of facial expressions",
        kind: "leitlinie",
        snippet:
          "Detecting readily apparent expressions does not fall under the prohibition; inferring inner states does.",
        cosine: 0.768,
        rankBefore: 3,
        rankAfter: 2,
        rerankScore: 0.79,
        used: true,
      },
      {
        id: "cop-transparency",
        title: "CoP on transparency obligations – labelling of emotion recognition",
        kind: "cop",
        snippet:
          "Concerns disclosure obligations under Art. 50, not the prohibition itself – not relevant for this question.",
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
      "<strong>No – this is prohibited under the EU AI Act.</strong> An AI " +
      "system that uses the webcam to detect the mood of your employees " +
      "(“being in a bad mood”) falls under the <strong>prohibited " +
      "practices</strong> and must not be used.\n\n" +
      "<strong>Why?</strong>\n\n" +
      "<ul>" +
      "<li><strong>It is an emotion recognition system</strong> (Art. 3(39) " +
      "AI Act): an inner emotional state is inferred from the facial " +
      "expression. Merely detecting a facial expression would be " +
      "unproblematic – inferring the “mood” from it is not.</li>" +
      "<li><strong>Use in the workplace is explicitly prohibited</strong> " +
      "(Art. 5(1)(f) AI Act). Recital 44 justifies this among other things " +
      "with the imbalance of power and the questionable scientific " +
      "reliability of such systems.</li>" +
      "<li><strong>No exception applies</strong>: only <em>medical " +
      "reasons</em> or <em>safety reasons</em> would be permitted (e.g. " +
      "fatigue detection for pilots). According to the guidelines of the " +
      "European Commission, monitoring mood or temper in order to control " +
      "performance or behaviour is not covered by this – not even with " +
      "consent.</li>" +
      "<li><strong>The prohibition already applies</strong>: under " +
      "Art. 113(a) AI Act, Art. 5 has been applicable since " +
      "<strong>2 February 2025</strong>, with no grandfathering for older " +
      "systems.</li>" +
      "</ul>\n\n" +
      "<strong>An important distinction:</strong> what is prohibited is " +
      "<em>emotion or mood recognition</em>. A purely data-protection-compliant " +
      "recording of work activity <em>without</em> inferring emotions may, by " +
      "contrast, qualify as a <strong>high-risk AI system</strong> under " +
      "Annex III point 4 and then triggers obligations of its own.\n\n" +
      "For a legally sound assessment of your specific plans, please contact " +
      "the (human) team of the <strong>AI Service Desk</strong>."
    );
  }
}
