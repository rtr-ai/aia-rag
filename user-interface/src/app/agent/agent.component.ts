import { Component, Inject, LOCALE_ID, NgZone } from "@angular/core";
import { NgClass } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { environment } from "../../environments/environment";
import {
  AgentEvent,
  EnergyData,
  Entry,
  Limits,
  LlmMessage,
  NestedLlmCall,
  ToolSpec,
} from "./agent-models";

@Component({
  selector: "app-agent",
  imports: [NgClass, FormsModule],
  templateUrl: "./agent.component.html",
  styleUrl: "./agent.component.scss",
})
export class AgentComponent {
  readonly exampleQuestion =
    $localize`:@@agentExampleQuestion:Gilt das Verbot der Emotionserkennung am Arbeitsplatz (Art. 5 AI Act) heute schon?`;
  userPrompt = "";
  submittedPrompt = "";
  maxLength = 500;

  running = false;
  done = false;
  backendAvailable = true;
  errorMessage = "";
  maxTurnsReached = false;
  entries: Entry[] = [];
  tools: ToolSpec[] = [];
  toolDisplayNames: Record<string, string> = {};
  limits: Limits | null = null;
  totalEnergy: EnergyData | null = null;

  private nextId = 0;
  private deltaCount = 0;
  private userScrolled = false;
  private readonly errorGeneric = $localize`:@@agentErrorGeneric:Bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten.`;
  private readonly errorRecoverable = $localize`:@@agentErrorRecoverable:Das Sprachmodell ist derzeit ausgelastet. Bitte versuchen Sie es später erneut.`;

  constructor(
    private zone: NgZone,
    @Inject(LOCALE_ID) public locale: string
  ) {
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

  toolCallCount(tool: string): number {
    return this.entries.filter((e) => e.kind === "tool" && e.toolName === tool)
      .length;
  }
  formatTokens(n: number | undefined): string {
    return n === undefined ? "" : new Intl.NumberFormat("de-DE").format(n);
  }
  formatKWh(value: number): string {
    return value.toFixed(6).replace(".", ",");
  }
  formatSeconds(s: number): string {
    return s.toFixed(1).replace(".", ",");
  }
  formatScore(score: number): string {
    return score.toFixed(3).replace(".", ",");
  }
  skipReasonLabel(reason: string): string {
    switch (reason) {
      case "duplicate":
        return $localize`:@@agentSourcesSkipDuplicate:Dublette`;
      case "context_window":
        return $localize`:@@agentSourcesSkipContextWindow:Kontextfenster voll`;
      default:
        return $localize`:@@agentSourcesSkipGeneric:übersprungen`;
    }
  }
  ragNestedCall(e: Entry): NestedLlmCall | undefined {
    if (e.toolName !== "search_ai_act") return undefined;
    return e.nestedCalls?.[e.nestedCalls.length - 1];
  }
  ragResultText(e: Entry): string {
    if (e.resultText !== undefined) return e.resultText;
    return this.ragNestedCall(e)?.content ?? "";
  }
  toggle(id: string) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("uk-hidden");
  }

  isFinal(e: Entry): boolean {
    return e.kind === "llm" && !e.toolCalls?.length;
  }
  toggleCollapse(e: Entry) {
    if (!this.isFinal(e)) e.collapsed = !e.collapsed;
  }
  loadExample() {
    this.userPrompt = this.exampleQuestion;
  }

  toolIcon(name?: string): string {
    switch (name) {
      case "current_date":
        return "bi-calendar3";
      case "search_ai_act":
        return "bi-search";
      default:
        return "bi-gear";
    }
  }
  toolDisplayName(name?: string): string {
    return name ? this.toolDisplayNames[name] ?? name : "";
  }
  toolArgLabel(name?: string): string {
    switch (name) {
      case "current_date":
        return $localize`:@@agentToolArgTimezone:Zeitzone`;
      case "search_ai_act":
        return $localize`:@@agentToolArgQuery:Suchbegriff`;
      default:
        return $localize`:@@agentToolArgDefault:Argument`;
    }
  }
  toolArg(e: Entry): string {
    if (e.args) {
      return Object.values(e.args)
        .filter((v) => v !== null && v !== undefined && v !== "")
        .map((v) => String(v))
        .join(" · ");
    }
    return e.argsJson ?? "";
  }
  toolSignature(tool: ToolSpec): string {
    const props = tool.parameters.properties ?? {};
    const required = tool.parameters.required ?? [];
    const params = Object.entries(props).map(([name, prop]) =>
      required.includes(name)
        ? `${name}: ${prop.type ?? "any"}`
        : `${name}?: ${prop.type ?? "any"}`
    );
    return `${tool.name}(${params.join(", ")})`;
  }

  private truncate(s: string, n = 90): string {
    const t = s.replace(/\s+/g, " ").trim();
    return t.length > n ? t.slice(0, n) + " …" : t;
  }
  messageSummary(m: LlmMessage): string {
    switch (m.role) {
      case "system":
        return $localize`:@@agentCtxSummarySystem:System-Prompt (+ Tool-Definitionen)`;
      case "user":
        return this.truncate(m.content);
      case "tool":
        return `${m.tool_name}() → ${this.truncate(m.content, 80)}`;
      default: {
        if (!m.tool_calls.length) {
          return this.truncate(m.content);
        }
        const names = m.tool_calls.map((tc) => tc.name).join(", ");
        return `${this.truncate(m.reasoning || m.content, 64)} → ${names}(…)`;
      }
    }
  }
  messageFullText(m: LlmMessage): string {
    if (m.role === "assistant") {
      let text = m.reasoning ? m.reasoning + "\n\n" : "";
      text += m.content;
      for (const tc of m.tool_calls) {
        text += `\n\n→ tool_call: ${tc.name}(${tc.raw_arguments})`;
      }
      return text.trim();
    }
    return m.content;
  }

  get roundCount(): number {
    return this.entries.filter((e) => e.kind === "llm").length;
  }
  get chips(): Entry[] {
    return this.entries.filter((e) => e.kind === "llm" || e.kind === "tool");
  }
  get loopActive(): { node: string; edge: string } {
    if (this.done) return { node: "antwort", edge: "nein" };
    const e = this.entries[this.entries.length - 1];
    if (!e || !this.running || e.status !== "running") return { node: "", edge: "" };
    if (e.kind === "tool") return { node: "tool", edge: "zurueck" };
    if (e.kind !== "llm") return { node: "llm", edge: "" };
    if (e.toolCalls?.length) return { node: "llm", edge: "ja" };
    if (e.content) return { node: "antwort", edge: "nein" };
    return { node: "llm", edge: "" };
  }
  chipLabel(e: Entry): string {
    if (e.kind === "llm") {
      return e.toolCalls?.length ? "LLM" : $localize`:@@agentChipFinalAnswer:✓ Antwort`;
    }
    return this.toolDisplayName(e.toolName) + (e.isError ? " ✗" : "");
  }
  chipClass(e: Entry): string {
    const cls: string[] = [];
    if (e.kind === "llm") cls.push(e.toolCalls?.length ? "chip-llm" : "chip-final");
    else cls.push("chip-tool");
    if (e.isError) cls.push("chip-error");
    if (e.status === "running") cls.push("chip-running");
    return cls.join(" ");
  }
  jumpTo(id: number) {
    this.userScrolled = true;
    const entry = this.entries.find((e) => e.id === id);
    if (entry) entry.collapsed = false;
    document
      .getElementById("entry-" + id)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  private scrollToEntry(id: number) {
    if (this.userScrolled) return;
    setTimeout(() => {
      if (this.userScrolled) return;
      const el = document.getElementById("entry-" + id);
      if (!el) return;
      const delta = el.getBoundingClientRect().bottom - (window.innerHeight - 200);
      if (delta > 0) window.scrollBy({ top: delta, behavior: "smooth" });
    }, 60);
  }

  async run() {
    if (this.running) return;
    this.entries = [];
    this.nextId = 0;
    this.deltaCount = 0;
    this.done = false;
    this.running = true;
    this.userScrolled = false;
    this.errorMessage = "";
    this.maxTurnsReached = false;
    this.tools = [];
    this.toolDisplayNames = {};
    this.limits = null;
    this.totalEnergy = null;
    this.submittedPrompt = this.userPrompt;

    setTimeout(
      () =>
        document
          .getElementById("agentRun")
          ?.scrollIntoView({ behavior: "smooth" }),
      150
    );

    try {
      await fetchEventSource(`${environment.LLM_ENDPOINT}/agent`, {
        method: "POST",
        openWhenHidden: true,
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          prompt: this.userPrompt,
          language: this.locale === "en" ? "en" : "de",
        }),
        onopen: (response: Response): Promise<void> => {
          if (response.ok && response.status === 200) {
            return Promise.resolve();
          } else if (
            response.status >= 400 &&
            response.status < 500 &&
            response.status !== 429
          ) {
            console.error("Client-Side Error opening LLM Stream", response);
          }
          throw new Error("Error opening LLM Stream");
        },
        onerror: () => {
          this.onErrorHappened();
          throw new Error();
        },
        onmessage: (event: { data: string }) => {
          if (!event.data || event.data.length === 0) {
            return;
          }
          try {
            const ev = JSON.parse(event.data) as AgentEvent;
            this.zone.run(() => this.handleEvent(ev));
          } catch (e: any) {
            console.error("Unable to parse JSON", e);
            console.log("Received data", event.data);
            this.onErrorHappened();
          }
        },
        onclose: () => {
          this.zone.run(() => {
            if (this.running) {
              this.running = false;
              this.errorMessage = this.errorGeneric;
            }
          });
        },
      });
    } catch {
      this.onErrorHappened();
    }
  }

  private onErrorHappened() {
    this.running = false;
    this.backendAvailable = false;
    setTimeout(() => (this.backendAvailable = true), 12000);
  }

  private handleEvent(ev: AgentEvent) {
    if (ev.parent_step_id !== null) {
      this.handleNestedEvent(ev);
      return;
    }
    switch (ev.type) {
      case "run_started":
        this.tools = ev.payload.tools;
        this.toolDisplayNames = ev.payload.tool_display_names;
        this.limits = ev.payload.limits;
        break;
      case "system_prompt":
        this.addEntry({ kind: "system", status: "done", systemText: ev.payload.text });
        break;
      case "user_message":
        this.addEntry({ kind: "user", status: "done", userText: ev.payload.text });
        break;
      case "turn_started":
        this.addEntry({
          kind: "llm",
          status: "running",
          stepId: ev.step_id,
          contextMessages: ev.payload.message_count,
          contextTokens: ev.payload.context_tokens,
          reasoning: "",
          content: "",
          toolCalls: [],
        });
        break;
      case "llm_request": {
        const entry = this.entryFor(ev.step_id);
        if (entry) entry.messages = ev.payload.messages;
        break;
      }
      case "reasoning_delta":
      case "content_delta": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        if (ev.type === "reasoning_delta") {
          entry.reasoning += ev.payload.text;
        } else {
          entry.content += ev.payload.text;
        }
        this.deltaCount++;
        if (this.deltaCount % 8 === 0) {
          this.scrollToEntry(entry.id);
        }
        break;
      }
      case "tool_call": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        entry.toolCalls!.push(ev.payload);
        this.scrollToEntry(entry.id);
        break;
      }
      case "llm_finished": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        entry.status = "done";
        entry.energy = ev.payload.energy;
        entry.duration = ev.payload.duration;
        entry.ttft = ev.payload.time_to_first_token;
        entry.generationDuration = ev.payload.generation_duration;
        for (const p of this.entries) {
          if (p.id < entry.id && p.kind !== "system") {
            p.collapsed = true;
          }
        }
        break;
      }
      case "tool_started": {
        const tc = this.entries
          .flatMap((e) => (e.kind === "llm" ? e.toolCalls ?? [] : []))
          .find((t) => t.tool_call_id === ev.payload.tool_call_id);
        this.addEntry({
          kind: "tool",
          status: "running",
          stepId: ev.step_id,
          toolName: ev.payload.name,
          args: ev.payload.arguments,
          callIndex: ev.payload.call_index,
          argsJson: tc?.arguments,
        });
        break;
      }
      case "tool_substep": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        if (!entry.subSteps) entry.subSteps = [];
        entry.subSteps.push(ev.payload);
        this.scrollToEntry(entry.id);
        break;
      }
      case "rag_sources": {
        const entry = this.entryFor(ev.step_id);
        if (entry) {
          entry.sources = ev.payload.sources;
          entry.sourcesDuration = ev.payload.duration;
        }
        break;
      }
      case "tool_result": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        entry.resultText = ev.payload.text;
        entry.isError = ev.payload.is_error;
        this.scrollToEntry(entry.id);
        break;
      }
      case "tool_finished": {
        const entry = this.entryFor(ev.step_id);
        if (!entry) break;
        entry.status = "done";
        entry.duration = ev.payload.duration;
        entry.energy = ev.payload.energy;
        break;
      }
      case "limit_reached":
        if (ev.payload.limit === "max_turns") {
          this.maxTurnsReached = true;
        }
        break;
      case "error":
        this.errorMessage = ev.payload.recoverable
          ? this.errorRecoverable
          : this.errorGeneric;
        break;
      case "run_finished":
        for (const e of this.entries) {
          if (e.status === "running") {
            e.status = "done";
          }
        }
        this.totalEnergy = ev.payload.energy;
        this.running = false;
        this.done = true;
        break;
    }
  }

  private handleNestedEvent(ev: AgentEvent) {
    const parent = this.entryFor(ev.parent_step_id);
    if (!parent) return;
    switch (ev.type) {
      case "system_prompt": {
        const call = this.nestedCallFor(parent, ev.step_id!);
        call.systemText = ev.payload.text;
        break;
      }
      case "llm_request": {
        const call = this.nestedCallFor(parent, ev.step_id!);
        call.messages = ev.payload.messages;
        break;
      }
      case "reasoning_delta":
      case "content_delta": {
        const call = this.nestedCallFor(parent, ev.step_id!);
        if (ev.type === "reasoning_delta") {
          call.reasoning += ev.payload.text;
        } else {
          call.content += ev.payload.text;
        }
        break;
      }
      case "llm_finished": {
        const call = this.nestedCallFor(parent, ev.step_id!);
        call.status = "done";
        call.duration = ev.payload.duration;
        call.ttft = ev.payload.time_to_first_token;
        call.generationDuration = ev.payload.generation_duration;
        call.energy = ev.payload.energy;
        break;
      }
    }
  }

  private nestedCallFor(entry: Entry, stepId: string): NestedLlmCall {
    if (!entry.nestedCalls) entry.nestedCalls = [];
    let call = entry.nestedCalls.find((c) => c.stepId === stepId);
    if (!call) {
      call = { stepId, reasoning: "", content: "", status: "running" };
      entry.nestedCalls.push(call);
    }
    return call;
  }

  private entryFor(stepId: string | null): Entry | undefined {
    return this.entries.find((e) => e.stepId === stepId);
  }

  private addEntry(partial: Partial<Entry> & Pick<Entry, "kind" | "status">) {
    const entry: Entry = { id: this.nextId++, stepId: null, ...partial };
    this.entries.push(entry);
    this.scrollToEntry(entry.id);
  }
}
