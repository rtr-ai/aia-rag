import {
  AfterViewInit,
  Component,
  ElementRef,
  OnInit,
  ViewChild,
} from "@angular/core";
import { NgClass } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import {
  LLMMessageParams,
  PowerDataDisplayed,
  PowerUsageData,
  POWER_FIELDS,
  missingPower,
  requestPowerTotal,
  validPowerValue,
  Source,
  Step,
} from "./models";
import { environment } from "../../environments/environment";
import { NgZone } from "@angular/core";
import { WidgetInstance } from "friendly-challenge";
import { EnvService } from "../../services/env.service";
import { LOCALE_ID, Inject } from "@angular/core";
@Component({
    selector: "app-aiabot",
    imports: [NgClass, FormsModule],
    templateUrl: "./aiabot.component.html",
    styleUrl: "./aiabot.component.scss"
})
export class AiabotComponent implements OnInit, AfterViewInit {
  displayAnswer: string = "";
  step: Step = "initial";
  sources: Source[] = [];
  prompt: string = "";
  multiplier = 1;
  userPromptsPerLocale: Record<string, string[]> = {
    de: [
      "Ich möchte Lebensläufe von Bewerber:innen mit KI filtern. Ist das eine Hochrisiko-KI-Anwendung?",
      "Ich möchte E-Mails automatisch mit einem LLM beantworten. Muss ich das offenlegen?",
      "Muss ich bei KI-generierten Bildern kennzeichnen, dass diese mit KI generiert wurden?",
      "Wie kann ich KI-Kompetenz in meinem Unternehmen umsetzen?",
      "Ich entwickle KI-Systeme für Märkte außerhalb der EU. Gilt der AI Act für mich?",
    ],
    en: [
      "I want to filter applicant CVs with AI. Is this a high-risk AI application?",
      "I want to automatically answer emails with an LLM. Do I need to disclose this?",
      "Do I need to label AI-generated images as being created with AI?",
      "How can I implement AI competency in my company?",
      "I develop AI systems for markets outside the EU. Does the AI Act apply to me?",
    ],
  };
  userPrompts: string[] = [];
  userPrompt = "";
  useRerank = false;
  submittedUserPrompt = "";
  placeholderPrompt = "";
  maxLength: number = 500;
  mailtoLink: string = "mailto:ki@rtr.at?subject=Feedback%20RAG%20EU%20AI-Act";
  inputHeight: number = 70;
  tokensUsedFormatted: string = "";
  powerData: PowerDataDisplayed[] = [];
  totalProQuery: number | null = null;
  backendAvailable: boolean = true;
  firstTokenProgressPercent: number = 0;
  secondsToFirstToken = 40; //approx time until first token is expected
  avgSecondsPerRequest = 40;
  queueMessage: string = "";
  progressbarInterval: null | number = null;
  totalConsumption: PowerDataDisplayed = {
    name: "total",
    label: "Gesamter Energieverbrauch",
    cpu_kWh: 0,
    gpu_kWh: 0,
    ram_kWh: 0,
    total_kWh: 0,
    duration: 0,
  };
  sitekey: string;
  captchaSolution: string = "";
  @ViewChild("captchaContainer", { static: false })
  captchaContainer!: ElementRef;
  isCaptchaCompleted: boolean = false;

  constructor(
    private zone: NgZone,
    private envService: EnvService,
    @Inject(LOCALE_ID) public locale: string
  ) {
    this.sitekey = this.envService.friendlyCaptchaSitekey;
    this.initializeLocaleSpecificContent();
  }
  ngOnInit(): void {}
  ngAfterViewInit(): void {
    if (!this.sitekey || this.sitekey == "") {
      console.warn("Captcha is disabled because sitekey is not set or empty.");
      this.isCaptchaCompleted = true;
      return;
    }

    if (this.captchaContainer) {
      const widget = new WidgetInstance(this.captchaContainer.nativeElement, {
        sitekey: this.sitekey,
        language: "de",
        doneCallback: (solution) => {
          this.captchaSolution = solution;
          console.log("solution", solution);
          this.isCaptchaCompleted = true;
        },
        errorCallback: (error) => {
          console.warn("Captcha error", error);
          this.captchaSolution = "";
          this.isCaptchaCompleted = false;
        },
      });
    }
  }
  updateMailtoLink() {
    const recipient = "ki@rtr.at";
    const subject = $localize`:@@mailtoSubject:Feedback AI Act Chatbot`;
    const body = $localize`:@@mailtoBody:
    Mein Feedback betrifft folgende Anfrage:\n\n${this.userPrompt}\n\n
    Folgende Antwort hat die KI ausgegeben:\n\n${this.displayAnswer}\n\n
  `;

    const encodedSubject = encodeURIComponent(subject);
    const encodedBody = encodeURIComponent(body);

    this.mailtoLink = `mailto:${recipient}?subject=${encodedSubject}&body=${encodedBody}`;
  }
  private getQueueMessage(
    queuePosition: number,
    estimatedTime: number
  ): string {
    if (queuePosition === 1) {
      return "";
    } else if (queuePosition === 2) {
      return $localize`:@@queueMessageSingle:Es ist aktuell 1 Anfrage in der Warteschlange. Dauer bis zur Bearbeitung Ihrer Anfrage: ca. ${estimatedTime}:estimatedTime: Sekunden.`;
    } else {
      const queueCount = queuePosition - 1;
      return $localize`:@@queueMessageMultiple:Es sind aktuell ${queueCount}:queueCount: Anfragen in der Warteschlange. Dauer bis zur Bearbeitung Ihrer Anfrage: ca. ${estimatedTime}:estimatedTime: Sekunden.`;
    }
  }
  private initializePowerDataLabels() {
    return {
      index: $localize`:@@powerLabelIndex:Indexierung von relevanten Daten (einmalig pro Serverstart)`,
      prompt: $localize`:@@powerLabelRetrievalSeparate:Retrieve/Augment (ohne Re-Ranking)`,
      combined: $localize`:@@powerLabelRetrievalCombined:Altformat: Retrieve/Augment inklusive Re-Ranking (kombiniert)`,
      rerank: $localize`:@@powerLabelRerankSeparate:Re-Ranking`,
      response: $localize`:@@powerLabelResponse:Generierung der Antwort („Generate")`,
      total: $localize`:@@powerLabelRequestTotal:Anfrage gesamt (ohne Indexierung)`,
    };
  }

  promptLLM = async () => {
    const controller = new AbortController();
    const signal = controller.signal;
    const server = environment.LLM_ENDPOINT;
    if (server.length === 0) {
      throw Error("No LLM endpoint has been configured");
    }

    const datasetsPerLocale: Record<string, string> = {
      en: "ai_act_en",
      de: "ai_act_de",
      da: "data_act_de"
    };
    let defaultDataset = datasetsPerLocale["de"];
    if (this.locale && typeof datasetsPerLocale[this.locale] != "undefined") {
      defaultDataset = datasetsPerLocale[this.locale];
    }
    const params = {
      prompt: this.userPrompt,
      frc_captcha_solution: this.captchaSolution,
      dataset: defaultDataset,
      use_rerank: this.useRerank,
    };
    const updateSources = (sources: Source[]) => {
      this.sources = sources;
      calculateTotalTokens();
    };

    const updateSecondsToFirstToken = (queuePosition: number) => {
      const queueCount = Math.max(queuePosition, 1);
      const estimatedTime =
        Math.max(queueCount - 1, 1) * this.avgSecondsPerRequest;
      this.queueMessage = this.getQueueMessage(queuePosition, estimatedTime);
    };

    const updatePowerData = (input: PowerUsageData | string, eventType: string) => {
      const data: PowerUsageData = typeof input === "string" ? JSON.parse(input) : input;
      const labels = this.initializePowerDataLabels();
      const label = eventType === "power_index" ? labels.index
        : eventType === "power_prompt" ? (data.measurement_version === 2 ? labels.prompt : labels.combined)
        : eventType === "power_rerank" ? labels.rerank : labels.response;
      const normalized = { ...data };
      for (const field of POWER_FIELDS) normalized[field] = validPowerValue(data[field]);
      this.zone.run(() => {
        this.powerData = [
          ...this.powerData.filter(row => row.name !== eventType),
          { ...normalized, label, name: eventType },
        ];
      });
    };
    const calculateTotalPowerConsumption = () => {
      const labels = this.initializePowerDataLabels();
      const total = requestPowerTotal(this.powerData);
      this.zone.run(() => {
        for (const [name, label] of [["power_prompt", labels.prompt], ["power_rerank", labels.rerank], ["power_response", labels.response]]) {
          if (!this.powerData.some(row => row.name === name)) {
            this.powerData.push({ ...missingPower(), name, label, status: "unavailable" });
          }
        }
        const order = ["power_index", "power_prompt", "power_rerank", "power_response"];
        this.powerData.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
        this.totalConsumption = { ...total, name: "total", label: labels.total };
        this.totalProQuery = total.total_kWh;
      });
    };
    const calculateTotalTokens = () => {
      const tokensUsed = this.sources.reduce((total, source) => {
        const sourceTokens = !source.skip ? source.num_tokens : 0;

        const relevantTokens = source.relevantChunks.reduce(
          (chunkTotal, chunk) => {
            return chunkTotal + (!chunk.skip ? chunk.num_tokens : 0);
          },
          0
        );

        return total + sourceTokens + relevantTokens;
      }, 0);
      this.tokensUsedFormatted = formatWithSeperator(tokensUsed);
    };
    const formatWithSeperator = (value: number): string => {
      return Intl.NumberFormat("de-DE").format(value);
    };
    const updateStep = (step: Step) => {
      this.step = step;
      if (step === "done") {
        this.updateMailtoLink();
      }
    };
    const startCountdownToFirstToken = () => {
      const startOfInterval = new Date().getTime() / 1000;
      if (this.progressbarInterval !== null) {
        self.clearInterval(this.progressbarInterval);
      }
      const interval = self.setInterval(() => {
        if (this.displayAnswer.length > 0) {
          this.firstTokenProgressPercent = 100;
          self.clearInterval(interval);
          return;
        }
        const currentTime = new Date().getTime() / 1000;
        const elapsedTime = currentTime - startOfInterval;
        const progress =
          elapsedTime / Math.max(this.secondsToFirstToken, elapsedTime + 4);
        this.firstTokenProgressPercent = progress * 100;
      }, 500);
      this.progressbarInterval = interval;
    };
    const updatePrompt = (prompt: string) => {
      const lines = prompt.split("\n");
      const formattedLines = lines.map((line) => {
        if (line.trim().startsWith("Titel:")) {
          return "<hr>" + line;
        }
        return line;
      });
      this.prompt = formattedLines.join("\n");
    };
    const onErrorHappened = () => {
      this.backendAvailable = false;
      this.step = "initial";
      setTimeout(() => (this.backendAvailable = true), 12000);
    };
    let buffer = "";
    let updateTimeout: any = null;

    const appendAnswer = (answer: string) => {
      buffer += answer;
      if (!updateTimeout) {
        updateTimeout = setTimeout(() => {
          this.zone.run(() => {
            this.displayAnswer += buffer;
            buffer = "";
            updateTimeout = null;
          });
        }, 100);
      }
    };
    this.displayAnswer = "";
    this.powerData = [];
    this.totalProQuery = null;
    this.submittedUserPrompt = this.userPrompt;
    this.totalConsumption = {
      name: "total",
      label: "Gesamter Energieverbrauch",
      cpu_kWh: 0,
      gpu_kWh: 0,
      ram_kWh: 0,
      total_kWh: 0,
      duration: 0,
    };
    this.queueMessage = "";
    if (this.progressbarInterval !== null) {
      self.clearInterval(this.progressbarInterval);
    }

    try {
      await fetchEventSource(`${server}/chat`, {
        signal: signal,
        method: "POST",
        openWhenHidden: true,
        body: JSON.stringify(params),
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        onopen(response: Response): Promise<void> {
          if (response.ok && response.status === 200) {
            updateStep("research");
            return Promise.resolve();
          } else if (
            response.status >= 400 &&
            response.status < 500 &&
            response.status !== 429
          ) {
            console.error("Client-Side Errror  opening LLM Stream", response);
          }
          throw new Error("Error opening LLM Stream");
        },
        onerror() {
          onErrorHappened();
          throw new Error();
        },
        onmessage(event: { data: string }) {
          if (!event.data || event.data.length == 0) {
            return;
          }
          try {
            const data: LLMMessageParams = JSON.parse(event.data);
            switch (data.type) {
              case "sources":
                const sources: Source[] = JSON.parse(data.content);
                updateSources(sources);
                updateStep("prompt");
                setTimeout(() => {
                  document
                    .getElementById("modelContent")
                    ?.scrollIntoView({ behavior: "smooth" });
                }, 100);
                break;
              case "user":
                updatePrompt(data.content);
                updateStep("output");
                startCountdownToFirstToken();
                break;
              case "assistant":
                appendAnswer(data.content);
                break;
              case "power_index":
              case "power_prompt":
              case "power_rerank":
              case "power_response":
                updatePowerData(data.content as any, data.type);
                break;
              case "metadata":
                console.info("Chat model metadata:", data.content);
                break;
              case "queue_position":
                console.log(
                  `Current queue position: ${(data.content as any).position}`
                );
                updateSecondsToFirstToken((data.content as any).position || 0);
                break;
              default:
                console.log(
                  `Event of type <${data.type}> is not supported yet.`
                );
                break;
            }
          } catch (e: any) {
            console.error("Unable to parse JSON", e);
            console.log("Received data", event.data);
            onErrorHappened();
          }
        },
        onclose() {
          updateStep("done");
          calculateTotalPowerConsumption();
        },
      });
    } catch {
      onErrorHappened();
    }
  };
  private initializeLocaleSpecificContent(): void {
    this.userPrompts =
      this.userPromptsPerLocale[this.locale] || this.userPromptsPerLocale["de"];
    this.placeholderPrompt =
      this.userPrompts[Math.floor(Math.random() * this.userPrompts.length)];
  }
  onInput(event: Event): void {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 300)}px`;
  }
  answerQuery = async () => {
    await this.promptLLM();
  };
  formatPower(value: number | null, seconds = false): string {
    const number = validPowerValue(value);
    if (number === null) return $localize`:@@powerNotAvailable:Nicht verfuegbar`;
    return number.toFixed(seconds ? 2 : 6).replace(".", ",") + (seconds ? " Sek." : " kWh");
  }

  powerStatus(status?: string): string {
    if (status === "not_run" || status === "skipped") return $localize`:@@powerNotRun:Nicht ausgefuehrt`;
    if (status === "failed_fallback") return $localize`:@@powerFailedFallback:Fehlgeschlagen; normale Suche verwendet`;
    if (status === "unavailable") return $localize`:@@powerNotAvailable:Nicht verfuegbar`;
    if (status === "no_candidates") return $localize`:@@powerNoCandidates:Keine Kandidaten`;
    return "";
  }

  formatScore(score: number): string {
    return (score * 100).toFixed(1) + "%";
  }

  /**
   * Get number of sources, including relevant chunks
   * @param sources
   */
  getTotalSources(sources: Source[]) {
    let totalNumber = 0;
    for (let source of sources) {
      totalNumber++;
      if (source.relevantChunks) {
        totalNumber = totalNumber + source.relevantChunks.length;
      }
    }
    return totalNumber;
  }

  /**
   * Get number of sources with "skip" set to false
   * @param sources
   */
  getNotSkippedSources(sources: Source[]) {
    let totalNumber = 0;
    for (let source of sources) {
      if (!source.skip) {
        totalNumber++;
      }
      if (source.relevantChunks) {
        totalNumber =
          totalNumber + source.relevantChunks.filter((r) => !r.skip).length;
      }
    }
    return totalNumber;
  }
  toggleAccordion(index: number) {
    const element = document.getElementById(`content-${index}`);
    if (element) {
      element.classList.toggle("uk-hidden");
    }
  }
}
