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
  submittedUserPrompt = "";
  placeholderPrompt = "";
  maxLength: number = 500;
  mailtoLink: string = "mailto:ki@rtr.at?subject=Feedback%20RAG%20EU%20AI-Act";
  inputHeight: number = 70;
  tokensUsedFormatted: string = "";
  powerData: PowerDataDisplayed[] = [];
  totalProQuery: number = 0;
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
      prompt: $localize`:@@powerLabelPrompt:Erstellung des Prompts („Retrieve" und „Augment")`,
      response: $localize`:@@powerLabelResponse:Generierung der Antwort („Generate")`,
      total: $localize`:@@powerLabelTotal:Gesamter Energieverbrauch`,
    };
  }

  // ---------------------------------------------------------------------------
  // PROTOTYPE MODE – always replays a fixed fake response
  // ---------------------------------------------------------------------------
  private readonly PROTOTYPE_MODE = true;

  private readonly FAKE_SOURCES: Source[] = [
    {
      score: 0.923,
      title: "Art. 6 KI-VO – Klassifizierungsregeln für Hochrisiko-KI-Systeme",
      content:
        "Ein KI-System, das in Anhang III aufgeführte Bereiche betrifft und Entscheidungen trifft, die natürliche Personen erheblich beeinflussen (z. B. Beschäftigung, Bildung, Kreditvergabe), gilt als Hochrisiko-KI-System und unterliegt den Anforderungen des Titels III Kapitel 2.",
      num_tokens: 312,
      skip: false,
      skip_reason: "context_window",
      relevantChunks: [
        {
          id: "art6-abs1",
          title: "Art. 6 Abs. 1 KI-VO",
          content:
            "Ungeachtet des Absatzes 2 gilt ein KI-System, das in Anhang II aufgeführten Produkten als Sicherheitsbauteil dient oder selbst ein solches Produkt darstellt, als Hochrisiko-KI-System.",
          num_tokens: 98,
          skip: false,
          skip_reason: "context_window",
        },
        {
          id: "art6-abs2",
          title: "Art. 6 Abs. 2 KI-VO",
          content:
            "KI-Systeme gemäß Anhang III gelten als Hochrisiko-KI-Systeme, es sei denn, sie stellen kein erhebliches Risiko für die Gesundheit, Sicherheit oder Grundrechte natürlicher Personen dar.",
          num_tokens: 104,
          skip: false,
          skip_reason: "context_window",
        },
      ],
    },
    {
      score: 0.871,
      title: "Anhang III KI-VO – Liste der Hochrisiko-KI-Systeme",
      content:
        "Anhang III enthält die Liste der KI-Systeme, die gemäß Art. 6 Abs. 2 als Hochrisiko eingestuft werden, darunter: Biometrie, kritische Infrastruktur, Bildung, Beschäftigung, wesentliche Dienstleistungen, Strafverfolgung, Migration sowie Rechtspflege.",
      num_tokens: 287,
      skip: false,
      skip_reason: "context_window",
      relevantChunks: [
        {
          id: "annex3-sec4",
          title: "Anhang III Nr. 4 – Beschäftigung und Personalmanagement",
          content:
            "KI-Systeme, die für die Einstellung oder Auswahl natürlicher Personen genutzt werden, insbesondere für die Auswahl oder das Screening von Bewerbungen, gehören zu den Hochrisiko-KI-Systemen.",
          num_tokens: 115,
          skip: false,
          skip_reason: "context_window",
        },
      ],
    },
    {
      score: 0.754,
      title: "ErwG 48 KI-VO – Hochrisiko-Begründung im Bereich Beschäftigung",
      content:
        "KI-Systeme im Bereich Beschäftigung können erhebliche Auswirkungen auf künftige Berufsaussichten und Lebensläufe der betroffenen Personen haben und sind daher als Hochrisiko einzustufen.",
      num_tokens: 143,
      skip: true,
      skip_reason: "context_window",
      relevantChunks: [],
    },
  ];

  private readonly FAKE_PROMPT =
    "Systemanweisung: Sie sind ein hilfreicher KI-Assistent der KI-Servicestelle RTR, der Fragen zum EU AI Act beantwortet.\n\nTitel: Art. 6 KI-VO – Klassifizierungsregeln für Hochrisiko-KI-Systeme\nEin KI-System, das in Anhang III aufgeführte Bereiche betrifft ...\n\nTitel: Anhang III Nr. 4 – Beschäftigung und Personalmanagement\nKI-Systeme, die für die Einstellung oder Auswahl natürlicher Personen genutzt werden ...\n\nFrage des Nutzers: Ich möchte Lebensläufe von Bewerber:innen mit KI filtern. Ist das eine Hochrisiko-KI-Anwendung?";

  private readonly FAKE_ANSWER =
    "Ja, das Filtern von Lebensläufen mit einem KI-System gilt nach dem EU AI Act sehr wahrscheinlich als <strong>Hochrisiko-KI-Anwendung</strong>.\n\n" +
    "<strong>Warum?</strong>\n\n" +
    "Gemäß <strong>Art. 6 Abs. 2 i. V. m. Anhang III Nr. 4</strong> der KI-Verordnung (KI-VO) zählen KI-Systeme, die im Bereich <em>Beschäftigung und Personalmanagement</em> eingesetzt werden – insbesondere zur Einstellung, Auswahl oder zum Screening von Bewerber:innen – zu den Hochrisiko-KI-Systemen. Das automatisierte Filtern von Bewerbungsunterlagen fällt klar in diese Kategorie.\n\n" +
    "<strong>Was bedeutet das für Sie?</strong>\n\n" +
    "Als Anbieter oder Betreiber eines solchen Systems müssen Sie unter anderem folgende Anforderungen erfüllen:\n" +
    "<ul><li><strong>Risikomanagementsystem</strong> (Art. 9 KI-VO): kontinuierliche Identifikation und Minimierung von Risiken.</li>" +
    "<li><strong>Daten-Governance</strong> (Art. 10 KI-VO): Sicherstellung hochwertiger, repräsentativer Trainingsdaten.</li>" +
    "<li><strong>Transparenz & Dokumentation</strong> (Art. 11–13 KI-VO): technische Dokumentation und Gebrauchsanweisung.</li>" +
    "<li><strong>Menschliche Aufsicht</strong> (Art. 14 KI-VO): sicherstellen, dass eine menschliche Kontrollinstanz die KI-Entscheidungen überprüfen kann.</li>" +
    "<li><strong>Registrierung</strong> (Art. 49 KI-VO): Eintrag in die EU-Datenbank für Hochrisiko-KI-Systeme.</li></ul>\n\n" +
    "<strong>Ausnahme möglich?</strong>\n\n" +
    "Seit der finalen Fassung der KI-VO kann unter bestimmten Voraussetzungen eine Ausnahme von der Hochrisiko-Einstufung gelten (Art. 6 Abs. 3 KI-VO), wenn das System kein erhebliches Risiko für Gesundheit, Sicherheit oder Grundrechte darstellt. Dies muss jedoch <em>dokumentiert und begründet</em> werden.\n\n" +
    "Für eine rechtssichere Einschätzung Ihres konkreten Anwendungsfalls empfehlen wir, das (menschliche) Team der <strong>KI-Servicestelle</strong> zu kontaktieren.";

  promptLLM = async () => {
    const delay = (ms: number) =>
      new Promise<void>((resolve) => setTimeout(resolve, ms));

    const updateSources = (sources: Source[]) => {
      this.sources = sources;
      calculateTotalTokens();
    };
    const updatePowerData = (data: PowerUsageData, eventType: string) => {
      const labels = this.initializePowerDataLabels();
      let name = "";
      switch (eventType) {
        case "power_index": name = labels.index; break;
        case "power_prompt": name = labels.prompt; break;
        case "power_response": name = labels.response; break;
      }
      this.zone.run(() => {
        this.powerData = [...this.powerData, { label: name, name: eventType, ...data }];
      });
    };
    const calculateTotalPowerConsumption = () => {
      this.totalProQuery = 0;
      this.totalConsumption = { name: "total", label: "Gesamter Energieverbrauch", cpu_kWh: 0, gpu_kWh: 0, ram_kWh: 0, total_kWh: 0, duration: 0 };
      this.powerData.forEach((item) => {
        if (item.name !== "power_index") this.totalProQuery += item.total_kWh;
        this.totalConsumption.cpu_kWh += item.cpu_kWh;
        this.totalConsumption.gpu_kWh += item.gpu_kWh;
        this.totalConsumption.ram_kWh += item.ram_kWh;
        this.totalConsumption.total_kWh += item.total_kWh;
        this.totalConsumption.duration += item.duration;
      });
    };
    const calculateTotalTokens = () => {
      const tokensUsed = this.sources.reduce((total, source) => {
        const sourceTokens = !source.skip ? source.num_tokens : 0;
        const relevantTokens = source.relevantChunks.reduce(
          (chunkTotal, chunk) => chunkTotal + (!chunk.skip ? chunk.num_tokens : 0), 0
        );
        return total + sourceTokens + relevantTokens;
      }, 0);
      this.tokensUsedFormatted = Intl.NumberFormat("de-DE").format(tokensUsed);
    };
    const updateStep = (step: Step) => {
      this.step = step;
      if (step === "done") this.updateMailtoLink();
    };
    const startCountdownToFirstToken = () => {
      const startOfInterval = new Date().getTime() / 1000;
      if (this.progressbarInterval !== null) self.clearInterval(this.progressbarInterval);
      const interval = self.setInterval(() => {
        if (this.displayAnswer.length > 0) {
          this.firstTokenProgressPercent = 100;
          self.clearInterval(interval);
          return;
        }
        const elapsedTime = new Date().getTime() / 1000 - startOfInterval;
        this.firstTokenProgressPercent =
          (elapsedTime / Math.max(this.secondsToFirstToken, elapsedTime + 4)) * 100;
      }, 500);
      this.progressbarInterval = interval;
    };
    const updatePrompt = (prompt: string) => {
      const lines = prompt.split("\n");
      this.prompt = lines
        .map((line) => (line.trim().startsWith("Titel:") ? "<hr>" + line : line))
        .join("\n");
    };

    // Reset state
    this.displayAnswer = "";
    this.powerData = [];
    this.submittedUserPrompt = this.userPrompt;
    this.totalConsumption = { name: "total", label: "Gesamter Energieverbrauch", cpu_kWh: 0, gpu_kWh: 0, ram_kWh: 0, total_kWh: 0, duration: 0 };
    this.queueMessage = "";
    if (this.progressbarInterval !== null) self.clearInterval(this.progressbarInterval);

    if (this.PROTOTYPE_MODE) {
      // --- Fake SSE replay ---
      updateStep("research");
      await delay(1200);

      // sources event
      this.zone.run(() => {
        updateSources(this.FAKE_SOURCES);
        updateStep("prompt");
        setTimeout(() => document.getElementById("modelContent")?.scrollIntoView({ behavior: "smooth" }), 100);
      });
      await delay(900);

      // power_index event
      updatePowerData({ cpu_kWh: 0.000012, gpu_kWh: 0.000000, ram_kWh: 0.000003, total_kWh: 0.000015, duration: 0.43 }, "power_index");
      await delay(400);

      // power_prompt event
      updatePowerData({ cpu_kWh: 0.000031, gpu_kWh: 0.000000, ram_kWh: 0.000008, total_kWh: 0.000039, duration: 1.12 }, "power_prompt");
      await delay(200);

      // user event
      this.zone.run(() => {
        updatePrompt(this.FAKE_PROMPT);
        updateStep("output");
        startCountdownToFirstToken();
      });
      await delay(800);

      // assistant events – stream word by word
      const words = this.FAKE_ANSWER.split(" ");
      for (let i = 0; i < words.length; i++) {
        const chunk = (i === 0 ? "" : " ") + words[i];
        this.zone.run(() => { this.displayAnswer += chunk; });
        await delay(28 + Math.random() * 22);
      }

      // power_response event
      updatePowerData({ cpu_kWh: 0.000187, gpu_kWh: 0.002341, ram_kWh: 0.000054, total_kWh: 0.002582, duration: 18.74 }, "power_response");
      await delay(200);

      // close
      this.zone.run(() => {
        updateStep("done");
        calculateTotalPowerConsumption();
      });
      return;
    }

    // ---------------------------------------------------------------------------
    // REAL implementation (only runs when PROTOTYPE_MODE = false)
    // ---------------------------------------------------------------------------
    const controller = new AbortController();
    const signal = controller.signal;
    const server = environment.LLM_ENDPOINT;
    if (server.length === 0) {
      throw Error("No LLM endpoint has been configured");
    }
    const datasetsPerLocale: Record<string, string> = { en: "ai_act_en", de: "ai_act_de" };
    let defaultDataset = datasetsPerLocale["de"];
    if (this.locale && typeof datasetsPerLocale[this.locale] != "undefined") {
      defaultDataset = datasetsPerLocale[this.locale];
    }
    const params = { prompt: this.userPrompt, frc_captcha_solution: this.captchaSolution, dataset: defaultDataset };

    let buffer = "";
    let updateTimeout: any = null;
    const appendAnswer = (answer: string) => {
      buffer += answer;
      if (!updateTimeout) {
        updateTimeout = setTimeout(() => {
          this.zone.run(() => { this.displayAnswer += buffer; buffer = ""; updateTimeout = null; });
        }, 100);
      }
    };
    const onErrorHappened = () => {
      this.backendAvailable = false;
      this.step = "initial";
      setTimeout(() => (this.backendAvailable = true), 12000);
    };
    const updateSecondsToFirstToken = (queuePosition: number) => {
      const queueCount = Math.max(queuePosition, 1);
      const estimatedTime = Math.max(queueCount - 1, 1) * this.avgSecondsPerRequest;
      this.queueMessage = this.getQueueMessage(queuePosition, estimatedTime);
    };

    try {
      await fetchEventSource(`${server}/chat`, {
        signal: signal,
        method: "POST",
        openWhenHidden: true,
        body: JSON.stringify(params),
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        onopen(response: Response): Promise<void> {
          if (response.ok && response.status === 200) { updateStep("research"); return Promise.resolve(); }
          else if (response.status >= 400 && response.status < 500 && response.status !== 429) {
            console.error("Client-Side Errror  opening LLM Stream", response);
          }
          throw new Error("Error opening LLM Stream");
        },
        onerror() { onErrorHappened(); throw new Error(); },
        onmessage(event: { data: string }) {
          if (!event.data || event.data.length == 0) return;
          try {
            const data: LLMMessageParams = JSON.parse(event.data);
            switch (data.type) {
              case "sources":
                const sources: Source[] = JSON.parse(data.content);
                updateSources(sources);
                updateStep("prompt");
                setTimeout(() => document.getElementById("modelContent")?.scrollIntoView({ behavior: "smooth" }), 100);
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
              case "power_response":
                updatePowerData(data.content as any, data.type);
                break;
              case "queue_position":
                updateSecondsToFirstToken((data.content as any).position || 0);
                break;
              default:
                console.log(`Event of type <${data.type}> is not supported yet.`);
                break;
            }
          } catch (e: any) {
            console.error("Unable to parse JSON", e);
            onErrorHappened();
          }
        },
        onclose() { updateStep("done"); calculateTotalPowerConsumption(); },
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
