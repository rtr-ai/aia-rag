import { Component, OnInit } from "@angular/core";
import { NgClass, NgForOf, NgIf } from "@angular/common";
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

@Component({
  selector: "app-aiabot",
  standalone: true,
  imports: [NgIf, NgForOf, NgClass, FormsModule],
  templateUrl: "./aiabot.component.html",
  styleUrl: "./aiabot.component.scss",
})
export class AiabotComponent implements OnInit {
  displayAnswer: string = "";
  step: Step = "done";
  sources: Source[] = [];
  prompt: string = "";
  multiplier = 1;
  userPrompts = [
    "Ich möchte Lebensläufe von Bewerber:innen mit KI filtern. Ist das eine Hochrisiko-KI-Anwendung?",
    "Ich möchte E-Mails automatisch mit einem LLM beantworten. Muss ich das offenlegen?",
    "Muss ich bei KI-generierten Bildern kennzeichnen, dass diese mit KI generiert wurden?",
    "Wie kann ich KI-Kompetenz in meinem Unternehmen umsetzen?",
    "Ich entwickle KI-Systeme für Märkte außerhalb der EU. Gilt der AI Act für mich?",
  ];
  userPrompt =
    this.userPrompts[Math.floor(Math.random() * this.userPrompts.length)];
  maxLength: number = 500;
  inputHeight: number = 70;
  tokensUsedFormatted: string = "";
  powerData: PowerDataDisplayed[] = [];
  totalProQuery: number = 0;
  totalConsumption: PowerDataDisplayed = {
    name: "total",
    label: "Gesamter Energieverbrauch",
    cpu_kWh: 0,
    gpu_kWh: 0,
    ram_kWh: 0,
    total_kWh: 0,
    duration: 0,
  };

  constructor(private zone: NgZone) {}
  ngOnInit(): void {}

  promptLLM = async () => {
    const controller = new AbortController();
    const signal = controller.signal;
    const server = environment.LLM_ENDPOINT;
    if (server.length === 0) {
      throw Error("No LLM endpoint has been configured");
    }
    const params = {
      prompt: this.userPrompt,
    };
    const updateSources = (sources: Source[]) => {
      this.sources = sources;
      calculateTotalTokens();
    };
    const updatePowerData = (data: PowerUsageData, eventType: string) => {
      let name = "";
      switch (eventType) {
        case "power_index":
          name = "Indexierung von relevanten Daten (einmalig pro Serverstart)";
          break;
        case "power_prompt":
          name = "Erstellung des Prompts („Retrieve” und „Augment”)";
          break;
        case "power_response":
          name = "Generierung der Antwort („Generate”)";
          break;
      }
      this.zone.run(() => {
        this.powerData = [
          ...this.powerData,
          { label: name, name: eventType, ...data },
        ];
      });
    };
    const calculateTotalPowerConsumption = () => {
      this.totalProQuery = 0;
      this.totalConsumption = {
        name: "total",
        label: "Gesamter Energieverbrauch",
        cpu_kWh: 0,
        gpu_kWh: 0,
        ram_kWh: 0,
        total_kWh: 0,
        duration: 0,
      };
      this.powerData.forEach((item) => {
        if (item.name !== "power_index") {
          this.totalProQuery += item.total_kWh;
        }
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
    this.totalConsumption = {
      name: "total",
      label: "Gesamter Energieverbrauch",
      cpu_kWh: 0,
      gpu_kWh: 0,
      ram_kWh: 0,
      total_kWh: 0,
      duration: 0,
    };

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
              break;
            case "assistant":
              appendAnswer(data.content);
              break;
            case "power_index":
            case "power_prompt":
            case "power_response":
              updatePowerData(data.content as any, data.type);
              break;
            default:
              console.log(`Event of type <${data.type}> is not supported yet.`);
              break;
          }
        } catch (e: any) {
          console.error("Unable to parse JSON", e);
          console.log("Received data", event.data);
        }
      },
      onclose() {
        updateStep("done");
        calculateTotalPowerConsumption();
      },
    });
  };

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
