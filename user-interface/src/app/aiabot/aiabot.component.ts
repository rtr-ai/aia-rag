import { Component, OnInit } from "@angular/core";
import { CommonModule, NgClass, NgForOf, NgIf } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { answer1, answer2 } from "./knowledge";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { LLMMessageParams, Source, Step } from "./models";
import { environment } from "../. ./environments/environment";
import { ChangeDetectorRef } from "@angular/core";
@Component({
  selector: "app-aiabot",
  standalone: true,
  imports: [NgIf, NgForOf, NgClass, FormsModule],
  templateUrl: "./aiabot.component.html",
  styleUrl: "./aiabot.component.scss",
})
export class AiabotComponent implements OnInit {
  displayAnswer: string = "";
  step: Step = "initial";
  sources: Source[] = [];
  prompt: string = "";
  multiplier = 1;
  userPrompt: string =
    "Ich möchte LLama-3 für meine Firma fine-tunen. Gilt der AIA für mich?";
  constructor(private cdr: ChangeDetectorRef) {}
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
    };
    const updateStep = (step: Step) => {
      this.step = step;
    };
    const updatePrompt = (prompt: string) => {
      this.prompt = prompt;
    };
    let buffer = "";
    let updateTimeout: any = null;

    const appendAnswer = (answer: string) => {
      buffer += answer;
      if (!updateTimeout) {
        updateTimeout = setTimeout(() => {
          this.displayAnswer += buffer;
          buffer = "";
          updateTimeout = null;
          this.cdr.detectChanges();
        }, 100);
      }
    };
    this.displayAnswer = "";
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
        const data: LLMMessageParams = JSON.parse(event.data);
        switch (data.type) {
          case "sources":
            const sources: Source[] = JSON.parse(data.content);
            updateSources(sources);
            updateStep("prompt");
            break;
          case "user":
            updatePrompt(data.content);
            updateStep("output");
            break;
          case "assistant":
            appendAnswer(data.content);
            break;
          default:
            console.log(`Event of type <${data.type}> is not supported yet.`);
            break;
        }
      },
      onclose() {
        updateStep("done");
      },
    });
  };

  playDemo = async () => {
    await this.promptLLM();
  };
  formatScore(score: number): string {
    return (score * 100).toFixed(1) + "%";
  }
  toggleAccordion(index: number) {
    const element = document.getElementById(`content-${index}`);
    if (element) {
      const isExpanded = element.style.display === "block";
      element.style.display = isExpanded ? "none" : "block";
    }
  }

  copyStringWordByWord = async (sourceStr: string, delay = 50) => {
    let sourceWords = sourceStr.split(/\s|\n/); // Split by space or newline
    let destStr = "";

    for (let i = 0; i < sourceWords.length; i++) {
      let word = sourceWords[i];
      // await only if word is not empty (it can be empty because of \n)
      if (word) {
        await new Promise((r) => setTimeout(r, delay));
        destStr += word + " ";
      } else {
        // if word is empty append newline
        destStr += "<br /><br/>";
      }

      this.displayAnswer = destStr.trim();
    }
  };
}
