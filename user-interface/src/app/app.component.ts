import { Component, Inject, LOCALE_ID } from '@angular/core';
import { Title } from '@angular/platform-browser';
import { RouterOutlet } from '@angular/router';
import {AiabotComponent} from "./aiabot/aiabot.component";

@Component({
    selector: 'app-root',
    imports: [RouterOutlet, AiabotComponent],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'ki-servicestelle-aiabot';

  constructor(@Inject(LOCALE_ID) public locale: string, titleService: Title) {
    titleService.setTitle($localize`:@@pageTitle:AI Act Chatbot | KI-Servicestelle | RTR`);
  }
}
