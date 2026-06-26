import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {AiabotComponent} from "./aiabot/aiabot.component";
import {AgentDemoComponent} from "./agent-demo/agent-demo.component";

@Component({
    selector: 'app-root',
    imports: [RouterOutlet, AiabotComponent, AgentDemoComponent],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'ki-servicestelle-aiabot';
}
