import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {AiabotComponent} from "./aiabot/aiabot.component";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, AiabotComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'ki-servicestelle-aiabot';
}
