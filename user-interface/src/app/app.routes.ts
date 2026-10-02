import { Routes } from '@angular/router';
import { AiabotComponent } from './aiabot/aiabot.component';
import { AgentComponent } from './agent/agent.component';

export const routes: Routes = [
  { path: '', component: AiabotComponent },
  { path: 'agent', component: AgentComponent },
];
