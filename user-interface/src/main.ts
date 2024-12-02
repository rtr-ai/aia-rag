/// <reference types="@angular/localize" />

import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';
import 'uikit/dist/js/uikit.min.js';

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
