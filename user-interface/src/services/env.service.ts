import { Injectable } from "@angular/core";

@Injectable({
  providedIn: "root",
})
export class EnvService {
  private _friendlyCaptchaSitekey = "";

  constructor() {
    // Read values from window.env
    const w = window as any;
    if (w.env) {
      this._friendlyCaptchaSitekey = w.env.FRIENDLY_CAPTCHA_SITEKEY || "";
    }
  }

  get friendlyCaptchaSitekey(): string {
    return this._friendlyCaptchaSitekey;
  }
}
