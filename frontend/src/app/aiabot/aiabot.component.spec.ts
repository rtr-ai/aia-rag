import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AiabotComponent } from './aiabot.component';
import { CommonModule } from '@angular/common';

describe('AiabotComponent', () => {
  let component: AiabotComponent;
  let fixture: ComponentFixture<AiabotComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AiabotComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AiabotComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
