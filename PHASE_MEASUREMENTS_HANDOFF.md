# Separate Phase Measurements: Handoff

## What changed

Each new request reports three separate, non-overlapping stages:

- Retrieve/Augment: retrieval and prompt assembly, excluding reranking.
- Re-Ranking: candidate preparation, instruction/model preparation, ranking and result handling. Lazy model loading is included when it occurs.
- Generate: answer generation.

Request total = Retrieve/Augment + Re-Ranking + Generate. Startup indexing is displayed separately and is never added to each request total.

Durations use elapsed wall-clock time. CPU and GPU energy use hardware-counter differences where available. RAM energy remains an estimate from RAM usage. Missing counters produce null / Not available, not zero. Zero means the stage was explicitly not run or skipped.

Counters cover the host CPU, GPU 0 and RAM, not isolated per-request hardware consumption. Concurrent workloads can affect energy readings. No energy is proportionally allocated by duration.

## Files for the backend and chatbot GitHub project

Root: `C:/Users/bma/aia-rag-bojan-test`

Deploy these application files together:

1. `llm-service/src/services/power_meter_service.py`
2. `llm-service/src/services/index_service.py`
3. `llm-service/src/services/chat_service.py`
4. `user-interface/src/app/aiabot/models.ts`
5. `user-interface/src/app/aiabot/aiabot.component.ts`
6. `user-interface/src/app/aiabot/aiabot.component.html`
7. `user-interface/src/app/aiabot/aiabot.component.scss`
8. `user-interface/src/locale/messages.en.xlf`
9. `user-interface/src/locale/messages.da.xlf`

No Docker Compose or model configuration change is required for this feature. The existing unrelated `.gitignore` edit was left untouched.

## Files for the local testbench

The running comparison tools are in a DIFFERENT checkout.

Root: `C:/Users/bma/aia-rag`

1. `auto_test/sse_utils.py`
2. `auto_test/phase_metrics.py` (new shared calculations)
3. `auto_test/chunk_power_summary.py`
4. `auto_test/templates/index.html`
5. `answers_auto_test/power_summary.py`
6. `answers_auto_test/app.py`
7. `answers_auto_test/templates/index.html`

These are testbench application files, not files required by the deployed chatbot. Existing unrelated changes in that checkout were preserved.

## Verification-only files

Backend/frontend root:

- `llm-service/tests/test_phase_measurements.py`
- `llm-service/tests/phase_preview_server.py` (loopback-only synthetic SSE preview)
- `user-interface/src/app/aiabot/models.spec.ts`
- `llm-service/tests/separate-phase-preview.jpg` (synthetic UI screenshot)

Testbench root:

- `auto_test/test_chunk_measurements.py`
- `answers_auto_test/test_power_summary.py`
- `answers_auto_test/test_report_metrics.py`

The backend test directory is currently ignored by Git. It does not need to be deployed for the application to work. The preview server and screenshot are verification aids, not production services.

## Who does what next

1. You transfer the nine backend/frontend application files to the corresponding paths in your GitHub project, using your usual workflow.
2. Your colleague deploys/rebuilds the backend AND chatbot frontend together.
3. You restart the local testbench so it loads its updated Python helpers and templates.
4. You start a NEW Import Requests run for chunk comparisons, and a fresh answer-comparison run. Do not reuse a cached old answer/retrieval run to verify the new measurements.
5. Check that the stream includes `power_prompt`, `power_rerank` and `power_response`, with `measurement_version: 2`.
6. Check the new UI/Excel summary: three separate request stages, request total, separate startup indexing, and available-question counts for every averaged metric.

Retrieval-only requests emit all three events; Generate is explicitly not run and has zero cost. They do not call the answer-generating model. Reranking off has status `not_run` and zero cost. A failed rerank has status `failed_fallback`; its attempted duration/energy are retained while normal retrieval is used.

Measurement collection does not require a metadata event. When metadata exists, its reranking duration matches `power_rerank.duration`.

## Older responses and averages

Old unversioned `power_prompt` is labelled combined Retrieve/Augment + Re-Ranking. Its separate reranking breakdown is Not available. It is not added twice.

Each average uses only questions with that measurement available and shows its count. Request totals are calculated per question before averaging; a total is unavailable when a required component is missing. This avoids inventing totals from partial averages.

No previously generated Excel, HTML or PDF report was changed. Fresh imports/exports use the new calculations.

## Local verification

- Backend controlled-clock/counter tests: 6 passed.
- Chunk-comparison suite: 61 passed.
- Answer-comparison suite: 69 run, 66 passed, 3 skipped.
- Frontend totals tests: 5 passed in ChromeHeadless.
- Localized production Angular build: passed. Existing Sass, bundle-budget, CommonJS and unrelated locale warnings remain.
- UI checked with a local synthetic stream at desktop and mobile widths. Synthetic durations 2 + 3 + 5 produce a 10-second request total; the separate 100-second indexing value is excluded.
- Missing-energy and failed-fallback UI behavior checked. Existing source/answer fixtures remain unchanged.
- Gortex change detection and contract checks completed; no configured guards. Its graph does not include the ignored backend tests or the separate testbench checkout, so those were verified directly with their test suites.

Actual deployed SSE output and hardware-counter availability still require the fresh server import described above. No Git commit, push or deployment was performed.
