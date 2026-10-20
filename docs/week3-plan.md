# Week 3 Plan: October 15 to 21, 2026

Goal: local-only and cloud-only baselines with both workers, real cost numbers, and Weekly261021 (due Wednesday night). This plan is restructured around Prof. Cai's Oct 15 feedback: baselines are the graded deliverable, and every number I report carries its protocol, worker model, supervisor model, and grading method.

Weekend work is limited to the no-GPU verification tasks in Phase 1.

## Phase 1: Decide and verify (Sat Oct 17 to Sun Oct 18, no GPU needed)

1. Hand-check all 10 pilot answers against the gold labels in `benchmark_v5.json`. Build the 10-row table: patient, question, predicted, gold, match, notes. Pay special attention to patient_08 question 1 and record exactly why it is ambiguous.
2. Test whether `gemini-3.8-flash` 503s are gone and the free-tier quota reset. Record the result; this decides the supervisor freeze.
3. Freeze the worker: run `ollama list` on the laptop and confirm whether the real tag is `qwen3:4b` or `qwen3.5:4b`. Update the proposal text to match the code.
4. Freeze the supervisor: 3.8-flash if healthy (the proposal's choice; the cost model must be measured on it), else flash-lite, stated openly in the report.
5. Freeze the 20 warmup question IDs and the 200 eval question IDs. Record SHA-256 hashes of both lists. Verify zero overlap with the 10 pilot questions: (03,0), (07,0), (11,0), (15,0), (19,0), (04,1), (08,1), (12,1), (16,1), (20,1).
6. Decide padding N (distractor records per question) and write the one-line rationale.
7. Decide the 4B local-only design: frozen 2026-10-17, see `decision-4b-local-only-design.md`. In short: ~1,500-token document chunks (8 to 9 per record), one 4B call per chunk, majority vote over per-chunk letters, ties broken by earliest chunk, no supervisor. Do not run `pilot_4b.py` as written: it hands the whole 11,000 to 13,000 token record to a 2,048-token context, which is the same silent truncation found on Oct 13.
8. Answer the subscription question: Claude/ChatGPT yes or no, with one line of reasoning. (It does not include API credits; it is for coding, debugging, and reading papers.)

Done when: the 10-row hand-check table exists, both models are frozen in code/proposal/report, both question sets are hashed with the overlap check saved, padding N is frozen, and the subscription answer is written down.

## Phase 2: Build the harness (Mon Oct 19)

1. `data_loader.py`: reads LongHealth `benchmark_v5.json` and produces standard records (patient, question, documents, gold answer) with padding applied.
2. `grader.py`: rewritten fixed rule-based letter extractor (finds the stated choice at the end of the text, not the options list; regression-tested on the real pilot phrasings), plus a hand-check queue for unreadable answers. The hardened API grader stays as a cross-check only, so grading burns no supervisor budget.
3. Logging (`runlog.py`): per question, record wall time, worker tokens, supervisor tokens (thinking tokens separate), estimated API cost, and the full prompt and full answer (never truncated). Local GPU utilization logging is still pending.
4. `local_only_27b.py` (bosgame): 27B worker alone on the question, no supervisor. Flags: `--split`, `--think on/off`, `--limit`.
5. `local_only_4b_chunked.py` (laptop): 4B worker alone, ~1,500-token document chunks, majority vote over per-chunk letters, ties to the earliest chunk, no supervisor.
6. `cloud_only.py` (bosgame): gemini-3.8-flash alone on the full document.

Done when: all six run end to end on one warmup question without errors, and the log row contains every field above.

## Phase 3: Run the baselines (Tue Oct 20)

Run order on the 20 warmup questions: 27B local-only, 4B local-only (chunked per the Phase 1 decision), cloud-only. That is 3 conditions x 2 workers x 20 questions = 120 runs to validate the pipeline before scaling to the 200-question eval set in Week 4.

Also run the thinking-off check: 20 warmup questions through the 27B worker with `think=False` at `num_ctx=16384`, graded against gold. Pass bar: accuracy within 10 percent of the think-on reference. Record accuracy, worker tokens, and wall time per question.

Record real supervisor tokens per call on the frozen supervisor model, with thinking tokens listed separately. These numbers replace the cost model's assumptions.

Done when: baseline accuracy, cost, and latency exist for both workers on all 20 warmup questions, all labeled with protocol/worker/supervisor/grading, and the thinking-off delta is stated.

## Phase 4: Cost estimate and report (Wed Oct 21)

1. Fill the cost spreadsheet with the measured Week 3 token numbers.
2. Write the cost estimate with Cai's breakdown: supervisor calls by component (plans, step reviews, aggregation per protocol, cloud-only baseline, MinionS baseline, oracle search, both workers); real tokens per call with thinking as output; debugging allowance; days the full study takes on free tier only; what gets dropped if cost or quota breaks.
3. Make the billing decision with the corrected math: (a) flash-lite free, ~24 days at $0; (b) paid 3.8-flash, ~$91 + debugging; (c) the Google AI Pro student offer quota check first. Free-on-3.8-flash is off the table (~20 req/day = ~585 days).
4. Write Weekly261021. Required contents: the run table (every number labeled), the cost estimate, the proposal timeline walked item by item (done/not done), the billing decision, and the subscription answer. First person, no em/en dashes, AI usage stated as debugging help only.
5. Commit and push the week's code and docs to GitHub from the laptop.

## Standing rules (from the Oct 15 feedback)

- Every number states protocol, worker model, supervisor model, and grading method. No exceptions.
- Report and code are cross-checked both ways before every commit and every report: every protocol, model name, and number named in a doc must exist in the code with the same name and value, and everything the code does that a reader would care about must appear in the report. When they drift, fix the doc or the code the same day, never later.
- Never label a Minion run as MinionS.
- Never run the 4B on a full record without chunking.
- Cache every supervisor response to disk keyed by (model, prompt hash). Never pay twice for the same call.
- Commit and push from the laptop at the end of each session. Never commit API keys or secrets.
