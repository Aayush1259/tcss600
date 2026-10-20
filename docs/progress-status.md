# Progress Status: October 20, 2026

Independent study, TCSS 600, Prof. Wei Cai. Week 3 of 10 (Oct 15 to 21). Weekly261021 report due Wednesday, Oct 21, night.

## What is done so far

**Weeks 1 and 2 (Sep 30 to Oct 14):**
- Laptop track: Ollama running, `qwen3:4b` worker, Minions code installed in WSL2. The 4B JSON failure turned out to be a harness bug, not a model weakness: adding `structured_output_schema=JobOutput` (native Ollama JSON mode) fixed the full flow, which then returned the correct 13.2 g/dL.
- Lab track (bosgame): SSH over Tailscale, user-local Ollama 0.40.3, `qwen3.8:27b` pulled, smoke test passed. (Correction to Report 1: bosgame has 120 GB unified memory.)
- Found and fixed a silent context-truncation bug: the worker window `num_ctx=8192` was smaller than the ~12,766-token LongHealth document, so the server truncated the prompt to 4,098 tokens and the worker answered from a document it never fully saw. `num_ctx=16384` fixed it.
- Ran a 10-question 27B pilot. Corrected record (see below): the pilot ran the **Minion** protocol with a **gemini-3.5-flash-lite** supervisor.
- Weekly Report 1 (Weekly261014) finalized. GitHub repo `Aayush1259/tcss600` is public.
- Read Fang et al. (arXiv:2509.24050): same escalation question as RQ1, solved with RL post-training. Their finding that prompt surface features fail at routing supports the logprobs-based signal premise of RQ1.

**Oct 15 to 17:**
- Received Prof. Cai's feedback on Weekly Report 1. Full response in `docs/feedback-oct15-response.md`.
- Hand-verified all 10 pilot answers against the gold labels: **8 of 10 confirmed, 1 ambiguous** (patient_08 q1: the CT report lists both candidate findings, so the question itself is ambiguous; useful for RQ1), **1 inconclusive** (patient_04 q1: the pilot saved only the first 500 characters of the answer, so it cannot be verified; the new harness saves full answers).
- Models frozen: worker `qwen3:4b` (the proposal's `qwen3.5:4b` was a naming error, corrected in the proposal), supervisor `gemini-3.8-flash` (health test passed Oct 17).
- Question sets frozen: warmup = question index 2 for all 20 patients (20 questions); eval = indices 3 to 12 for all 20 patients (200 questions). Zero overlap with the pilot questions, verified. Padding frozen at N = 0 (each question's own case only).
- Subscription answer for Prof. Cai: I have a Claude subscription and the free 1-year Google AI Pro student offer. Whether the student plan raises the Gemini API quota is unverified; I will learn it from the Week 3 and 4 runs.

**Oct 17 to 18 (harness and baselines):**
- Built the Week 3 harness (`harness/`): data loader, rule-based grader, JSONL logger (full prompts and answers, protocol/worker/supervisor/grading labels on every row), 27B local-only runner, cloud-only runner, 4B chunked runner, and a regrade tool.
- Found and fixed a harness bug: Ollama's `think` flag was passed inside `options`, where the API silently ignores it, so "think-off" runs were actually thinking. `think` is a top-level chat parameter. The affected run was relabeled honestly as the think-on reference.
- Found that a 500-token output cap cripples both think conditions (think-on: thinking consumed the whole budget, leaving empty answers; think-off: the model narrates evidence in the open and never reaches its choice). Both conditions now use a 1200-token budget.
- 4B local-only design: each record is split into ~1,500-token chunks; one worker call per chunk using native Ollama JSON mode (`{"letter": "B"}`), because the 4B cannot answer reliably in free text (it narrates the excerpt instead); final answer is the majority vote over chunk letters, ties broken by the earliest chunk. Design note: `docs/decision-4b-local-only-design.md`.
- Baseline results on the 20 warmup questions (all at the 1200-token budget):
  - 27B local-only, think on: **12/20 correct**, 5 flagged for hand-check.
  - 27B local-only, think off: **3/20 correct by the rule-based grader, 15 flagged for hand-check.** The flagged rows decide whether this is an answer-format problem or a real quality drop; hand-check in progress.
  - 4B local-only (chunked, JSON-mode votes): **10/20 correct**, 0 hand-checks.
  - Cloud-only (gemini-3.8-flash alone): in progress (free-tier rate limit, 5 requests/minute, interrupted the first run; resuming).
- Measured supervisor cost: one cloud-only question = 11,967 input / 48 output / 142 thinking tokens, about $0.0097.

## What is pending (this week)

- Finish the cloud-only baseline (20/20).
- Hand-check the flagged baseline rows (think-off 15, think-on 5) and finalize the baseline table.
- Finalize the cost estimate with Prof. Cai's breakdown and the billing decision.
- Write and submit Weekly261021 (due Wed Oct 21, night).

## Open decisions

1. Billing: the lab has no funding. Options: (a) flash-lite free tier (~500 requests/day, about 24 days, supervisor stated as flash-lite); (b) paid gemini-3.8-flash (about $91 plus a 30% debugging allowance); (c) check whether the Google AI Pro student offer raises the API quota first. The 3.8-flash free tier is only ~20 requests/day, so free-only on 3.8-flash is not feasible for the full study. Decision in Weekly261021.
2. Weekly261014 upload: still needs Prof. Cai's shared-folder link.
