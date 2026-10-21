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
- Baseline results on the 20 warmup questions (all at the 1200-token budget), after hand-checking every answer the rule-based grader could not parse:
  - 27B local-only, think on: **15/20 correct**. The grader alone scored 12/20; of the 5 flagged answers, 3 were correct in substance and 2 came back empty because the hidden thinking consumed the full output budget.
  - 27B local-only, think off: **15/20 correct**. The grader alone scored 3/20, which was an artifact: think-off answers usually state their choice as a bare trailing letter or a bare "(C)" line, formats the grader does not parse. Of the 15 flagged, 12 were correct in substance, 1 was genuinely wrong, and 2 hit the output cap mid-reasoning and never chose.
  - 4B local-only (chunked, JSON-mode votes): **10/20 correct**, 0 hand-checks.
  - Cloud-only (gemini-3.8-flash alone): **14/20 correct** (grader 12/20, both flagged answers correct in substance). Total measured cost $0.1991, about $0.00996 per question.
  - Thinking-off verdict: think-off and think-on finish level at 15/20, so turning thinking off does not reduce quality on the warmup set. Full verdict table: `docs/handcheck-results.md`.
- Measured supervisor cost: one cloud-only question = 11,967 input / 48 output / 142 thinking tokens, about $0.0097.

## What is pending (this week)

- Weekly261021 report: complete, with the final baseline table, the thinking-off verdict, and the reconciled cost estimate (9,204 calls including the 30% debugging allowance; $66.05 expected, $85.11 worst case). Submitting Wednesday night, Oct 21.
- Billing decision: self-pay on the gemini-3.8-flash paid tier, as stated in Weekly261021.

## Open decisions

- None on the study itself. Billing is decided: self-pay on the gemini-3.8-flash paid tier (about $66 expected, $85 worst case for the full study). The lab has no funding, the 3.8-flash free tier (about 20 requests/day) cannot carry the study's 9,204 calls, and my Claude and Google AI Pro subscriptions do not include API credits.
