# Frozen rules, Week 4 (2026-10-24, after Cai's Oct 24 feedback)

These rules are frozen. Changing one means a new frozen version, a note
in this file, and a fresh log for every run after the change. Old logs
are never rewritten and never resumed under new settings.

## 1. Final-answer JSON output (all models)

Every final answer is JSON only, one option letter:

{"answer": "B"}

Applies to: the 27B worker, the cloud supervisor, every protocol
aggregation step. The 4B worker already uses JSON mode for chunk votes;
its final vote is recorded in the same JSON shape.

Grading rule for new runs: the grader parses JSON only (grade_json).
A well-formed letter is graded. A malformed answer counts as wrong:
letter None, match False, flagged for the hand-check queue as a spot
check, never rescued into a correct answer. The old free-text grader
(grade) stays untouched for old logs only.

## 2. Step output structure

A question trace is a plan: an ordered list of steps. Each step
carries: step text, log-probability score, paraphrase votes for the
agreement signal. A supervisor review carries: step id, verdict (keep
or revise), revised text when revised. Aggregation output is the
final-answer JSON from section 1.

Log-probability aggregation per step: tested both ways on warmup
(mean of step token log-probs vs minimum token log-prob), and the
better separator of correct from incorrect steps is frozen here.
Until that test lands, the field is recorded but no aggregation is
frozen. Record the choice in this file the day the test lands.

## 3. Thinking levels (frozen)

Supervisor (gemini-3.8-flash): thinking_level = low, set explicitly
in the API call, stored as a constant in the runner, written into
every log row as supervisor_thinking_level, stated in the report and
in the pre-registration. No reliance on the API default.

Local workers (Ollama): think is a top-level chat parameter, never
inside options (inside options it is silently ignored). JSON runs use
think off. Think on/off stays a frozen per-condition value, recorded
per row.

## 4. Context windows

27B worker: num_ctx 16384 at padding N=0 (verified Oct 13, largest
record about 12,766 tokens). Padding N=2 uses num_ctx 32768. Padding
N=3 uses num_ctx 65536. Each level is proven by a one-question smoke
before any full run at that level.

## 5. Padding

Padding is a parameter, not a fixed value. Selection is deterministic:
the question's own record first, then the records of the next N
patients in benchmark order, wrapping to the start. The padding
patients are recorded per row (padding_patients) together with
padding_n, so any run is reproducible from its log.

Test order: N=2 first. N=3 runs only if N=2 does not open a clear
capability gap. The chosen N is frozen here and in the
pre-registration after the warmup choice.

## 6. Fresh logs on any settings change

A run under new settings (JSON output, thinking level, padding,
context, worker model) always starts a fresh log. The cloud runner
takes --fresh for this; the 27B runner always starts a fresh log.
Never resume an old log under new settings: mixed-setting rows would
corrupt the comparison.

## 7. Costing and spend tracking

Every cloud run carries a call cap (--max-calls) and a dollar cap
(--max-usd) sized to that run, retries counted. Local runs are
bounded by --limit and split.

Cumulative spend is reported as two labeled figures, never one
blended number: log-estimated cost (list-price token metric from
the run logs) and console-billed cost (Google Cloud Billing,
Reports and Transactions; credit draws first, then card). Each
weekly report carries both.

Budget staging (user, Oct 22): $20 budget alert first, plus $50
later. The budget alert is a warning line, not a cutoff; the run
caps are the cutoff.

## 8. Gap decision rule (Cai, Oct 24)

Try worker and padding combinations on warmup. Choose the setting
where the local model alone scores clearly below the cloud model
but still answers a reasonable share of questions. Rerun the
warmup baselines at the chosen setting, explain the choice and
reasons in Weekly261028, then freeze the pre-registration (gate:
Nov 3). If the worker model changes, the proposal worker line is
updated with a clear statement, and the Week 3 27B numbers stay in
the report as the reason for the change.

## 9. Deadlines and backstops

MinionS running on warmup by end of Week 4 (Oct 28), or Cai is
informed that day and scope is adjusted with him. Weekly261028 due
Wednesday Oct 28 night. Pre-registration frozen by Nov 3. Interim
memo Nov 4 to 10. Final report and code repo Dec 11.

## 10. Frozen choice, Oct 24 evening: N=2, 27B answers via schema-constrained JSON

Warmup results that fixed the choice:
- 27B prompt-based JSON at N=2: 3/20, with 16/20 malformed (valid
  but wrong-schema outputs, for example {"region": ...}).
- 27B prompt-based JSON at N=0: 17/20, 0 malformed, matching the
  free-text 17/20. This proved the N=2 failure is length-driven
  instruction loss, not JSON itself.
- 27B schema-constrained JSON at N=2 (ANSWER_SCHEMA in
  local_only_27b.py, answer enum A-E): 6/20, 0 malformed, 20/20
  parseable.
- Cloud prompt-based JSON at N=2, thinking low: 19/20, 0 malformed,
  $0.5233.

Frozen by user decision Oct 24: padding N=2 (27B num_ctx 32768);
the 27B final answers use schema-constrained JSON; the cloud final
answers use prompt-based JSON with thinking low (it complies 20/20).
The gap rule in section 8 is satisfied: 65 points (30% vs 95%),
local share 30% with every answer valid. N=3 is not run: it would
risk floor effects without adding to a 65-point gap. MinionS and
the experimental protocols run at N=2 under these rules.

Logs: local_only_27b_warmup_think-off_pad2_json_20261024_173825.jsonl
(27B 6/20), cloud_only_warmup_pad2_json_20261024_164202.jsonl
(cloud 19/20), local_only_27b_warmup_think-off_pad0_json_20261024_170042.jsonl
(27B N=0 17/20).
