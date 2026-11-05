"""MinionS 27B fix test (October 31): the Week 4 MinionS warmup runner with a
stricter synthesis prompt, run on warmup patients 01 and 02. Machine: bosgame.
The test ran without use_async=True, so the worker answered one job per round
and its result does not stand. See correction-2026-11-04.md.

Published MinionS routine from the Minions package (Minions class):
the supervisor decomposes the task into small jobs over the documents,
the local worker solves them in parallel, the supervisor combines the
answers, repeating in rounds as needed. This is RQ2's MinionS reference
point (no review vs review every step vs MinionS).

Frozen setting (2026-10-24, freeze record section 10): padding N=2,
27B num_ctx 32768, supervisor gemini-3.8-flash with thinking_level low,
final answers JSON (grade_json). The Minions class already extracts a
JSON "answer" field as its final answer, so the task text instructs
that the answer field hold ONLY the option letter.

Worker outputs (Oct 24 smoke fix): the first smoke crashed because a
worker returned a fenced Python class definition instead of a JSON
JobOutput, and the class validates worker output with pydantic
directly (no retry, no fence-stripping). Worker calls are therefore
constrained by the Ollama client's native structured_output_schema
(JobOutputSchema below, mirroring the package's JobOutput), the same
mechanism as the local-only constrained-JSON fix. Note: constructor
prompt overrides for worker/synthesis prompts do NOT work in this
package (their lines read DEFAULT or kwargs.get(...), so the default
always wins); only decompose prompts are kwarg-overridable. If a
future failure needs a prompt change, assign the attribute after
construction instead.

Robustness: each question's Minions call is wrapped in try/except,
so one failed question logs one failed row (predicted None, match
False, hand-check flagged) instead of killing the run.

Clients (Week 2 fixes carried over from the pilot script):
- Worker: QuietOllamaClient (think forced off on every call, built-in
  probe skipped because Ollama 0.40.3 rejects system-only messages
  with 500), num_ctx 32768, max_tokens 1200 (a cap, not a target;
  short subtask answers cost nothing extra).
- Supervisor: CountingRetryGeminiClient (gemini-3.8-flash,
  thinking_level low, max_tokens 2048, retries on 503 with backoff,
  counts every supervisor call attempt for the caps).

Cost caps (freeze section 7): every run carries --max-usd (hard stop
on estimated supervisor cost summed from per-question remote usage)
and --max-calls (hard stop on counted supervisor call attempts,
retries included). The run stops between questions when a cap is hit
and the summary records the stop. No artificial pacing: MinionS rounds
are sequential and worker jobs take minutes, so supervisor calls are
naturally spaced. Worst-case cost bound for the defaults: 2 supervisor
calls per round, 5 rounds, about $0.027 per call at N=2 prompt sizes
gives about $0.27 per question, about $5.40 for 20 questions; the
$10.00 default cap covers that with headroom inside the $20 stage.

Usage and timing come back per question in the Minions result
(remote_usage, local_usage, timing) and are logged per row. GPU
seconds are NOT reported by the Minions routine, so worker_gpu_s is
logged as None; local latency is timing.local_call_time. The
Minions conversation logs go to minion_logs/ and, like run logs,
contain full benchmark text and stay OUT of the repo. Rows do NOT
duplicate the documents: split, patient, question index, padding,
rounds, chunking, and retrieval settings reproduce a run from
data_loader deterministically; the Minions log file is recorded per
row for the full trace.

Usage (the fix test ran on the first 2 warmup questions):
    python3 minions_run_27b_warmup_fix.py --split warmup --limit 2 --padding 2

Requires the Ollama server running on bosgame and GEMINI_API_KEY set.
Run from ~.
"""
import argparse
import os
import sys
import time

# Fix for the Oct 13 import-shadowing lesson: ~/ollama/ shadows the
# installed `ollama` package unless site-packages comes first. The
# Minions clients import ollama internally, so this must precede them.
_sp = "/home/localai/is-env/lib/python3.14/site-packages"
if os.path.isdir(_sp) and _sp not in sys.path:
    sys.path.insert(0, _sp)

sys.path.insert(0, "/home/localai/minions")

from minions.minions import Minions  # noqa: E402
from minions.clients.ollama import OllamaClient  # noqa: E402
from minions.clients.gemini import GeminiClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import data_loader  # noqa: E402
import grader  # noqa: E402
import runlog  # noqa: E402

WORKER_MODEL = "qwen3.8:27b"
WORKER_NUM_CTX = 32768  # frozen N=2 context
SUPERVISOR_MODEL = "gemini-3.8-flash"
SUPERVISOR_THINKING_LEVEL = "low"  # frozen, passed natively by the client
# 3.8-flash intro prices through Dec 31 2026 (thinking bills as output).
PRICE_IN = 0.75 / 1e6
PRICE_OUT = 3.75 / 1e6


class JobOutputSchema(BaseModel):
    """Mirror of the package's JobOutput, for constrained worker decoding.

    The package's JobOutput has explanation (required str), citation
    (optional str), answer (optional str). This schema requires all
    three as plain strings: a string is accepted for the optional
    fields, and requiring them avoids null-vs-missing edge cases in
    the grammar. The worker emits "None" or "" for no-finding chunks,
    which the supervisor's generated transform filter handles.
    """
    explanation: str
    citation: str
    answer: str


class QuietOllamaClient(OllamaClient):
    """Ollama worker: capped context, thinking forced off on every call."""

    def __init__(self, model_name=WORKER_MODEL, num_ctx=WORKER_NUM_CTX,
                 **kwargs):
        super().__init__(model_name=model_name, num_ctx=num_ctx, **kwargs)

    def _ensure_model_available(self):
        # Skip the built-in probe: it sends a system-only message that
        # Ollama 0.40.3 rejects with 500. Model presence was verified
        # separately with `ollama list`.
        return

    def chat(self, messages, **kwargs):
        kwargs["think"] = False
        return super().chat(messages, **kwargs)

    def achat(self, messages, **kwargs):
        # Passthrough that still forces think off whether the parent
        # achat is sync or async: returning the parent's result (or
        # coroutine) unchanged keeps the caller's contract intact.
        kwargs["think"] = False
        return super().achat(messages, **kwargs)


class CountingRetryGeminiClient(GeminiClient):
    """Gemini supervisor: retry on 503s, count every call attempt."""

    def __init__(self, model_name=SUPERVISOR_MODEL,
                 thinking_level=SUPERVISOR_THINKING_LEVEL, **kwargs):
        super().__init__(model_name=model_name,
                         thinking_level=thinking_level, **kwargs)
        self.call_count = 0

    def chat(self, messages, **kwargs):
        delays = [15, 30, 60, 120, 240]
        for attempt in range(6):
            self.call_count += 1
            try:
                return super().chat(messages, **kwargs)
            except Exception as e:
                if "503" in str(e) and attempt < 5:
                    print(f"Gemini 503, retrying in {delays[attempt]}s "
                          f"(attempt {attempt + 2}/6)", flush=True)
                    time.sleep(delays[attempt])
                else:
                    raise

    def achat(self, messages, **kwargs):
        # Count only: retrying an async call needs awaiting, which this
        # passthrough deliberately avoids so it works whether the parent
        # achat is sync or async. The smoke test verifies which method
        # the Minions class actually uses (count must move).
        self.call_count += 1
        return super().achat(messages, **kwargs)


def remote_cost_usd(usage):
    """Estimated supervisor cost from a Minions remote_usage dict.

    Key names are tried flexibly because the usage shape is not
    documented in the package; the runner prints the raw dict per
    question, so a miss shows up immediately instead of silently
    zeroing the cost. Thinking bills as output at 3.8-flash prices.
    """
    if not isinstance(usage, dict):
        return 0.0, {"input_tokens": 0, "output_tokens": 0,
                     "thinking_tokens": 0}
    inp = (usage.get("input_tokens", usage.get(
        "prompt_tokens", usage.get("prompt_token_count", 0))) or 0)
    out = (usage.get("output_tokens", usage.get(
        "completion_tokens", usage.get("candidates_token_count", 0))) or 0)
    think = usage.get("thinking_tokens", 0) or 0
    cost = inp * PRICE_IN + (out + think) * PRICE_OUT
    return cost, {"input_tokens": inp, "output_tokens": out,
                  "thinking_tokens": think}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--padding", type=int, default=2,
                    help="distractor patients per question (frozen N=2)")
    ap.add_argument("--max-rounds", type=int, default=5,
                    help="Minions class default; rounds actually used are "
                         "logged per row from len(meta)")
    ap.add_argument("--max-usd", type=float, default=10.00,
                    help="hard stop on summed estimated supervisor cost")
    ap.add_argument("--max-calls", type=int, default=300,
                    help="hard stop on counted supervisor call attempts")
    args = ap.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY is not set in this shell; "
                         "export it before running (the pilot and the "
                         "cloud runner use the same variable).")

    worker = QuietOllamaClient(model_name=WORKER_MODEL,
                               num_ctx=WORKER_NUM_CTX,
                               structured_output_schema=JobOutputSchema)
    worker.max_tokens = 1200
    supervisor = CountingRetryGeminiClient(
        model_name=SUPERVISOR_MODEL,
        thinking_level=SUPERVISOR_THINKING_LEVEL)
    supervisor.max_tokens = 2048
    minions = Minions(local_client=worker, remote_client=supervisor,
                      max_rounds=args.max_rounds, log_dir="minion_logs")

    # Quick fix test, Oct 31 2026, Week 5 MinionS regressions.
    # Constructor kwargs do NOT override synthesis prompts in this
    # package (REMOTE_SYNTHESIS_* defaults win), so assign the
    # attributes after construction. Cot prompt placeholders must
    # stay exactly {scratchpad}, {question}, {extractions}; no other
    # braces in the cot text, or .format will fail. Json prompt is
    # used as-is, so keep its doubled braces exactly as in the
    # original. Rule added, nothing else changed in the originals:
    # if collected job outputs do not contain specific, cited
    # evidence for the named patient and the exact fact asked, the
    # evidence is insufficient, do NOT guess from domain knowledge,
    # prior knowledge, benchmark familiarity, or most common /
    # most standard answers, and do NOT treat absence in processed
    # chunks as proof that a value is Not specified in the full
    # record. In that case the json decision must be exactly
    # "request_additional_info", answer null, feedback specific
    # missing patient record, section, and fact to search next.
    # Only use "provide_final_answer" when every part of the
    # answer is supported by a citation in collected job outputs.
    minions.synthesis_cot_prompt = """\
Now synthesize the findings from multiple junior workers (LLMs).
Your task is to analyze the collected information and think step-by-step about whether we can answer the question.
Be brief and concise in your analysis.

## Previous Progress
{scratchpad}

## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

## Instructions
Think step-by-step about:
1. What information we have gathered
2. Whether it is sufficient to answer the question
3. If not sufficient, what specific information is missing
4. If sufficient, how we would calculate or derive the answer
5. If there are conflicting answers:
--> Use citations to select the correct answer if there are conflicting answers.

## Insufficient evidence rule
If the collected job outputs do not contain specific, cited evidence for the named patient and the exact fact asked, the evidence is insufficient, even if you know a likely answer from domain knowledge, prior knowledge, benchmark familiarity, or because an answer is most common or most standard. Do not guess in that case. Do not treat absence in the processed chunks as proof that a value is Not specified in the full record. Not specified is only supported if a retrieved record for that patient was examined and no value was found in the relevant section, not if no relevant section was retrieved at all. If insufficient, state exactly what patient record, section, and fact is missing, so the next round can search for it.

Be brief and concise. No need for structured output - just think through the steps. You MUST respond in markdown format (for $ signs make sure to use escape character \\ before the $ sign).
"""
    minions.synthesis_json_prompt = """\
Based on your analysis, return a single JSON object with no triple backticks or extra text. The JSON should have this exact structure:

{{
  "explanation": "",
  "feedback": null,
  "decision": "",
  "answer": null,
  "scratchpad": ""
}}

Field Descriptions:
- explanation: A brief statement of your reasoning.
- feedback: Specific information to look for, if needed. Use null if not applicable.
- decision: Either "provide_final_answer" or "request_additional_info".
- answer: The final answer if providing one; null otherwise.
- scratchpad: Summary of gathered information and current analysis for future reference.

Insufficient evidence rule: if your analysis says the evidence is insufficient, or if any part of the answer would rely on domain knowledge, prior knowledge, benchmark familiarity, or a most common answer rather than a citation in the collected job outputs, or if absence in processed chunks is the only support for a Not specified answer, then decision must be exactly "request_additional_info", answer must be null, and feedback must name the specific patient record, section, and fact to search next. Only use "provide_final_answer" when every part of the answer is supported by a citation in the collected job outputs.

Ensure the response is a valid JSON object without any additional text or formatting.

"""

    recs = data_loader.load_split(args.split, pad_n=args.padding)
    if args.limit:
        recs = recs[:args.limit]

    log_path = runlog.new_log(
        f"minions_27b_fix_{args.split}_pad{args.padding}_json")
    print(f"logging to {log_path}")
    print(f"budget caps for this run: max-calls={args.max_calls} "
          f"max-usd={args.max_usd:.2f}")

    run_usd = 0.0
    stopped = None
    for i, rec in enumerate(recs):
        opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
        task = (
            f"{rec['question']}\n{opts}\n\n"
            f"Your final answer's \"answer\" field must contain ONLY the "
            f"single letter (A, B, C, D, or E) of the correct option. "
            f"Do not put explanation text in the answer field.")
        t0 = time.time()
        try:
            result = minions(
                task=task,
                doc_metadata="LongHealth clinical records (question patient "
                             "plus distractor patients)",
                context=rec["documents"])
        except Exception as e:
            # One bad worker output (or any Minions failure) costs one
            # logged row, not the whole run. The partial supervisor
            # spend on this question is unmeasured, so est_cost_usd is
            # 0.0 here; the supervisor call count still moved.
            print(f"Minions call failed for {rec['patient']} "
                  f"q{rec['q_idx']}: {e}", flush=True)
            runlog.log_row(log_path, {
                "split": args.split,
                "patient": rec["patient"],
                "q_idx": rec["q_idx"],
                "q_no": rec["q_no"],
                "protocol": "minions",
                "worker": (f"{WORKER_MODEL}, ctx {WORKER_NUM_CTX}, "
                           f"think off"),
                "supervisor": (f"{SUPERVISOR_MODEL}, thinking "
                               f"{SUPERVISOR_THINKING_LEVEL}"),
                "grading": "json (error) + hand-check queue",
                "output_format": "json",
                "padding_n": rec["padding_n"],
                "padding_patients": rec["padding_patients"],
                "worker_num_ctx": WORKER_NUM_CTX,
                "max_rounds": args.max_rounds,
                "rounds_used": None,
                "num_tasks_per_round": 3,
                "chunk_fn": "chunk_by_section",
                "use_retrieval": None,
                "predicted": None,
                "gold": rec["gold"],
                "match": False,
                "needs_handcheck": True,
                "wall_time_s": round(time.time() - t0, 2),
                "worker_time_s": None,
                "supervisor_time_s": None,
                "worker_gpu_s": None,
                "local_usage": None,
                "remote_usage": None,
                "remote_cost_usd": 0.0,
                "est_cost_usd": 0.0,
                "run_total_usd": round(run_usd, 4),
                "supervisor_calls": supervisor.call_count,
                "minions_log_file": None,
                "final_answer_raw": "",
                "error": str(e),
                "task": task,
                "doc_chars": sum(len(d) for d in rec["documents"]),
                "doc_count": len(rec["documents"]),
            })
            continue
        fa = (result.get("final_answer") or "").strip()
        if len(fa) == 1 and fa in "ABCDE":
            grade_text = '{"answer": "%s"}' % fa
        else:
            grade_text = fa
        g = grader.grade_json(grade_text, rec["gold"], rec["options"])
        cost, tok = remote_cost_usd(result.get("remote_usage"))
        run_usd += cost
        timing = result.get("timing") or {}
        rounds_used = len(result.get("meta") or [])
        print(f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
              f"final_answer={fa!r} pred={g['letter']} "
              f"gold={rec['gold']} match={g['match']} "
              f"rounds={rounds_used} "
              f"q_cost=${cost:.4f} (run total ${run_usd:.4f}, "
              f"{supervisor.call_count} calls)", flush=True)
        print(f"  remote_usage={result.get('remote_usage')}", flush=True)
        print(f"  local_usage={result.get('local_usage')}", flush=True)
        print(f"  timing={timing}", flush=True)
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "minions",
            "worker": (f"{WORKER_MODEL}, ctx {WORKER_NUM_CTX}, "
                       f"think off"),
            "supervisor": (f"{SUPERVISOR_MODEL}, thinking "
                           f"{SUPERVISOR_THINKING_LEVEL}"),
            "grading": f"json ({g['method']}) + hand-check queue",
            "output_format": "json",
            "padding_n": rec["padding_n"],
            "padding_patients": rec["padding_patients"],
            "worker_num_ctx": WORKER_NUM_CTX,
            "max_rounds": args.max_rounds,
            "rounds_used": rounds_used,
            "num_tasks_per_round": 3,
            "chunk_fn": "chunk_by_section",
            "use_retrieval": None,
            "predicted": g["letter"],
            "gold": rec["gold"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "wall_time_s": timing.get("total_time"),
            "worker_time_s": timing.get("local_call_time"),
            "supervisor_time_s": timing.get("remote_call_time"),
            "worker_gpu_s": None,
            "local_usage": result.get("local_usage"),
            "remote_usage": result.get("remote_usage"),
            "remote_cost_usd": round(cost, 4),
            "est_cost_usd": round(cost, 4),
            "run_total_usd": round(run_usd, 4),
            "supervisor_calls": supervisor.call_count,
            "minions_log_file": result.get("log_file"),
            "final_answer_raw": fa,
            "task": task,
            "doc_chars": sum(len(d) for d in rec["documents"]),
            "doc_count": len(rec["documents"]),
        })
        if run_usd >= args.max_usd:
            stopped = "max-usd"
            print(f"STOPPED early: run total ${run_usd:.4f} reached "
                  f"max-usd ${args.max_usd:.2f}", flush=True)
            break
        if supervisor.call_count >= args.max_calls:
            stopped = "max-calls"
            print(f"STOPPED early: {supervisor.call_count} supervisor "
                  f"calls reached max-calls {args.max_calls}", flush=True)
            break

    runlog.summarize(log_path)
    if stopped:
        print(f"run ended early on {stopped}; remaining questions unrun",
              flush=True)


if __name__ == "__main__":
    main()
