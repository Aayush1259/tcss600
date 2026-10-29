"""MinionS baseline, 4B worker, on the frozen question sets. Machine: main laptop (AAYUSHLAPTOP, WSL2).

Published MinionS routine from the Minions package (Minions class):
the supervisor decomposes the task into small jobs over the documents,
the local worker solves them in parallel, the supervisor combines the
answers, repeating in rounds as needed. This is RQ2's MinionS reference
point (no review vs review every step vs MinionS).

Adapted from the proven bosgame 27B runner (minions_run_27b_warmup.py),
with laptop specifics:
- Worker: qwen3:4b on the laptop's Ollama, num_ctx 2048 (frozen 4B
  context). max_tokens 800 (a cap, not a target): chunk (~750 tokens)
  plus task (~200) plus output fits the 2048 window.
- Package: the Minions repo already sits at /home/aayush/minions
  (matches bosgame: structured_output_schema, JobOutput). Inserted
  on sys.path. No site-packages hack: the laptop venv is clean (the
  import test passed with only the minions path inserted).
- Environment: run ONLY with ~/is-env/bin/python. System python3
  lacks the study packages and its pip is blocked by PEP 668.
- Data: the laptop's own data_loader.py refuses padding N>0, so
  records come from data_loader_frozen.py (the freeze logic copied
  verbatim, paths via ~), imported as data_loader. The laptop's
  data_loader.py is untouched. Records are built explicitly with
  get_question(bench, patient, qidx, pad_n=args.padding) over the
  frozen id lists, so the padding is explicit and identical to the
  27B run.
- Grading: the laptop's grader.py predates the freeze JSON functions,
  so the freeze parse_json_answer + grade_json + matches are bundled
  here verbatim (identical semantics to the 27B runner). The laptop's
  grader.py is NOT overwritten.

Frozen setting (2026-10-24, freeze record section 10): padding N=2,
supervisor gemini-3.8-flash with thinking_level low, final answers
JSON (grade_json). The Minions class already extracts a JSON "answer"
field as its final answer, so the task text instructs that the answer
field hold ONLY the option letter.

Worker outputs (Oct 24 smoke fix, carried over): worker calls are
constrained by the Ollama client's native structured_output_schema
(JobOutputSchema below, mirroring the package's JobOutput), the same
mechanism as the local-only constrained-JSON fix. Note: constructor
prompt overrides for worker/synthesis prompts do NOT work in this
package (their lines read DEFAULT or kwargs.get(...), so the default
always wins); only decompose prompts are kwarg-overridable.

Robustness: each question's Minions call is wrapped in try/except,
so one failed question logs one failed row (predicted None, match
False, hand-check flagged) instead of killing the run.

Cost caps (freeze section 7): every run carries --max-usd (hard stop
on estimated supervisor cost summed from per-question remote usage)
and --max-calls (hard stop on counted supervisor call attempts,
retries included). The run stops between questions when a cap is hit
and the summary records the stop. No artificial pacing: MinionS rounds
are sequential and worker jobs take minutes, so supervisor calls are
naturally spaced.

Usage and timing come back per question in the Minions result
(remote_usage, local_usage, timing) and are logged per row. GPU
seconds are NOT reported by the Minions routine, so worker_gpu_s is
logged as None; local latency is timing.local_call_time. The
Minions conversation logs go to minion_logs/ next to this script and,
like run logs, contain full benchmark text and stay OUT of the repo.
Rows do NOT duplicate the documents: split, patient, question index,
padding, rounds, chunking, and retrieval settings reproduce a run
from data_loader deterministically; the Minions log file is recorded
per row for the full trace.

Usage (run from anywhere; the script directory supplies the sibling
imports, and minion_logs/ sits next to this script):
    ~/is-env/bin/python minions_run_4b_warmup.py --split warmup --limit 1 --padding 2
        # smoke: one question, then read final_answer, remote_usage,
        # timing, and rounds used before sizing the full run
    ~/is-env/bin/python minions_run_4b_warmup.py --split warmup --padding 2
        # full warmup (20 questions; run inside tmux for an overnight run,
        # laptop plugged in and awake)

Requires the Ollama server running on the laptop and GEMINI_API_KEY set.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, "/home/aayush/minions")

from minions.minions import Minions  # noqa: E402
from minions.clients.ollama import OllamaClient  # noqa: E402
from minions.clients.gemini import GeminiClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import data_loader_frozen as data_loader  # noqa: E402
import runlog  # noqa: E402

WORKER_MODEL = "qwen3:4b"
WORKER_NUM_CTX = 2048  # frozen 4B context
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
        # some Ollama versions reject with 500. Model presence was
        # verified separately with `ollama list`.
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


# Freeze grading functions, copied verbatim from the workspace freeze
# grader.py (2026-10-24). The laptop's grader.py predates them, so
# they live here to guarantee identical grading semantics to the 27B
# runner. The laptop's grader.py is untouched.
LETTER_GOLD = re.compile(r"^[A-E](\s*(and|&|,)\s*[A-E])*$")
JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def matches(letter, gold, options):
    """Match a predicted letter against the gold answer.

    Gold is sometimes a bare letter/combo ('B', 'B and C') and sometimes
    the full option text ('Neutropenic fever'). Letter mode applies ONLY
    when the gold has the shape of a letter code.
    """
    if not letter:
        return False
    g = str(gold).strip()
    if LETTER_GOLD.match(g):
        return letter in re.findall(r"[A-E]", g)
    return options.get(letter, "").strip().lower() == g.lower()


def parse_json_answer(text):
    """Returns the single letter from a JSON-only answer, or None."""
    if text is None:
        return None
    t = text.strip()
    m = JSON_FENCE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    ans = obj.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if len(ans) == 1 and ans in "ABCDE":
            return ans
    return None


def grade_json(text, gold, options):
    """JSON-only grading for Week 4 runs. Same dict shape as grade()."""
    letter = parse_json_answer(text)
    return {
        "letter": letter,
        "match": matches(letter, gold, options),
        "needs_handcheck": letter is None,
        "method": "json" if letter else "json-malformed",
    }


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
                         "export it before running.")

    worker = QuietOllamaClient(model_name=WORKER_MODEL,
                               num_ctx=WORKER_NUM_CTX,
                               structured_output_schema=JobOutputSchema)
    worker.max_tokens = 800
    supervisor = CountingRetryGeminiClient(
        model_name=SUPERVISOR_MODEL,
        thinking_level=SUPERVISOR_THINKING_LEVEL)
    supervisor.max_tokens = 2048
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minions = Minions(local_client=worker, remote_client=supervisor,
                      max_rounds=args.max_rounds,
                      log_dir=os.path.join(script_dir, "minion_logs"))

    bench = data_loader.load_benchmark()
    id_path = (data_loader.WARMUP_IDS if args.split == "warmup"
               else data_loader.EVAL_IDS)
    recs = [data_loader.get_question(bench, p, i, pad_n=args.padding)
            for p, i in data_loader.load_id_list(id_path)]
    if args.limit:
        recs = recs[:args.limit]

    log_path = runlog.new_log(
        f"minions_4b_{args.split}_pad{args.padding}_json")
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
        g = grade_json(grade_text, rec["gold"], rec["options"])
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
