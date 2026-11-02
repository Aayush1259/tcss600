"""Cloud-only baseline on the frozen question sets. Machine: bosgame.

Supervisor alone on the full document, no worker (RQ2's accuracy ceiling
and most-expensive reference point). Supervisor is gemini-3.8-flash,
frozen 2026-10-17 after the health test returned OK.

Week 4 freeze (2026-10-24, see frozen-rules-2026-10-24.md): final
answers are JSON only (grade_json); supervisor thinking level is
frozen at low and logged per row; padding is a parameter; any
settings change starts a fresh log (--fresh).

Labels on every row: protocol cloud-only | worker none |
supervisor gemini-3.8-flash, thinking low | grading json + hand-check
queue. Thinking tokens are recorded separately: they bill as output.

Usage:
    python3 cloud_only.py --split warmup            # all 20 warmup
    python3 cloud_only.py --split warmup --limit 1  # smoke test (1 API call)
    python3 cloud_only.py --split eval --max-calls 25 --max-usd 0.50
        # hard budget caps: the run stops by itself at either cap
    python3 cloud_only.py --split warmup --fresh --padding 2 \
        --max-calls 25 --max-usd 0.50
        # Week 4 gap test: fresh log, padded warmup, JSON answers

Needs GEMINI_API_KEY in the environment. Run from ~.
"""
import argparse
import glob
import json
import os
import sys
import time

_sp = "/home/localai/is-env/lib/python3.14/site-packages"
if os.path.isdir(_sp) and _sp not in sys.path:
    sys.path.insert(0, _sp)

from google import genai  # noqa: E402

import data_loader  # noqa: E402
import grader  # noqa: E402
import runlog  # noqa: E402

SUPERVISOR_MODEL = "gemini-3.8-flash"  # frozen 2026-10-17
# Week 4 freeze (2026-10-24): supervisor thinking level set explicitly
# to low, never the API default. Logged per row; stated in the report
# and the pre-registration.
THINKING_LEVEL = "low"
RETRY_DELAYS = [15, 30, 60, 120, 240]

# Budget guard (added Oct 22): set from --max-calls in main(). Every API
# attempt, retries included, counts; the run stops before any attempt that
# would pass the cap, so a retry storm cannot run up calls on its own.
ATTEMPTS = 0
MAX_CALLS = None


class BudgetStop(Exception):
    """Raised when the run's call cap is reached; main() stops cleanly."""


def ask_supervisor(client, prompt):
    global ATTEMPTS
    last = None
    for attempt in range(6):
        if MAX_CALLS is not None and ATTEMPTS >= MAX_CALLS:
            raise BudgetStop(f"call cap reached ({MAX_CALLS} attempts)")
        ATTEMPTS += 1
        try:
            t0 = time.time()
            r = client.models.generate_content(
                model=SUPERVISOR_MODEL, contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_level=THINKING_LEVEL)))
            dt = time.time() - t0
            u = getattr(r, "usage_metadata", None)
            return {
                "text": r.text,
                "prompt_tokens": getattr(u, "prompt_token_count", None),
                "completion_tokens": getattr(u, "candidates_token_count", None),
                "thinking_tokens": getattr(u, "thoughts_token_count", 0) or 0,
                "wall_time_s": round(dt, 1),
            }
        except Exception as e:
            last = e
            msg = str(e)
            # Oct 18: the free tier is 5 req/min; faster pacing hits 429.
            # The API's RetryInfo suggests ~22s; use at least 30s.
            if ("503" in msg or "429" in msg) and attempt < 5:
                delay = RETRY_DELAYS[attempt]
                if "429" in msg:
                    delay = max(delay, 30)
                    code = "429"
                else:
                    code = "503"
                print(f"  {code}, retry in {delay}s", flush=True)
                time.sleep(delay)
            else:
                raise
    raise last


def latest_log_for(split):
    """Newest non-regraded log for this split, for resume after a crash."""
    cands = sorted(glob.glob(os.path.join(
        runlog.LOG_DIR, f"cloud_only_{split}_*.jsonl")))
    cands = [c for c in cands if "regraded" not in c]
    return cands[-1] if cands else None


def done_ids(log_path):
    done = set()
    if log_path:
        for line in open(log_path):
            try:
                r = json.loads(line)
                done.add((r["patient"], r["q_idx"]))
            except Exception:
                pass
    return done


def build_prompt(rec):
    opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
    doc = "\n\n---\n\n".join(rec["documents"])
    return (
        "You are answering a multiple-choice medical question. Read the "
        "records carefully, then respond with JSON only, exactly in "
        "this shape, with X the single letter of the correct option:\n"
        '{"answer": "X"}\n\n'
        f"{rec['question']}\n{opts}\n\nMedical records:\n{doc}"
    )


def main():
    global MAX_CALLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None,
                    help="stop the run after this many API attempts")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="stop the run once estimated spend reaches this")
    ap.add_argument("--padding", type=int, default=0,
                    help="distractor patients per question (Week 4 freeze)")
    ap.add_argument("--fresh", action="store_true",
                    help="start a fresh log even if a log exists (required "
                    "on any settings change: JSON, thinking, padding)")
    args = ap.parse_args()
    MAX_CALLS = args.max_calls
    if args.max_calls is not None or args.max_usd is not None:
        print(f"budget caps for this run: max-calls={args.max_calls} "
              f"max-usd={args.max_usd}", flush=True)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    recs = data_loader.load_split(args.split, pad_n=args.padding)
    if args.limit:
        recs = recs[:args.limit]

    log_path = None if args.fresh else latest_log_for(args.split)
    done = set() if args.fresh else done_ids(log_path)
    if log_path and done:
        print(f"resuming {log_path} ({len(done)} already done)")
    else:
        log_path = runlog.new_log(
            f"cloud_only_{args.split}_pad{args.padding}_json")
        print(f"logging to {log_path}")

    spent = 0.0
    stopped_on_budget = False
    for i, rec in enumerate(recs):
        if (rec["patient"], rec["q_idx"]) in done:
            print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
                  f"already done, skipping", flush=True)
            continue
        if args.max_usd is not None and spent >= args.max_usd:
            print(f"budget cap reached: estimated spend ${spent:.4f} "
                  f">= --max-usd {args.max_usd}; stopping. "
                  f"Relaunch to resume.", flush=True)
            stopped_on_budget = True
            break
        prompt = build_prompt(rec)
        try:
            out = ask_supervisor(client, prompt)
        except BudgetStop as e:
            print(f"budget cap reached: {e}; stopping. "
                  f"Relaunch to resume.", flush=True)
            stopped_on_budget = True
            break
        g = grader.grade_json(out["text"], rec["gold"], rec["options"])
        cost = runlog.supervisor_cost_usd(
            out["prompt_tokens"], out["completion_tokens"],
            out["thinking_tokens"])
        spent += cost
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "cloud-only",
            "worker": None,
            "supervisor": f"{SUPERVISOR_MODEL}, thinking {THINKING_LEVEL}",
            "supervisor_thinking_level": THINKING_LEVEL,
            "grading": f"json ({g['method']}) + hand-check queue",
            "output_format": "json",
            "padding_n": rec["padding_n"],
            "padding_patients": rec["padding_patients"],
            "predicted": g["letter"],
            "gold": rec["gold"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "wall_time_s": out["wall_time_s"],
            "supervisor_prompt_tokens": out["prompt_tokens"],
            "supervisor_completion_tokens": out["completion_tokens"],
            "supervisor_thinking_tokens": out["thinking_tokens"],
            "est_cost_usd": round(cost, 6),
            "prompt": prompt,
            "answer": out["text"],
        })
        print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
              f"pred={g['letter']} gold={rec['gold']} match={g['match']} "
              f"tok={out['prompt_tokens']}/{out['completion_tokens']}"
              f"(+{out['thinking_tokens']} think) ${cost:.4f} "
              f"(run total ${spent:.4f}, {ATTEMPTS} calls)", flush=True)
        # Free tier is 5 req/min; pace calls to stay under it (Oct 18).
        time.sleep(15)

    runlog.summarize(log_path)


if __name__ == "__main__":
    main()
