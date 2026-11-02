"""Cloud-only baseline on the frozen question sets. Machine: bosgame.

Supervisor alone on the full document, no worker (RQ2's accuracy ceiling
and most-expensive reference point). Supervisor is gemini-3.8-flash,
frozen 2026-10-17 after the health test returned OK.

Labels on every row: protocol cloud-only | worker none |
supervisor gemini-3.8-flash | grading rule-based + hand-check queue.
Thinking tokens are recorded separately: they bill as output.

Usage:
    python3 cloud_only.py --split warmup            # all 20 warmup
    python3 cloud_only.py --split warmup --limit 1  # smoke test (1 API call)

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
RETRY_DELAYS = [15, 30, 60, 120, 240]


def ask_supervisor(client, prompt):
    last = None
    for attempt in range(6):
        try:
            t0 = time.time()
            r = client.models.generate_content(
                model=SUPERVISOR_MODEL, contents=prompt)
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
        "records carefully, then answer with the letter of the correct "
        "option.\n\n"
        f"{rec['question']}\n{opts}\n\nMedical records:\n{doc}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    recs = data_loader.load_split(args.split)
    if args.limit:
        recs = recs[:args.limit]

    log_path = latest_log_for(args.split)
    done = done_ids(log_path)
    if log_path and done:
        print(f"resuming {log_path} ({len(done)} already done)")
    else:
        log_path = runlog.new_log(f"cloud_only_{args.split}")
        print(f"logging to {log_path}")

    for i, rec in enumerate(recs):
        if (rec["patient"], rec["q_idx"]) in done:
            print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
                  f"already done, skipping", flush=True)
            continue
        prompt = build_prompt(rec)
        out = ask_supervisor(client, prompt)
        g = grader.grade(out["text"], rec["gold"], rec["options"])
        cost = runlog.supervisor_cost_usd(
            out["prompt_tokens"], out["completion_tokens"],
            out["thinking_tokens"])
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "cloud-only",
            "worker": None,
            "supervisor": SUPERVISOR_MODEL,
            "grading": f"rule-based ({g['method']}) + hand-check queue",
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
              f"(+{out['thinking_tokens']} think) ${cost:.4f}", flush=True)
        # Free tier is 5 req/min; pace calls to stay under it (Oct 18).
        time.sleep(15)

    runlog.summarize(log_path)


if __name__ == "__main__":
    main()
