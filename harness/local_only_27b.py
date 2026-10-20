"""27B local-only baseline on the frozen question sets. Machine: bosgame.

Worker alone, no supervisor (RQ2's 'no review' reference point for the
strong worker). Full patient record per question at num_ctx=16384, the
window verified Oct 13 to hold the largest record (~12,766 tokens).

Labels on every row: protocol local-only | worker qwen3.8:27b, ctx 16384,
think off | supervisor none | grading rule-based + hand-check queue.

Usage:
    python3 local_only_27b.py --split warmup            # all 20 warmup
    python3 local_only_27b.py --split warmup --limit 1  # smoke test
    python3 local_only_27b.py --split warmup --think on # think-on reference

Requires the Ollama server running on bosgame (it runs in tmux; restart
with ~/ollama/bin/ollama serve if the connection fails). Run from ~.
"""
import argparse
import os
import sys
import time

# Fix for the Oct 13 import-shadowing lesson: ~/ollama/ shadows the
# installed `ollama` package unless site-packages comes first.
_sp = "/home/localai/is-env/lib/python3.14/site-packages"
if os.path.isdir(_sp) and _sp not in sys.path:
    sys.path.insert(0, _sp)

import ollama  # noqa: E402

import data_loader  # noqa: E402
import grader  # noqa: E402
import runlog  # noqa: E402

WORKER_MODEL = "qwen3.8:27b"
NUM_CTX = 16384  # verified Oct 13, 2026


def ask_worker(prompt, think=False):
    t0 = time.time()
    # NOTE (Oct 17): `think` is a TOP-LEVEL chat parameter. Inside
    # `options` it is silently ignored and the model thinks by default,
    # which once produced 500 thinking tokens and an empty answer.
    # NOTE (Oct 18): num_predict=1200, not 500. With think off the model
    # narrates evidence in the open and needs room to reach its choice;
    # with think on it needs room for thinking AND the answer. At 500,
    # both conditions were artificially capped (empty answers / cut-off
    # narration). Both think conditions use the same budget.
    resp = ollama.chat(
        model=WORKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=think,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 1200},
    )
    dt = time.time() - t0
    return {
        "text": resp.message.content,
        "prompt_tokens": getattr(resp, "prompt_eval_count", None),
        "completion_tokens": getattr(resp, "eval_count", None),
        "wall_time_s": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--think", default="off", choices=["on", "off"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    think = args.think == "on"

    recs = data_loader.load_split(args.split)
    if args.limit:
        recs = recs[:args.limit]

    log_path = runlog.new_log(f"local_only_27b_{args.split}_think-{args.think}")
    print(f"logging to {log_path}")

    for i, rec in enumerate(recs):
        prompt = data_loader.build_mcq_prompt(rec)
        out = ask_worker(prompt, think=think)
        g = grader.grade(out["text"], rec["gold"], rec["options"])
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "local-only",
            "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think {'on' if think else 'off'}",
            "supervisor": None,
            "grading": f"rule-based ({g['method']}) + hand-check queue",
            "predicted": g["letter"],
            "gold": rec["gold"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "wall_time_s": out["wall_time_s"],
            "worker_prompt_tokens": out["prompt_tokens"],
            "worker_completion_tokens": out["completion_tokens"],
            "est_cost_usd": 0.0,
            "prompt": prompt,
            "answer": out["text"],
        })
        print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
              f"pred={g['letter']} gold={rec['gold']} match={g['match']} "
              f"{out['wall_time_s']}s", flush=True)

    runlog.summarize(log_path)


if __name__ == "__main__":
    main()
