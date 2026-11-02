"""27B local-only baseline on the frozen question sets. Machine: bosgame.

Worker alone, no supervisor (RQ2's 'no review' reference point for the
strong worker). Full patient record per question at num_ctx=16384, the
window verified Oct 13 to hold the largest record (~12,766 tokens).

Week 4 freeze (2026-10-24, see frozen-rules-2026-10-24.md): final
answers are JSON only (grade_json); padding is a
parameter (own record first, then the next N patients in benchmark
order); context rises with padding (32768 at N=2, 65536 at N=3).
JSON runs use think off. This runner always starts a fresh log.
Oct 24 follow-up: elicitation is schema-constrained JSON (ANSWER_SCHEMA),
not plain format="json", after the N=2 warmup showed plain "json" lets
valid but wrong-schema outputs through at long context.

Labels on every row: protocol local-only | worker qwen3.8:27b, ctx,
think, json | supervisor none | grading json + hand-check queue.

Usage:
    python3 local_only_27b.py --split warmup            # all 20 warmup
    python3 local_only_27b.py --split warmup --limit 1  # smoke test
    python3 local_only_27b.py --split warmup --think on # think-on reference
    python3 local_only_27b.py --split warmup --padding 2 --num-ctx 32768
        # Week 4 gap test: padded warmup, JSON answers

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

# Oct 24: plain format="json" only guarantees VALID JSON, not the answer
# schema. At N=2 (Oct 24 warmup) the model returned valid but wrong-schema
# JSON on 16/20 questions (e.g. {"region": ...}), which grade_json
# correctly counted as malformed. The N=0 diagnostic the same day
# (17/20, 0 malformed) proved the failure is length-driven instruction
# loss, not JSON itself. So the runner now constrains decoding with a
# JSON Schema: the only permitted output shape is {"answer": "X"} with
# X one letter A-E. The output is still exactly {"answer": "X"}.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def ask_worker(prompt, think=False, num_ctx=NUM_CTX):
    t0 = time.time()
    # NOTE (Oct 17): `think` is a TOP-LEVEL chat parameter. Inside
    # `options` it is silently ignored and the model thinks by default,
    # which once produced 500 thinking tokens and an empty answer.
    # NOTE (Oct 18): num_predict=1200, not 500. With think off the model
    # narrates evidence in the open and needs room to reach its choice;
    # with think on it needs room for thinking AND the answer. At 500,
    # both conditions were artificially capped (empty answers / cut-off
    # narration). Both think conditions use the same budget.
    # Week 4 freeze: final answers are JSON only (grade_json).
    # Oct 24: format is the ANSWER_SCHEMA, not the string "json":
    # "json" alone only guarantees valid JSON, which let wrong-schema
    # outputs (e.g. {"region": ...}) through at N=2. The schema
    # constrains decoding to {"answer": "X"} with X in A-E.
    resp = ollama.chat(
        model=WORKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=think,
        format=ANSWER_SCHEMA,
        options={"num_ctx": num_ctx, "temperature": 0, "num_predict": 1200},
    )
    dt = time.time() - t0
    # NOTE (Oct 20): GPU seconds for the proposal's local-compute metric.
    # Ollama reports prompt_eval_duration and eval_duration in
    # nanoseconds; their sum is the model's compute time for the call.
    gpu_s = ((getattr(resp, "prompt_eval_duration", 0) or 0) +
             (getattr(resp, "eval_duration", 0) or 0)) / 1e9
    return {
        "text": resp.message.content,
        "prompt_tokens": getattr(resp, "prompt_eval_count", None),
        "completion_tokens": getattr(resp, "eval_count", None),
        "wall_time_s": round(dt, 1),
        "gpu_s": round(gpu_s, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--think", default="off", choices=["on", "off"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--padding", type=int, default=0,
                    help="distractor patients per question (Week 4 freeze)")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX,
                    help="Ollama context window (32768 at N=2, 65536 at N=3)")
    args = ap.parse_args()
    think = args.think == "on"

    recs = data_loader.load_split(args.split, pad_n=args.padding)
    if args.limit:
        recs = recs[:args.limit]

    log_path = runlog.new_log(
        f"local_only_27b_{args.split}_think-{args.think}"
        f"_pad{args.padding}_json")
    print(f"logging to {log_path}")

    for i, rec in enumerate(recs):
        prompt = data_loader.build_mcq_prompt(rec, json_output=True)
        out = ask_worker(prompt, think=think, num_ctx=args.num_ctx)
        g = grader.grade_json(out["text"], rec["gold"], rec["options"])
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "local-only",
            "worker": (f"{WORKER_MODEL}, ctx {args.num_ctx}, "
                       f"think {'on' if think else 'off'}, "
                       f"json-constrained"),
            "supervisor": None,
            "grading": f"json ({g['method']}) + hand-check queue",
            "output_format": "json",
            "padding_n": rec["padding_n"],
            "padding_patients": rec["padding_patients"],
            "worker_num_ctx": args.num_ctx,
            "predicted": g["letter"],
            "gold": rec["gold"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "wall_time_s": out["wall_time_s"],
            "worker_gpu_s": out["gpu_s"],
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
