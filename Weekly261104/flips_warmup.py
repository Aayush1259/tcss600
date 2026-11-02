"""Wrong-to-right labels on warmup, plan_once plus single-step flips.

For each shared warmup question, run plan_once (0 reviews) once, then
for every step, run one aggregation with only that step reviewed
(reviewed jobs if the step was redone, original jobs otherwise).
Classification, per flip, against plan_once:
  rescue:   plan_once wrong, flip correct
  breakage: plan_once correct, flip wrong
  neutral:  otherwise (same correctness, or flip aggregation identical
            to plan_once because the step was accepted, reviewed None,
            in which case the aggregation call is cached, fresh cost 0)

Fresh aggregation calls are only plan_once (20 per worker) plus flips
for steps that were actually redone, because accepted steps choose
identical answers to plan_once and share the cached call. Expected
fresh calls, warmup: 27B plan 20 + redo 6 = 26, 4B plan 20 + redo 35 = 55,
total 81 across both workers. At about $0.0012 per aggregation call
(prompt about 800-1500 tokens, completion plus thinking about 90
tokens, list price $0.75/$3.75 per 1M), expected fresh cost about
$0.10 total, cap $1.00 below is generous. Time is dominated by the
pipeline supervisor pace of 15 seconds between fresh calls: about
26*15 = 390 seconds for 27B, 55*15 = 825 seconds for 4B, if run
sequentially per worker, in parallel across machines about 14 minutes
for the slower worker, plus overhead.

Usage, from the pipeline folder on the target machine:
  /home/localai/is-env/bin/python flips_warmup.py --worker 27b --shared /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl
  ~/is-env/bin/python flips_warmup.py --worker 4b --shared /home/aayush/baseline_logs/shared_4b_warmup_20261028_015129.jsonl
Requires GEMINI_API_KEY in the run shell. Bosgame key is exported by
~/.bashrc, verify length 53 prefix AQ.A before running. Laptop key
must be set in the run shell only via read -s, valid Lab Machine API
key, not the old known-bad key. No worker calls, no Ollama needed.
Output log: flips_<worker>_warmup.jsonl in the pipeline log folder,
one row per plan_once plus one row per flip, with rescue/breakage
labels. Summary printed at the end. This script does NOT run
evaluate protocols or oracle pairs; it is only plan_once plus flips.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocols  # noqa: E402
import runlog  # noqa: E402
import run_pipeline  # noqa: E402


class Args:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["4b", "27b"], required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--split", default="warmup")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--supervisor-model", default=run_pipeline.SUPERVISOR_MODEL)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY") and not args.mock:
        sys.exit("GEMINI_API_KEY is not set in this shell.")
    rows = runlog.read_rows(args.shared)
    _, supervisor, _ = run_pipeline.build(args, need_worker=False)
    path = runlog.new_log(f"flips_{args.worker}_{args.split}")
    print(f"logging to {path}")
    fresh_start = supervisor.fresh
    plan_correct = plan_n = rescue = breakage = neutral = 0
    flip_rows = 0
    agg_usd_total = 0.0
    for r in rows:
        rec = {"question": r["question"], "options": r["options"]}
        plan = protocols.run_protocol(rec, r, supervisor, "plan_once",
                                      component="flip_aggregate")
        plan.update({"worker": r["worker"], "supervisor": r["supervisor"],
                     "grading": run_pipeline.GRADING, "flip_step": None,
                     "label": "plan_once"})
        runlog.log_row(path, plan)
        plan_n += 1
        plan_correct += plan["correct"]
        agg_usd_total += plan["agg_usd"]
        for i, s in enumerate(r["steps"]):
            flip = protocols.run_protocol(rec, r, supervisor, "flip",
                                          reviewed=(i,),
                                          component="flip_aggregate")
            if (not plan["correct"]) and flip["correct"]:
                label = "rescue"
                rescue += 1
            elif plan["correct"] and (not flip["correct"]):
                label = "breakage"
                breakage += 1
            else:
                label = "neutral"
                neutral += 1
            flip.update({"worker": r["worker"], "supervisor": r["supervisor"],
                         "grading": run_pipeline.GRADING, "flip_step": i,
                         "label": label,
                         "plan_predicted": plan["predicted"],
                         "plan_correct": plan["correct"]})
            runlog.log_row(path, flip)
            flip_rows += 1
            agg_usd_total += flip["agg_usd"]
        print(f"{r['patient']} q{r['q_idx']}: plan={plan['predicted']} "
              f"correct={plan['correct']} steps={len(r['steps'])} "
              f"fresh={supervisor.fresh}", flush=True)
    print(f"worker {args.worker}: plan_once {plan_correct}/{plan_n} = "
          f"{plan_correct/max(plan_n,1):.1%}")
    print(f"flips {flip_rows}: rescue {rescue}, breakage {breakage}, "
          f"neutral {neutral}")
    print(f"fresh aggregation calls this run: {supervisor.fresh - fresh_start} "
          f"(cached hits {supervisor.hits})")
    print(f"agg_usd summed over rows (protocol accounting, cached rows "
          f"counted at first-call cost): ${agg_usd_total:.4f}")
    print(f"rows -> {path}")


if __name__ == "__main__":
    main()
