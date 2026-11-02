"""Warmup rerun, mean log-prob triggered plus two random controls.

Purpose: warmup rerun, mean log-prob triggered plus two random controls.
This script
recomputes mean-of-chunk-means from the existing shared log out jobs, checks
the flagged count, then runs plan_once, check_every_step, triggered_agreement
(unchanged, for verification), triggered_logprob_mean, random_logprob, and
random_agreement, one aggregation per question per protocol. No worker calls,
no new review calls, no Ollama needed. Fresh cost is aggregation calls only,
where the reviewed set is new. Cached sets, plan, check, agreement, and any
random set that happens to equal a previous set, cost fresh 0.

Mean score, per step:
  vals = [job mean_logprob for job in step out jobs
          if job answer is present and job mean_logprob is not None]
  mean = sum(vals) / len(vals) if vals else None
  flagged if mean is not None and mean < threshold_mean
Skipped steps, mean None, are never flagged by mean, same as min.

Threshold, computed here from the shared log itself, median quantile,
linear interpolation, same rule as signals.calibrate_threshold, so the
threshold is not taken from a rounded printed value. The script prints
the full precision threshold and checks it rounds to the separator values:
27B -0.1181, 4B -0.5289. If the rounded value differs, the script exits
with an error.

Flagged count check, before any aggregation:
  27B must flag exactly 30 of 60 scored steps
  4B must flag exactly 31 of 63 scored steps
If the count differs, exit with an error and do not run protocols.
That check is the code correctness check for the warmup rerun.

Random controls, exploratory, not primary, not in Holm:
  Pool is all steps of a question, 0 to K-1, including steps log-prob
  cannot score, because every step has a review call in the shared log.
  One seed is one random draw. For question number i in log order,
  0-based, as in the shared log:
    rng_logprob = random.Random(20261103 + i)
    rng_agreement = random.Random(20261104 + i)
  Sample without replacement, size equal to triggered n_reviews for that
  question, for the corresponding triggered protocol, logprob mean count
  for random_logprob, agreement count for random_agreement.
  If size is 0, reviewed set is empty, aggregation is plan_once, cached.
  If size is K, reviewed set is all steps, aggregation is check, cached.

Usage, from the pipeline folder on the target machine:
  bosgame: /home/localai/is-env/bin/python mean_random_warmup.py --worker 27b --shared /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl
  laptop: ~/is-env/bin/python mean_random_warmup.py --worker 4b --shared /home/aayush/baseline_logs/shared_4b_warmup_20261028_015129.jsonl
Requires GEMINI_API_KEY in the run shell.
Output log: mean_random_<worker>_warmup.jsonl in the pipeline log folder,
one row per protocol per question. Summary printed at the end, including
plan, check, agreement, mean_logprob, random_logprob, random_agreement
scores, reviews, rescued and broken vs plan, fresh calls, agg_usd.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocols  # noqa: E402
import runlog  # noqa: E402
import run_pipeline  # noqa: E402
import signals  # noqa: E402

EXPECTED = {
    "27b": {"flagged": 30, "scored": 60, "rounded": -0.1181},
    "4b": {"flagged": 31, "scored": 63, "rounded": -0.5289},
}


def mean_score(jobs):
    vals = [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]
    return sum(vals) / len(vals) if vals else None


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
    # Compute mean scores and threshold, before any aggregation.
    mean_scores = []
    per_question = []
    for r in rows:
        q_means = []
        for s in r["steps"]:
            m = mean_score(s["out"])
            q_means.append(m)
            if m is not None:
                mean_scores.append(m)
        per_question.append(q_means)
    threshold = signals.quantile(mean_scores, 0.5)
    flagged_total = sum(1 for m in mean_scores if m < threshold)
    exp = EXPECTED[args.worker]
    print(f"worker {args.worker}: mean scored steps {len(mean_scores)}, "
          f"expected {exp['scored']}")
    print(f"mean threshold full precision {threshold!r}, "
          f"rounded {threshold:.4f}, expected rounded {exp['rounded']:.4f}")
    print(f"mean flagged {flagged_total}, expected {exp['flagged']}")
    if len(mean_scores) != exp["scored"] or flagged_total != exp["flagged"]:
        sys.exit(
            f"CHECK FAILED: scored {len(mean_scores)} flagged {flagged_total}, "
            f"expected scored {exp['scored']} flagged {exp['flagged']}. "
            "Do not run protocols. Bring this exact output back.")
    if round(threshold, 4) != exp["rounded"]:
        sys.exit(
            f"CHECK FAILED: threshold rounded {threshold:.4f}, "
            f"expected {exp['rounded']:.4f}. Do not run protocols.")
    print("Mean check passed. Running protocols.", flush=True)

    _, supervisor, _ = run_pipeline.build(args, need_worker=False)
    path = runlog.new_log(f"mean_random_{args.worker}_{args.split}")
    print(f"logging to {path}")
    fresh_start = supervisor.fresh
    stats = {}
    for name in ("plan_once", "check_every_step", "triggered_agreement",
                 "triggered_logprob_mean", "random_logprob", "random_agreement"):
        stats[name] = {"correct": 0, "n": 0, "reviews": 0, "rescued": 0,
                       "broken": 0, "agg_usd": 0.0}
    for qi, r in enumerate(rows):
        rec = {"question": r["question"], "options": r["options"]}
        mean_reviewed = tuple(i for i, m in enumerate(per_question[qi])
                              if m is not None and m < threshold)
        agreement_reviewed = protocols.reviewed_steps(r, "triggered_agreement")
        rng_log = random.Random(20261103 + qi)
        rng_agr = random.Random(20261104 + qi)
        k = len(r["steps"])
        rand_log = tuple(sorted(rng_log.sample(range(k), len(mean_reviewed)))) if k else ()
        rand_agr = tuple(sorted(rng_agr.sample(range(k), len(agreement_reviewed)))) if k else ()
        runs = [
            ("plan_once", None),
            ("check_every_step", None),
            ("triggered_agreement", None),
            ("triggered_logprob_mean", mean_reviewed),
            ("random_logprob", rand_log),
            ("random_agreement", rand_agr),
        ]
        plan_correct = None
        for name, reviewed in runs:
            if reviewed is None:
                proto = name
                rev_arg = None
            else:
                proto = "flip"
                rev_arg = reviewed
            row = protocols.run_protocol(rec, r, supervisor, proto,
                                         reviewed=rev_arg,
                                         component="mean_random_aggregate")
            row.update({"worker": r["worker"], "supervisor": r["supervisor"],
                        "grading": run_pipeline.GRADING,
                        "display_protocol": name,
                        "mean_threshold": threshold})
            if name == "plan_once":
                plan_correct = row["correct"]
            runlog.log_row(path, row)
            st = stats[name]
            st["n"] += 1
            st["correct"] += row["correct"]
            st["reviews"] += row["n_reviews"]
            st["agg_usd"] += row["agg_usd"]
            if name != "plan_once":
                if (not plan_correct) and row["correct"]:
                    st["rescued"] += 1
                elif plan_correct and (not row["correct"]):
                    st["broken"] += 1
        print(f"{r['patient']} q{r['q_idx']}: plan={plan_correct} "
              f"mean_reviewed={list(mean_reviewed)} "
              f"rand_log={list(rand_log)} rand_agr={list(rand_agr)} "
              f"fresh={supervisor.fresh}", flush=True)
    for name, st in stats.items():
        print(f"{name}: {st['correct']}/{st['n']}, reviews_total {st['reviews']}, "
              f"rescued {st['rescued']}, broken {st['broken']}, "
              f"agg_usd ${st['agg_usd']:.4f}")
    print(f"fresh aggregation calls this run: {supervisor.fresh - fresh_start} "
          f"(cached hits {supervisor.hits})")
    print(f"mean threshold full precision: {threshold!r}")
    print(f"rows -> {path}")


if __name__ == "__main__":
    main()
