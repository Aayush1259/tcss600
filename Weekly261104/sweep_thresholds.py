"""Compute log-prob sweep thresholds from a warmup shared log. No API calls.

Purpose: sweep thresholds for the freeze. Two more log-prob thresholds,
set now from warmup, not from eval. With plan once (0 steps) and check
every step (all steps) as end points, the mean threshold plus these two
gives a 5-point accuracy versus cost curve. Mean threshold stays the
only primary setting. Sweep points are exploratory, not Holm.

Thresholds, per worker, from mean-of-chunk-means scores of scored steps
only, same quantile method as signals.quantile, linear interpolation:
  25th percentile: flag steps below it, about 25 percent of scored steps
  50th percentile: median, primary mean threshold, about 50 percent
  75th percentile: flag steps below it, about 75 percent of scored steps
Skipped steps, mean None, are never flagged, same as mean.

Usage, from the pipeline folder on the target machine, no key needed:
  bosgame: /home/localai/is-env/bin/python sweep_thresholds.py --worker 27b --shared /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl
  laptop: ~/is-env/bin/python sweep_thresholds.py --worker 4b --shared /home/aayush/baseline_logs/shared_4b_warmup_20261028_015129.jsonl
Output: full precision thresholds, rounded values, scored count, flagged
counts at each threshold, and per question flagged counts, recorded
in the freeze at full precision.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runlog  # noqa: E402
import signals  # noqa: E402


def mean_score(jobs):
    vals = [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["4b", "27b"], required=True)
    ap.add_argument("--shared", required=True)
    args = ap.parse_args()
    rows = runlog.read_rows(args.shared)
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
    print(f"worker {args.worker}: scored steps {len(mean_scores)}, questions {len(rows)}")
    thresholds = {}
    for label, q in (("25th", 0.25), ("50th_median_primary", 0.5), ("75th", 0.75)):
        t = signals.quantile(mean_scores, q)
        flagged = sum(1 for m in mean_scores if m < t)
        thresholds[label] = t
        print(f"{label} threshold full precision {t!r}, rounded {t:.4f}, "
              f"flagged {flagged} of {len(mean_scores)}")
    for label, t in thresholds.items():
        counts = [sum(1 for m in q if m is not None and m < t) for q in per_question]
        print(f"{label} per question flagged counts: {counts}, total {sum(counts)}")
    print("Three full precision thresholds, one per worker. Median matches "
          "the mean threshold in the freeze.")


if __name__ == "__main__":
    main()
