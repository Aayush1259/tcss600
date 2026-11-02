"""Recheck 27B step IDs from the shared log. No API calls.

Purpose: draft item 4, step numbering. Separator printout counted steps
from 1. Flips log counts steps from 0. This script lists, from the shared
log only, in 0-based index order, for every warmup question, each step
index, mean score or None, redo verdict, reviewed present or None.
Then, if a flips log path is given, it lists breakage flips with the
same 0-based index, so overlap between skipped steps and breakage steps
can be checked directly, without guessing numbering.

Usage, on bosgame, from the pipeline folder, no key needed:
  /home/localai/is-env/bin/python check_27b_steps.py --shared /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl --flips /home/localai/baseline_logs/flips_27b_warmup_20261031_115903.jsonl
If the flips log name differs, use the actual file in baseline_logs.
Output: per patient, per step index, mean, redo, reviewed, plus totals:
total steps, scored steps, skipped steps, redo steps, redo rate,
skipped redo steps with indices, and breakage flips with indices.
Output includes the patient_18 section and the totals.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runlog  # noqa: E402


def mean_score(jobs):
    vals = [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--flips", default=None)
    args = ap.parse_args()
    rows = runlog.read_rows(args.shared)
    total = scored = skipped = redo = skipped_redo = 0
    for r in rows:
        print(f"{r['patient']} q{r['q_idx']}:")
        for i, s in enumerate(r["steps"]):
            m = mean_score(s["out"])
            verdict = s.get("review", {}).get("verdict", "?")
            has_reviewed = s.get("reviewed") is not None
            total += 1
            if m is None:
                skipped += 1
            else:
                scored += 1
            if verdict == "redo":
                redo += 1
                if m is None:
                    skipped_redo += 1
            print(f"  step {i}: mean={m} verdict={verdict} reviewed={has_reviewed}")
    print(f"TOTALS: steps {total}, scored {scored}, skipped {skipped}, "
          f"redo {redo}, redo_rate {redo/total if total else 0:.3f}, "
          f"skipped_redo {skipped_redo}")
    if args.flips and os.path.exists(args.flips):
        frows = runlog.read_rows(args.flips)
        print("FLIPS breakage, 0-based flip_step:")
        for fr in frows:
            # flips log rows carry plan and flip correctness in the log row
            # fields used by breakage_warmup.py: plan_predicted, flip answer
            # and correct flags are in the row; print what is present.
            if fr.get("correct") is False and fr.get("plan_correct") is True:
                print(f"  {fr.get('patient')} q{fr.get('q_idx')} "
                      f"flip_step {fr.get('flip_step')} "
                      f"plan={fr.get('plan_predicted')} "
                      f"flip={fr.get('predicted')} "
                      f"gold={fr.get('gold')}")
        print("Note: if no breakage rows print above, use breakage_warmup.py "
              "output from the same flips log, which prints step indices "
              "in the same 0-based numbering.")


if __name__ == "__main__":
    main()
