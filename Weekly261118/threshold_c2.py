"""The 4B log-prob threshold from a shared-parts warmup log (November 13). Read-only; no model call.

The rule is the frozen one (pre-registration Sections 3 and 10, and mean_random_warmup.py of
November 2): a step's score is the mean of mean_logprob over the chunks of its first pass ("out")
that gave an answer; a step with no such chunk has no score; the threshold is the median of the
scored steps (signals.quantile, linear interpolation, q = 0.5); a step is flagged only if its score
is strictly below the threshold.

For a guarded run (run_shared_c2.py) only rows with status "ok" are used; halt and failed rows
have no steps. Each question must appear once. Older logs have no status field and are read as is.

Validation before use on warmup4 (both must pass):
    python threshold_c2.py --shared <Nov 2 log> --expect -0.5289081293072037 --expect-scored 63 --expect-flagged 31
    python threshold_c2.py --shared <Nov 4 log at 3,072> --expect -0.8748829690247267 --expect-scored 35
Prints counts and the threshold only; no accuracy.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import runlog  # noqa: E402
import signals  # noqa: E402


def mean_score(jobs):
    vals = [j["mean_logprob"] for j in jobs if j.get("answer") and j.get("mean_logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def threshold_from_rows(rows):
    used = [r for r in rows if r.get("status", "ok") == "ok"]
    keys = collections.Counter((r["patient"], r["q_idx"]) for r in used)
    twice = [k for k, n in keys.items() if n > 1]
    if twice:
        raise SystemExit(f"refused: questions answered more than once {twice[:3]}")
    steps = [s for r in used for s in r["steps"]]
    scores = [m for m in (mean_score(s["out"]) for s in steps) if m is not None]
    if not scores:
        raise SystemExit("refused: no scored step")
    thr = signals.quantile(scores, 0.5)
    return {"questions": len(used), "skipped_rows": len(rows) - len(used), "steps": len(steps),
            "scored": len(scores), "threshold": thr, "flagged": sum(1 for m in scores if m < thr),
            "ties_at_threshold": sum(1 for m in scores if m == thr)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True, help="one run's shared log (jsonl)")
    ap.add_argument("--expect", type=float, help="expected threshold, full precision")
    ap.add_argument("--expect-scored", type=int)
    ap.add_argument("--expect-flagged", type=int)
    ap.add_argument("--expect-questions", type=int)
    args = ap.parse_args(argv)
    res = threshold_from_rows(runlog.read_rows(args.shared))
    print(f"{args.shared}: {res['questions']} questions ({res['skipped_rows']} halt or failed rows not used), "
          f"{res['steps']} steps, {res['scored']} scored")
    print(f"threshold (median of step mean scores) {res['threshold']!r}, rounded {res['threshold']:.4f}; "
          f"flagged {res['flagged']} of {res['scored']} (strictly below); scores equal to it {res['ties_at_threshold']}")
    bad = []
    if args.expect is not None and res["threshold"] != args.expect:
        bad.append(f"threshold {res['threshold']!r} is not {args.expect!r}")
    for k in ("scored", "flagged", "questions"):
        want = getattr(args, f"expect_{k}")
        if want is not None and res[k] != want:
            bad.append(f"{k} {res[k]} is not {want}")
    if bad:
        print("CHECK FAILED: " + "; ".join(bad))
        return 1
    if any(v is not None for v in (args.expect, args.expect_scored, args.expect_flagged, args.expect_questions)):
        print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
