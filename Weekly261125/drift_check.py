"""The drift check (my November 13 amendment; Amendment 3, section 5): cloud alone once more on the
first 20 of the 91 record-dependent questions in benchmark order (patient, then question index), with
the same prompt and settings as November 12, by the pushed cloud_alone.py, unchanged. Descriptive only.

The run writes to its own log folder (~/drift_logs), so it can never resume or touch the November 12
log, and it refuses to start if that folder already holds a drift log: the check is not rerun.

    python3 drift_check.py dry-run --set record_dependent_set.json --ref-log <Nov 12 cloud alone eval log>
        builds the 20 prompts and compares them, and their token counts, with the November 12 log; no call
    python3 drift_check.py run --set record_dependent_set.json --max-usd 0.60 --max-calls 30
    python3 drift_check.py compare --set record_dependent_set.json --ref-log <Nov 12 log> --log <drift log>
        prints the letters that match November 12 out of 20, and the accuracy of both
"""
import argparse
import glob
import json
import os
import sys

DRIFT_DIR = os.path.expanduser("~/drift_logs")
N = 20


def first_20(set_path):
    keys = sorted((q["patient"], q["q_idx"]) for q in json.load(open(set_path, encoding="utf-8"))["set"])
    assert len(keys) == 91, len(keys)
    return keys[:N]


def records(set_path):
    import cloud_alone
    import data_loader
    want = first_20(set_path)
    recs = {(r["patient"], r["q_idx"]): r for r in data_loader.load_split("eval", pad_n=cloud_alone.PAD_N)}
    return [recs[k] for k in want], cloud_alone


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answered(path):
    return {(r["patient"], r["q_idx"]): r for r in rows(path) if not r.get("failed")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dry-run", "run", "compare"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--ref-log")
    ap.add_argument("--log")
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--max-calls", type=int)
    args = ap.parse_args(argv)

    if args.mode == "compare":
        want = first_20(args.set)
        ref, new = answered(args.ref_log), answered(args.log)
        missing = [k for k in want if k not in new]
        same = sum(1 for k in want if k in new and new[k]["predicted"] == ref[k]["predicted"])
        print(f"drift check: {len(want) - len(missing)} of {N} answered ({len(missing)} missing or failed)")
        print(f"letters equal to November 12: {same} of {N}")
        print(f"accuracy: November 12 {sum(bool(ref[k]['match']) for k in want)} of {N}; "
              f"drift run {sum(bool(new[k]['match']) for k in want if k in new)} of {N}")
        versions = sorted({str(new[k].get('supervisor_model_version')) for k in want if k in new})
        cost = sum(new[k].get("est_cost_usd") or 0 for k in want if k in new)
        print(f"model version {versions}; estimated cost {cost:.4f} dollars")
        return 0

    os.environ["BASELINE_LOG_DIR"] = DRIFT_DIR           # before cloud_alone imports runlog
    recs, ca = records(args.set)
    if args.mode == "dry-run":
        if not args.ref_log:
            sys.exit("dry-run needs --ref-log (the November 12 cloud alone eval log)")
        return ca.dry_run(recs, args.ref_log)
    if args.max_usd is None or args.max_calls is None:
        sys.exit("run needs --max-usd and --max-calls")
    earlier = glob.glob(os.path.join(DRIFT_DIR, ca.log_prefix("eval") + "_*.jsonl"))
    if earlier:
        sys.exit(f"refused: a drift log already exists ({earlier[0]}); the drift check is not rerun")
    run_args = argparse.Namespace(split="eval", max_calls=args.max_calls, max_usd=args.max_usd,
                                  expect_version=ca.cloud_only.SUPERVISOR_MODEL, fresh=True)
    return ca.run(recs, run_args)


if __name__ == "__main__":
    sys.exit(main())
