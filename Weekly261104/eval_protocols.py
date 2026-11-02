"""Eval protocols, frozen thresholds and seeds as fixed constants.

Purpose: eval protocols, frozen thresholds and seeds, for eval after push.
run_pipeline.py evaluate must NOT be used for eval protocols, because
it compares the stored minimum scores, not mean scores. mean_random_warmup.py
must NOT be used for eval, because it recomputes thresholds from the log
it is given, which would be eval data on eval, and it exits on eval.
This script does neither: thresholds and seeds below are constants,
set from warmup only, never recomputed from the input log.

Frozen thresholds, mean of chunk means, strictly below flags a step.
A step where no chunk gave an answer has mean None and is never flagged.
Full precision, from sweep_thresholds.py on the warmup shared logs:
  27b: 25th -0.18209592998027802, mean -0.11813126318156719,
       75th -0.08691378527631362
  4b:  25th -0.7821582900699928, mean -0.5289081293072037,
       75th -0.41724940061128024
Random seeds, one draw per question, question index i in log order, 0-based:
  random_logprob: random.Random(20261103 + i)
  random_agreement: random.Random(20261104 + i)
Pool for random controls is all steps of the question, including skipped
steps, sample size equal to the matching triggered protocol's reviews
for that question, sample without replacement.

Protocols run, per question, in this order:
  plan_once, check_every_step, triggered_agreement,
  triggered_logprob_mean, random_logprob, random_agreement,
  sweep_25th, sweep_75th, then flips (one single-step flip per step).
Sweep points are exploratory, not Holm. Flips are plan once plus one
single step flip per step, rescue/breakage/neutral vs plan once.
No worker calls, no new review calls, no Ollama needed. Fresh cost is
aggregation calls only, where the reviewed set is new. Cached sets cost
fresh 0. Cap flags --max-calls and --max-usd count fresh supervisor
calls and dollars only, and stop the run cleanly at either cap, with
progress kept and exit code 2, no partial row written.

Warmup check, built in: with --split warmup, after the run the script
compares plan, check, triggered_logprob_mean, triggered_agreement,
random_logprob, random_agreement, and flips against the frozen warmup
key table, and exits 1 if any number differs. Warmup check passed
November 2 on both machines. The same script, unchanged, runs eval with
--split eval. On eval, no warmup check is applied, because eval
numbers are not known in advance.

Usage, from the pipeline folder on the target machine:
  bosgame: /home/localai/is-env/bin/python eval_protocols.py --worker 27b --split warmup --shared /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl
  laptop: ~/is-env/bin/python eval_protocols.py --worker 4b --split warmup --shared /home/aayush/baseline_logs/shared_4b_warmup_20261028_015129.jsonl
  eval, after push: same command with --split eval and the eval shared log,
  plus --max-calls and --max-usd from the freeze caps for that run.
Requires GEMINI_API_KEY in the run shell, except with --mock.
Output log: eval_protocols_<worker>_<split>.jsonl in the pipeline log
folder, one row per protocol per question, plus one flip row per step.
"""
import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocols  # noqa: E402
import runlog  # noqa: E402
import run_pipeline  # noqa: E402

# Frozen constants. Do not recompute these from the input log.
FROZEN = {
    "27b": {"mean": -0.11813126318156719,
            "25th": -0.18209592998027802,
            "75th": -0.08691378527631362},
    "4b": {"mean": -0.5289081293072037,
           "25th": -0.7821582900699928,
           "75th": -0.41724940061128024},
}

# Warmup key table, for the built-in warmup check only.
# (correct, reviews) per protocol, and flips (rescue, breakage, neutral).
EXPECTED_WARMUP = {
    "27b": {
        "plan_once": (20, 0), "check_every_step": (19, 75),
        "triggered_logprob_mean": (20, 30), "triggered_agreement": (20, 30),
        "random_logprob": (20, 30), "random_agreement": (20, 30),
        "flips": (0, 1, 74),
    },
    "4b": {
        "plan_once": (15, 0), "check_every_step": (13, 69),
        "triggered_logprob_mean": (15, 31), "triggered_agreement": (13, 49),
        "random_logprob": (15, 31), "random_agreement": (13, 49),
        "flips": (0, 4, 65),
    },
}

PROTOCOLS = ("plan_once", "check_every_step", "triggered_agreement",
             "triggered_logprob_mean", "random_logprob", "random_agreement",
             "sweep_25th", "sweep_75th")


def mean_score(jobs):
    vals = [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["4b", "27b"], required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--split", choices=["warmup", "eval"], required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--supervisor-model", default=run_pipeline.SUPERVISOR_MODEL)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY") and not args.mock:
        sys.exit("GEMINI_API_KEY is not set in this shell.")
    rows = runlog.read_rows(args.shared)
    if not rows:
        sys.exit(f"no rows in {args.shared}")
    splits = sorted({str(r.get("split")) for r in rows})
    if splits != [args.split]:
        sys.exit(f"refusing: shared log split(s) {splits} != --split {args.split}")
    thr = FROZEN[args.worker]
    print(f"worker {args.worker}: frozen thresholds mean {thr['mean']!r}, "
          f"25th {thr['25th']!r}, 75th {thr['75th']!r}, "
          "not recomputed from this log")
    _, supervisor, _ = run_pipeline.build(args, need_worker=False)
    path = runlog.new_log(f"eval_protocols_{args.worker}_{args.split}")
    print(f"logging to {path}")
    fresh_start = supervisor.fresh
    stats = {n: {"correct": 0, "n": 0, "reviews": 0, "rescued": 0,
                 "broken": 0, "agg_usd": 0.0} for n in PROTOCOLS}
    flips = {"rescue": 0, "breakage": 0, "neutral": 0, "n": 0}
    cap_hit = False
    for qi, r in enumerate(rows):
        if run_pipeline._cap_hit(supervisor, args):
            cap_hit = True
            break
        rec = {"question": r["question"], "options": r["options"]}
        means = [mean_score(s["out"]) for s in r["steps"]]
        mean_reviewed = tuple(i for i, m in enumerate(means)
                              if m is not None and m < thr["mean"])
        sweep25 = tuple(i for i, m in enumerate(means)
                        if m is not None and m < thr["25th"])
        sweep75 = tuple(i for i, m in enumerate(means)
                        if m is not None and m < thr["75th"])
        agreement_reviewed = protocols.reviewed_steps(r, "triggered_agreement")
        k = len(r["steps"])
        rng_log = random.Random(20261103 + qi)
        rng_agr = random.Random(20261104 + qi)
        rand_log = tuple(sorted(rng_log.sample(range(k), len(mean_reviewed)))) if k else ()
        rand_agr = tuple(sorted(rng_agr.sample(range(k), len(agreement_reviewed)))) if k else ()
        runs = [
            ("plan_once", None, None),
            ("check_every_step", None, None),
            ("triggered_agreement", None, None),
            ("triggered_logprob_mean", "flip", mean_reviewed),
            ("random_logprob", "flip", rand_log),
            ("random_agreement", "flip", rand_agr),
            ("sweep_25th", "flip", sweep25),
            ("sweep_75th", "flip", sweep75),
        ]
        plan_correct = None
        for name, proto, reviewed in runs:
            if proto is None:
                row = protocols.run_protocol(rec, r, supervisor, name,
                                             component="eval_protocols_aggregate")
            else:
                row = protocols.run_protocol(rec, r, supervisor, proto,
                                             reviewed=reviewed,
                                             component="eval_protocols_aggregate")
            row.update({"worker": r["worker"], "supervisor": r["supervisor"],
                        "grading": run_pipeline.GRADING,
                        "display_protocol": name,
                        "mean_threshold": thr["mean"],
                        "sweep_25th_threshold": thr["25th"],
                        "sweep_75th_threshold": thr["75th"]})
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
            if run_pipeline._cap_hit(supervisor, args):
                cap_hit = True
                break
        if cap_hit:
            break
        # Flips, one single-step flip per step, vs plan once above.
        for i, s in enumerate(r["steps"]):
            flip = protocols.run_protocol(rec, r, supervisor, "flip",
                                          reviewed=(i,),
                                          component="eval_flip_aggregate")
            if (not plan_correct) and flip["correct"]:
                label = "rescue"
                flips["rescue"] += 1
            elif plan_correct and (not flip["correct"]):
                label = "breakage"
                flips["breakage"] += 1
            else:
                label = "neutral"
                flips["neutral"] += 1
            flips["n"] += 1
            flip.update({"worker": r["worker"], "supervisor": r["supervisor"],
                         "grading": run_pipeline.GRADING,
                         "display_protocol": "flip", "flip_step": i,
                         "label": label, "plan_correct": plan_correct})
            runlog.log_row(path, flip)
            if run_pipeline._cap_hit(supervisor, args):
                cap_hit = True
                break
        print(f"{r['patient']} q{r['q_idx']}: plan={plan_correct} "
              f"mean_reviewed={list(mean_reviewed)} "
              f"fresh={supervisor.fresh}", flush=True)
        if cap_hit:
            break
    for name, st in stats.items():
        print(f"{name}: {st['correct']}/{st['n']}, reviews_total {st['reviews']}, "
              f"rescued {st['rescued']}, broken {st['broken']}, "
              f"agg_usd ${st['agg_usd']:.4f}")
    print(f"flips {flips['n']}: rescue {flips['rescue']}, "
          f"breakage {flips['breakage']}, neutral {flips['neutral']}")
    print(f"fresh aggregation calls this run: {supervisor.fresh - fresh_start} "
          f"(cached hits {supervisor.hits}), "
          f"fresh $ {getattr(supervisor, 'fresh_usd', 0.0):.4f}")
    print(f"rows -> {path}")
    if cap_hit:
        print(f"\n{run_pipeline._cap_msg(supervisor, args)} Progress is kept ({path}).")
        sys.exit(2)
    if args.split == "warmup":
        exp = EXPECTED_WARMUP[args.worker]
        failed = []
        for name in ("plan_once", "check_every_step", "triggered_logprob_mean",
                     "triggered_agreement", "random_logprob", "random_agreement"):
            got = (stats[name]["correct"], stats[name]["reviews"])
            if got != exp[name]:
                failed.append(f"{name}: got {got}, expected {exp[name]}")
        got_flips = (flips["rescue"], flips["breakage"], flips["neutral"])
        if got_flips != exp["flips"]:
            failed.append(f"flips: got {got_flips}, expected {exp['flips']}")
        if failed:
            print("WARMUP CHECK FAILED:")
            for f in failed:
                print(f"  {f}")
            sys.exit(1)
        print("WARMUP CHECK PASSED: key table and flips reproduced exactly.")


if __name__ == "__main__":
    main()
