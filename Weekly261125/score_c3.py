"""Scorer for the pipeline arms (code change C3, November 14; worker option, November 21). Replaces eval_protocols.py.

November 21 (Amendment 3, section 3): --worker selects the worker whose shared log is scored. 4b (the
default) is unchanged: the November 14 threshold and the same file names. 27b scores the 27B run of
November 6 with the 27B threshold pre-registered on November 2 (-0.11813126318156719, the median of
the same step score on the 27B warmup; eval_protocols.py FROZEN["27b"]["mean"]). The shared rows must
name that worker's model. --limit N scores only the first N questions (the one-question trace), and
--print-prompts prints each aggregation prompt and reply, for the trace.

The frozen eval_protocols.py runs eight protocols and the step flips, most of which are paused
(agreement, random controls, sweep, flips), and imports run_pipeline.py, which builds the
unguarded backends. This scorer runs only the three arms of Question A and Question B:

    plan_once                no step reviewed
    check_every_step         every step reviewed (a redone step uses its redone jobs)
    triggered_logprob_mean   steps whose mean of chunk mean log-probs is strictly below the
                             threshold (pre-registration Sections 2, 3 and 10; amendment 2)

Each arm is the frozen protocols.run_protocol, unchanged: the same chosen steps, the frozen
aggregation prompt, parse_letter, grader.matches and the frozen cost accounting (protocol
accounting cost, local seconds). Only the supervisor backend differs:

  --replay-only   reads the cache and never calls the cloud; a reply missing from the cache
                  stops the run. Used to show that this scorer reproduces the frozen warmup
                  scores of November 2 (plan once 15, check every step 13, triggered log-prob
                  15 at -0.5289, reviews 0, 69, 31) from the November 2 cache, with no call.
                  Those scores stay voided; this is a check of the code only.
  live (default)  the guarded, version-pinned supervisor of guarded_backend.py: the version is
                  checked before a reply is stored, an empty reply is a failure, the caps are
                  checked before every fresh call and seeded from this run's own calls log, and
                  every call writes a row.

Each row also carries worker_findings: how many worker jobs in the chosen steps gave an answer
("who does the work", amendment 2). Only shared rows with status "ok" are scored; a question
already scored in this run's log is skipped (resume). On the evaluation split the screen shows
counts only; the accuracy is read by the analysis script, together with the paired comparisons.

    python score_c3.py --shared <log> --split warmup --replay-only --cache-dir ~/pipeline_cache \
        --threshold -0.5289081293072037 --expect-nov2
    python score_c3.py --shared <eval shared log> --split eval --run-id eval1 --cache-dir <dir> \
        --threshold <amendment 2 value> --max-usd <cap> --max-calls <cap>
"""
import argparse
import collections
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cache as cache_mod  # noqa: E402
import guarded_backend as gb  # noqa: E402
import protocols  # noqa: E402
import runlog  # noqa: E402
from threshold_c2 import mean_score  # noqa: E402

SUPERVISOR_MODEL = "gemini-3.8-flash"
# The 4B threshold of amendment 2, computed by threshold_c2.py from the guarded warmup (warmup4) and
# fixed here before the push. A live run uses only this value; --threshold is for the replay check.
THRESHOLD_4B = -0.8578214314085831   # warmup4, Nov 14: 35 of 68 steps scored, 17 flagged
THRESHOLD_27B = -0.11813126318156719  # pre-registered Nov 2 (27B warmup, median of the same step score)
WORKER_MODEL = {"4b": "qwen3:4b", "27b": "qwen3.8:27b"}
MAX_FAILURES_IN_ROW, MAX_FAILURES_TOTAL = 3, 10      # amendment of November 12, section 3
CODE_ERRORS = (KeyError, TypeError, AttributeError, IndexError, NameError, ZeroDivisionError, ValueError)
ARMS =("plan_once", "check_every_step", "triggered_logprob_mean")
PROTOCOLS_SHA256 = "bda274505010d8b5c656163b34f241a5e45b6eea7f037848a4474e8995af44a2"
EXPECTED_NOV2_4B = {"plan_once": (15, 0), "check_every_step": (13, 69), "triggered_logprob_mean": (15, 31)}
# eval_protocols.py EXPECTED_WARMUP["27b"], passed on bosgame on November 2
EXPECTED_NOV2_27B = {"plan_once": (20, 0), "check_every_step": (19, 75), "triggered_logprob_mean": (20, 30)}


def fixed_threshold(worker):
    """Read at call time, so a test can patch the constants."""
    return THRESHOLD_4B if worker == "4b" else THRESHOLD_27B


def worker_problem(rows, worker):
    """A shared row that names another worker model refuses the run (rows without the field pass)."""
    want = WORKER_MODEL[worker]
    for r in rows:
        w = r.get("worker")
        if w is not None and not str(w).startswith(want + ","):
            return f"{r['patient']} q{r['q_idx']}: shared row worker {w!r}, expected {want}"
    return None


class ReplayOnly:
    """The frozen supervisor cache key, read only. A missing reply stops the run."""

    def __init__(self, cache, model=SUPERVISOR_MODEL):
        self.cache, self.model, self.hits, self.fresh, self.context = cache, model, 0, 0, {}

    def generate(self, prompt, component, **kw):
        payload = {"backend": "gemini", "model": self.model, "params": {}, "prompt": prompt, "kw": kw}
        hit = self.cache.get(payload)
        if hit is None:
            raise gb.Halt("replay only: an aggregation reply is not in the cache; no call was made")
        self.hits += 1
        return {**hit, "cached": True}


def reviewed_for(row, arm, threshold):
    steps = row["steps"]
    if arm == "plan_once":
        return ()
    if arm == "check_every_step":
        return tuple(range(len(steps)))
    means = [mean_score(s["out"]) for s in steps]
    return tuple(i for i, m in enumerate(means) if m is not None and m < threshold)


def worker_findings(row, reviewed):
    return sum(1 for _, jobs in protocols.chosen_steps(row, reviewed) for j in jobs if j.get("answer"))


def score_question(row, sup, threshold):
    rec = {"question": row["question"], "options": row["options"]}
    out = []
    for arm in ARMS:
        reviewed = reviewed_for(row, arm, threshold)
        # run_protocol takes the step set explicitly; its label only names the row
        res = protocols.run_protocol(rec, row, sup, arm if arm != "triggered_logprob_mean" else "flip",
                                     reviewed=reviewed, component="score_aggregate")
        res.update({"arm": arm, "threshold": threshold, "worker_findings": worker_findings(row, reviewed)})
        out.append(res)
    return out


def protocols_problem():
    with open(protocols.__file__, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    return None if got == PROTOCOLS_SHA256 else f"protocols.py {got[:8]}, pushed {PROTOCOLS_SHA256[:8]}"


def seeded_budget(calls_log, args):
    """The caps count every fresh scoring call and dollar on this split over all run ids (amendment 2),
    so a new run id never resets them. The replay check is never counted (it makes no call)."""
    import glob
    budget = gb.Budget(args.max_calls, args.max_usd)
    pattern = os.path.join(os.path.dirname(calls_log), f"calls_score_{getattr(args, 'worker', '4b')}_{args.split}_*.jsonl")
    for path in glob.glob(pattern):
        if path.endswith("_replay.jsonl"):
            continue
        b = gb.Budget.from_calls_log(path, None, None)
        budget.calls += b.calls
        budget.usd += b.usd
    return budget


def make_live(args, cache, calls_log):
    from google import genai
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set in this shell.")
    raw = gb.PinnedGeminiSupervisor(SUPERVISOR_MODEL, genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
                                    args.expect_version)
    budget = seeded_budget(calls_log, args)
    return gb.GuardedCachedBackend(raw, cache, calls_log, budget, args.run_id, expect_version=args.expect_version)


def main(argv=None, make_supervisor=None):
    ap = argparse.ArgumentParser(description="score plan once, check every step and triggered log-prob (C3)")
    ap.add_argument("--shared", required=True)
    ap.add_argument("--split", choices=["warmup", "eval"], required=True)
    ap.add_argument("--worker", choices=["4b", "27b"], default="4b")
    ap.add_argument("--limit", type=int, help="score only the first N questions (the trace)")
    ap.add_argument("--print-prompts", action="store_true", help="print each aggregation prompt and reply")
    ap.add_argument("--threshold", type=float, help="replay check only; a live run uses the worker's fixed threshold")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--run-id", default="replay")
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--expect-nov2", action="store_true", help="compare with the November 2 key table")
    ap.add_argument("--expect-version", default=SUPERVISOR_MODEL)
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--max-calls", type=int)
    args = ap.parse_args(argv)
    if not args.replay_only and (args.max_usd is None or args.max_calls is None):
        ap.error("a live run needs --max-usd and --max-calls")
    if args.replay_only:
        if args.threshold is None:
            ap.error("the replay check needs --threshold")
    else:
        fixed = fixed_threshold(args.worker)
        if fixed is None:
            print(f"refused: the {args.worker} threshold is not fixed in score_c3.py yet (amendment 2)")
            return 5
        if args.threshold is not None and args.threshold != fixed:
            print(f"refused: --threshold {args.threshold!r} differs from the fixed {fixed!r}")
            return 5
        args.threshold = fixed
    problem = protocols_problem()
    if problem:
        print("refused: the frozen protocols.py differs from the pushed one: " + problem)
        return 5

    rows = [r for r in runlog.read_rows(args.shared) if r.get("status", "ok") == "ok"]
    if {str(r.get("split", args.split)) for r in rows} != {args.split}:
        print(f"refused: the shared log is not all {args.split}")
        return 5
    keys = collections.Counter((r["patient"], r["q_idx"]) for r in rows)
    if any(n > 1 for n in keys.values()):
        print("refused: a question appears twice among the ok rows")
        return 5
    problem = worker_problem(rows, args.worker)
    if problem:
        print("refused: " + problem)
        return 5
    if args.limit is not None:
        rows = rows[:args.limit]

    cache = cache_mod.DiskCache(os.path.expanduser(args.cache_dir))
    tag = "replay" if args.replay_only else args.run_id
    out_path = os.path.join(runlog.LOG_DIR, f"scores_{args.worker}_{args.split}_{tag}.jsonl")
    calls_log = os.path.join(runlog.LOG_DIR, f"calls_score_{args.worker}_{args.split}_{tag}.jsonl")
    if args.replay_only:
        sup = ReplayOnly(cache)
    else:
        sup = (make_supervisor or make_live)(args, cache, calls_log)
    if args.print_prompts:
        inner = sup.generate

        def traced(prompt, component, **kw):
            print(f"--- aggregation prompt ({component}, {len(prompt)} characters) ---\n{prompt}\n--- end prompt ---", flush=True)
            res = inner(prompt, component, **kw)
            print(f"--- reply (cached {bool(res.get('cached'))}) ---\n{res.get('text')}\n--- end reply ---", flush=True)
            return res
        sup.generate = traced
    done, failed, in_row, total = set(), collections.Counter(), 0, 0
    if os.path.exists(out_path):
        per = collections.defaultdict(set)
        for r in runlog.read_rows(out_path):
            if r.get("status") == "failed":
                failed[(r["patient"], r["q_idx"])] += 1
                total += 1
                in_row += 1
            else:
                per[(r["patient"], r["q_idx"])].add(r.get("arm"))
                in_row = 0
        done = {k for k, arms in per.items() if set(ARMS) <= arms}
    print(f"{'resuming' if done else 'logging to'} {out_path}: {len(rows)} questions, {len(done)} already scored; "
          f"worker {args.worker}; threshold {args.threshold!r}; "
          f"{'replay only, no cloud call' if args.replay_only else 'live'}", flush=True)

    stats = {a: {"correct": 0, "n": 0, "reviews": 0} for a in ARMS}
    for i, row in enumerate(rows):
        key = (row["patient"], row["q_idx"])
        if key in done:
            continue
        if failed[key] >= 2:
            print(f"[{i + 1}/{len(rows)}] {row['patient']} q{row['q_idx']}: failed twice, final (counts as wrong)")
            continue
        sup.context = {"patient": row["patient"], "q_idx": row["q_idx"], "attempt": failed[key] + 1}
        try:
            results = score_question(row, sup, args.threshold)
        except gb.CapReached as e:
            print(f"CAP REACHED ({e}); questions scored so far are kept.")
            return 2
        except gb.Halt as e:
            print(f"{row['patient']} q{row['q_idx']}: HALT ({type(e).__name__}: {e}). Find the cause before relaunching.")
            return 3
        except CODE_ERRORS as e:                 # a bug, not an outage: never a failed row (amendment 2)
            print(f"{row['patient']} q{row['q_idx']}: HALT (code error {type(e).__name__}: {e}). Fix the code first.")
            return 3
        except Exception as e:  # noqa: BLE001  (an empty reply or an API error: a failed row, rerun on the next launch)
            failed[key] += 1
            runlog.log_row(out_path, {"split": args.split, "run_id": tag, "patient": row["patient"],
                                      "q_idx": row["q_idx"], "status": "failed", "attempt": failed[key],
                                      "error_type": type(e).__name__, "error_message": str(e)[:300]})
            print(f"[{i + 1}/{len(rows)}] {row['patient']} q{row['q_idx']}: FAILED ({type(e).__name__}); "
                  f"failure {failed[key]} of 2", flush=True)
            in_row += 1
            total += 1
            if in_row >= MAX_FAILURES_IN_ROW or total > MAX_FAILURES_TOTAL:
                print(f"stopping: {in_row} failures in a row, {total} in this run. Find the cause (an outage "
                      f"is not a reason to let questions become final) before relaunching.")
                return 4
            continue
        in_row = 0
        for res in results:                       # all three arms of a question are written together
            res.update({"split": args.split, "run_id": tag, "shared_log": os.path.basename(args.shared)})
            runlog.log_row(out_path, res)
            st = stats[res["arm"]]
            st["n"] += 1
            st["correct"] += bool(res["correct"])
            st["reviews"] += res["n_reviews"]
        print(f"[{i + 1}/{len(rows)}] {row['patient']} q{row['q_idx']}: scored "
              f"(reviews {[r['n_reviews'] for r in results]}, fresh cloud calls so far {sup.fresh})", flush=True)

    print(f"done: {sum(s['n'] for s in stats.values()) // len(ARMS)} questions scored in this launch; "
          f"fresh cloud calls {sup.fresh}, cached {sup.hits}")
    if args.split == "eval":
        print("Counts only on the evaluation split; the analysis script reads the rows.")
        return 0
    for a in ARMS:
        print(f"  {a}: {stats[a]['correct']} of {stats[a]['n']}, reviews {stats[a]['reviews']}")
    if args.expect_nov2:
        if done:
            print("CHECK NOT POSSIBLE: part of the log was scored in an earlier launch; use a fresh log")
            return 1
        expected = EXPECTED_NOV2_4B if args.worker == "4b" else EXPECTED_NOV2_27B
        bad = [f"{a}: got {(stats[a]['correct'], stats[a]['reviews'])}, expected {expected[a]}"
               for a in ARMS if (stats[a]["correct"], stats[a]["reviews"]) != expected[a]]
        if bad:
            print("CHECK FAILED: " + "; ".join(bad))
            return 1
        print("CHECK PASSED: the November 2 warmup scores are reproduced from the cache, with no call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
