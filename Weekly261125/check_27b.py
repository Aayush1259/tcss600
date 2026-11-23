"""Count check for the 27B run of November 6 before it is scored (Amendment 3, section 3). Read only.

The run predates the current guards, so before any score this checks, from its own two logs, that
  - the shared log has the expected questions once each, all on the eval split, worker qwen3.8:27b at
    the expected context, padding N = 2, and the same 200 questions as the given reference shared log;
  - the calls log has one plan call per question, one review call per reviewed step, and one worker
    call per worker job held in the shared log (first pass, redone jobs and agreement reruns);
  - no worker prompt reached half the context (Ollama cuts a long prompt to half the context plus 2
    tokens), so no prompt was cut.
It prints counts only (no answer, no accuracy) and exits 0 when every check passes, 1 otherwise. If a
check fails, the 27B run is reported as not scored, with the reason, and it is not rerun.

    python check_27b.py --shared shared_27b_eval_20261102_051205.jsonl --calls calls_27b_eval.jsonl \
        --reference <4B eval shared log> --expect-questions 200 --expect-plans 200 --expect-reviews 713 \
        --expect-worker-calls 48592 --num-ctx 16384
"""
import argparse
import collections
import json
import sys


def rows_of(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jobs_of(step):
    """Every worker job a step holds: first pass, redone jobs, agreement reruns."""
    jobs = list(step.get("out") or [])
    jobs += list(step.get("reviewed") or [])
    for run in step.get("agreement_runs") or []:
        jobs += run if isinstance(run, list) else [run]
    return [j for j in jobs if isinstance(j, dict)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--reference", required=True, help="a shared log of the same 200 questions (the 4B eval)")
    ap.add_argument("--expect-questions", type=int, required=True)
    ap.add_argument("--expect-plans", type=int, required=True)
    ap.add_argument("--expect-reviews", type=int, required=True)
    ap.add_argument("--expect-worker-calls", type=int, required=True)
    ap.add_argument("--num-ctx", type=int, required=True)
    ap.add_argument("--worker", default="qwen3.8:27b")
    args = ap.parse_args(argv)

    shared, calls = rows_of(args.shared), rows_of(args.calls)
    ref_rows = {(r["patient"], r["q_idx"]): r for r in rows_of(args.reference) if r.get("status", "ok") == "ok"}
    ref = set(ref_rows)
    keys = collections.Counter((r["patient"], r["q_idx"]) for r in shared)
    by_comp = collections.Counter(c["component"] for c in calls)
    worker_calls = [c for c in calls if c["component"] == "worker"]
    jobs = [j for r in shared for s in r["steps"] for j in jobs_of(s)]
    reviews_in_shared = sum(1 for r in shared for s in r["steps"] if s.get("review") is not None)
    half = args.num_ctx // 2
    max_call_prompt = max((c["prompt_tokens"] or 0) for c in worker_calls) if worker_calls else None
    max_job_prompt = max((j.get("prompt_tokens") or 0) for j in jobs) if jobs else None

    checks = [
        ("questions in the shared log", len(shared), args.expect_questions),
        ("questions once each", sum(1 for n in keys.values() if n == 1), args.expect_questions),
        ("same questions as the reference", len(set(keys) & ref), args.expect_questions),
        ("rows on the eval split", sum(1 for r in shared if r.get("split") == "eval"), args.expect_questions),
        ("rows with padding N = 2", sum(1 for r in shared if r.get("padding_n") == 2), args.expect_questions),
        ("same padding patients and chunk count as the reference",
         sum(1 for r in shared if (r["patient"], r["q_idx"]) in ref_rows
             and r.get("padding_patients") == ref_rows[(r["patient"], r["q_idx"])].get("padding_patients")
             and r.get("n_chunks") == ref_rows[(r["patient"], r["q_idx"])].get("n_chunks")), args.expect_questions),
        (f"rows with worker {args.worker} at ctx {args.num_ctx}",
         sum(1 for r in shared if str(r.get("worker", "")).startswith(f"{args.worker}, ctx {args.num_ctx},")),
         args.expect_questions),
        ("plan calls in the calls log", by_comp["plan"], args.expect_plans),
        ("plans in the shared log", sum(1 for r in shared if r.get("plan")), args.expect_plans),
        ("review calls in the calls log", by_comp["review"], args.expect_reviews),
        ("reviewed steps in the shared log", reviews_in_shared, args.expect_reviews),
        ("worker calls in the calls log", len(worker_calls), args.expect_worker_calls),
        ("worker jobs in the shared log", len(jobs), args.expect_worker_calls),
        ("calls by a model other than the worker or the supervisor",
         sum(1 for c in calls if c["component"] == "worker" and c["model"] != args.worker), 0),
        (f"worker calls at or above half the context ({half})",
         sum(1 for c in worker_calls if (c["prompt_tokens"] or 0) >= half), 0),
        (f"worker jobs at or above half the context ({half})",
         sum(1 for j in jobs if (j.get("prompt_tokens") or 0) >= half), 0),
    ]
    ok = True
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {got} (expected {want})")
    print(f"largest worker prompt: {max_call_prompt} tokens in the calls log, {max_job_prompt} in the shared log; "
          f"other components: {dict((k, v) for k, v in by_comp.items() if k not in ('plan', 'review', 'worker'))}")
    first = min(c["logged_at"] for c in calls)
    last = max(c["logged_at"] for c in calls)
    print(f"calls from {first[:19]} to {last[:19]}")
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED: report the 27B run as not scored, with the reason; do not rerun")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
