"""Count check for a guarded shared-parts run (C4, November 13). Read-only; prints no accuracy.

For one run id it reads the shared log and the calls log written by run_shared_c2.py and checks:
 1. rows: answered, failed and halt rows; no question answered twice; one config id;
 2. every worker call used (fresh or cached) carries the guard tag, one Ollama version and one
    model digest, and its prompt plus the 300-token reply fits the context; it stopped normally or
    used exactly its 300 reply tokens (reply_cut). Reply cuts are counted, and more than 1 percent
    of worker calls fails the check (the earlier rate was 24 in 44,729);
    Every worker call has one GPU residency (model size and bytes in GPU memory from /api/ps),
    never zero, equal to --expect-size / --expect-size-vram when given; retried calls are counted;
    With --expect-questions N: every one of the N questions is answered or final (failed twice).
    Jobs whose reply used all 300 tokens are counted, with how many were still read as an answer;
 3. every supervisor call used carries the expected model version;
 4. for each answered question, the worker calls logged in the launch that answered it equal its
    jobs, and over the whole run no job went to Ollama twice (fresh calls at most its jobs), so a
    stop and a relaunch are counted correctly;
 5. with --ollama-journal (Ollama's service log for the run's time window, for example
    `journalctl -u ollama --since "..." > journal.txt`): no "truncating input prompt" line, and the
    chat requests minus the fresh worker calls logged between 0 and the number of launches (each
    stop can leave one request that was sent but not logged); one layer split ("offloaded N/37
    layers to GPU") for every model load; GPU discovery timeouts reported; model process crashes ("CUDA error",
    "signal: aborted") counted and accepted only if a reload follows before the next answered request,
    and every request that got an HTTP 500 was retried or recorded (amendment 2);
 6. with --verify-inputs: each answered question's documents and question hashes equal what the
    frozen data_loader builds now at padding 2 (the loader cloud alone used);
 7. seconds per fresh worker job (median and largest) and the run's fresh supervisor dollars.
Exit code 0 if every check passes, 1 otherwise.

    python check_shared.py --run-id trace1 --split warmup
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import runlog  # noqa: E402

GUARD_TAG, NUM_CTX, NUM_PREDICT = "c2-v2", 3072, 300


def read(path):
    return runlog.read_rows(path) if os.path.exists(path) else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", choices=["warmup", "eval"], required=True)
    ap.add_argument("--log-dir", default=runlog.LOG_DIR)
    ap.add_argument("--expect-version", default="gemini-3.8-flash")
    ap.add_argument("--ollama-journal")
    ap.add_argument("--verify-inputs", action="store_true")
    ap.add_argument("--expect-questions", type=int,
                    help="questions the run must cover so far (20 warmup; 50 times the batch number on the eval)")
    ap.add_argument("--expect-size", type=int)
    ap.add_argument("--expect-size-vram", type=int)
    args = ap.parse_args(argv)

    shared = read(os.path.join(args.log_dir, f"shared_4b_{args.split}_{args.run_id}.jsonl"))
    calls = read(os.path.join(args.log_dir, f"calls_4b_{args.split}_{args.run_id}.jsonl"))
    fails = []

    def check(ok, text):
        print(("PASS  " if ok else "FAIL  ") + text)
        if not ok:
            fails.append(text)

    ok_rows = [r for r in shared if r.get("status") == "ok"]
    by_status = collections.Counter(r.get("status") for r in shared)
    answered = collections.Counter((r["patient"], r["q_idx"]) for r in ok_rows)
    print(f"run {args.run_id}: {len(shared)} rows {dict(by_status)}; {len(answered)} questions answered, "
          f"{len({r['patient'] for r in ok_rows})} patients")
    check(len(shared) > 0, "the shared log has rows")
    check(all(n == 1 for n in answered.values()), "no question answered twice")
    if args.expect_questions:
        fails_per_q = collections.Counter((r["patient"], r["q_idx"]) for r in shared if r.get("status") == "failed")
        final = {k for k, n in fails_per_q.items() if n >= 2 and k not in answered}
        open_q = args.expect_questions - len(answered) - len(final)
        check(open_q == 0, f"every question answered or final: {len(answered)} answered, {len(final)} final "
                           f"(counted wrong), {open_q} still open of {args.expect_questions}")
    cut_jobs = [j for r in ok_rows for s in r["steps"] for j in s["out"] + (s["reviewed"] or [])
                if j.get("completion_tokens") == NUM_PREDICT]
    print(f"      jobs whose reply used all {NUM_PREDICT} tokens: {len(cut_jobs)}, of them read as an answer: "
          f"{sum(1 for j in cut_jobs if j.get('answer'))}, unreadable: {sum(1 for j in cut_jobs if not j.get('parse_ok'))}")
    check(len({r.get("config_id") for r in shared}) <= 1, f"one config id {sorted({str(r.get('config_id')) for r in shared})}")

    worker = [c for c in calls if c.get("kind") == "ollama"]
    sup = [c for c in calls if c.get("kind") == "gemini" and c.get("status") == "ok"]
    rejected = [c for c in calls if c.get("status") not in (None, "ok")]
    check(all(c.get("guard") == GUARD_TAG for c in worker), f"every worker call has the guard tag ({len(worker)} calls)")
    cut = [c for c in worker if c.get("done_reason") == "length" and c.get("reply_cut")
           and c.get("completion_tokens") == NUM_PREDICT]
    check(all(c.get("done_reason") == "stop" or c in cut for c in worker),
          f"every worker call stopped normally or at the {NUM_PREDICT}-token reply cap")
    rate = len(cut) / len(worker) if worker else 0.0
    check(rate <= 0.01, f"reply cuts at {NUM_PREDICT} tokens: {len(cut)} of {len(worker)} worker calls "
                        f"({100 * rate:.2f} percent, limit 1)")
    check(len({c.get("ollama_version") for c in worker}) <= 1 and len({c.get("model_digest") for c in worker}) <= 1,
          f"one Ollama version {sorted({str(c.get('ollama_version')) for c in worker})} and one digest")
    res = collections.Counter((c.get("model_size"), c.get("model_size_vram")) for c in worker)
    check(len(res) == 1 and all(v for _, v in res),
          f"one GPU residency on every worker call, never zero (size, size_vram: {dict(res)})")
    if args.expect_size_vram:
        check(set(res) == {(args.expect_size, args.expect_size_vram)},
              f"GPU residency equals the reference {args.expect_size} / {args.expect_size_vram}")
    retried = sum(1 for c in worker if (c.get("attempts") or 1) > 1)
    print(f"      worker calls that needed a retry: {retried}")
    too_big = [c for c in worker if (c.get("prompt_tokens") or 0) + NUM_PREDICT > NUM_CTX]
    largest = max((c.get("prompt_tokens") or 0 for c in worker), default=0)
    check(not too_big, f"every worker prompt plus {NUM_PREDICT} fits {NUM_CTX} (largest prompt {largest} tokens)")
    check(all(c.get("model_version") == args.expect_version for c in sup),
          f"every supervisor call has version {args.expect_version} ({len(sup)} calls; "
          f"{dict(collections.Counter(c.get('model_version') for c in sup))})")
    print(f"      rejected supervisor replies logged: {len(rejected)} {dict(collections.Counter(c.get('status') for c in rejected))}")

    per_launch = collections.Counter((c.get("patient"), c.get("q_idx"), c.get("launch")) for c in worker)
    mismatched = [(r["patient"], r["q_idx"], r["n_jobs"], per_launch[(r["patient"], r["q_idx"], r.get("launch"))])
                  for r in ok_rows if per_launch[(r["patient"], r["q_idx"], r.get("launch"))] != r["n_jobs"]]
    check(not mismatched, f"worker calls in the answering launch equal jobs for every answered question {mismatched[:3]}")
    fresh_w = [c for c in worker if not c.get("cached")]
    fresh_q = collections.Counter((c.get("patient"), c.get("q_idx")) for c in fresh_w)
    twice = [(r["patient"], r["q_idx"], r["n_jobs"], fresh_q[(r["patient"], r["q_idx"])])
             for r in ok_rows if fresh_q[(r["patient"], r["q_idx"])] > r["n_jobs"]]
    check(not twice, f"no job sent to Ollama twice (fresh worker calls at most the jobs) {twice[:3]}")
    launches = {x.get("launch") for x in calls + shared if x.get("launch")}
    print(f"      launches in this run: {len(launches)}")

    if args.ollama_journal:
        with open(args.ollama_journal, encoding="utf-8", errors="replace") as f:
            text = f.read()
        truncating = text.count("truncating input prompt")
        chats = sum(1 for line in text.splitlines() if "/api/chat" in line)
        check(truncating == 0, f"no 'truncating input prompt' in the Ollama journal ({truncating})")
        splits = collections.Counter(line.split("load_tensors:", 1)[-1].strip()
                                     for line in text.splitlines() if "offloaded" in line and "layers to GPU" in line)
        check(len(splits) <= 1, f"one layer split for every model load in the journal {dict(splits)}")
        # Amendment 2 (Nov 14): a model process crash is reported; it is accepted only if Ollama reloaded the
        # model (a load line) before the next request was answered. With the one-split check above and the
        # GPU residency on every call, the replies after the crash ran on the same setup. GPU discovery
        # timeouts are reported only: warmup4 had them during a reload and once on their own, with every
        # call still at the reference; a real loss of the GPU halts the run through the residency check.
        lost = text.count("GPU discovery watchdog timed out")
        print(f"      GPU discovery timeouts in the journal: {lost} (reported; the residency check decides)")
        crashes, pending, served_before_reload = 0, False, 0
        for line in text.splitlines():
            if "CUDA error" in line or "llama-server terminated" in line or "signal: aborted" in line:
                if not pending:
                    crashes += 1
                pending = True
            elif "offloaded" in line and "layers to GPU" in line:
                pending = False
            elif pending and "/api/chat" in line and "| 500 |" not in line:
                served_before_reload += 1
        check(served_before_reload == 0, f"model process crashes in the journal: {crashes}, each followed by a reload "
                                         f"before the next answered request ({served_before_reload} answered before a reload)")
        errors_500 = sum(1 for line in text.splitlines() if "/api/chat" in line and "| 500 |" in line)
        extra = chats - len(fresh_w)
        retries = sum((c.get("attempts") or 1) - 1 for c in worker)
        failed_rows = sum(1 for r in shared if r.get("status") in ("failed", "halt"))
        check(errors_500 <= retries + failed_rows, f"every request that got an HTTP 500 was retried or recorded "
                                                    f"({errors_500} errors, {retries} retries, {failed_rows} failed or halt rows)")
        allowed = len(launches) + retries + failed_rows
        check(0 <= extra <= allowed, f"chat requests in the journal {chats}, fresh worker calls logged {len(fresh_w)}: "
                                     f"{extra} unlogged, allowed {allowed} (one per launch {len(launches)}, "
                                     f"retried requests {retries}, failed or halt rows {failed_rows})")

    if args.verify_inputs:
        import data_loader
        import run_shared_c2
        recs = {(r["patient"], r["q_idx"]): r for r in data_loader.load_split(args.split, pad_n=2)}
        wrong = [k for k, r in ((k, r) for r in ok_rows for k in [(r["patient"], r["q_idx"])])
                 if run_shared_c2.input_hashes(recs[k]) != {"documents_sha256": r.get("documents_sha256"),
                                                            "question_sha256": r.get("question_sha256")}]
        check(not wrong, f"inputs equal the frozen loader at padding 2 for every answered question {wrong[:3]}")

    secs = [c.get("wall_time_s") for c in fresh_w if c.get("wall_time_s")]
    usd = sum(c.get("usd") or 0.0 for c in calls if c.get("kind") == "gemini" and not c.get("cached"))
    if secs:
        print(f"      fresh worker jobs {len(secs)}: median {statistics.median(secs):.2f} s, largest {max(secs):.1f} s")
    print(f"      fresh supervisor dollars in this run: {usd:.4f}")
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
