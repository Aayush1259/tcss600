"""4B alone on the evaluation set, with the guards of the pipeline (November 14).

The design is the one confirmed on November 12 and frozen in local_alone_vote.py (pushed November
12): each 6,000-character chunk of the padded record, with the question and the options, answers a
letter or "none"; "none" and malformed replies do not vote; the most common letter wins; a tie is
broken by random.Random(20261109 + question index); a question with no letter vote is wrong. This
file imports chunk_prompt, parse_vote, vote and the reply schema from local_alone_vote.py unchanged
and refuses to start if that file's hash differs from the pushed one.

What it adds for the evaluation run:
- every call through guarded_backend.guarded_chat ("truncate": false, the stop, fit and
  characters-per-token checks, the 300-token reply rule) and, after every call, the GPU check of
  amendment 2 against the reference given on the command line (required on the evaluation split);
- the GPU lock shared with run_shared_c2.py, so it never runs at the same time as a pipeline batch;
- a fixed log per run id, resume (questions done are skipped), a failed row for a connection
  failure (rerun once on the next launch, then counted wrong), a halt row and exit 3 for a guard
  failure;
- counts only on the screen; the prediction and the match are in the log for the analysis;
- --compare-log: how many predictions equal those of an earlier log (for the warmup recheck
  against November 11), with no accuracy;
- --check: per-run counts (requests equal chunks, normal stops, one GPU residency, all-none
  questions, ties, malformed replies, reply cuts), no accuracy.

    BASELINE_LOG_DIR=~/c2_logs python local_alone_eval.py run --split eval --run-id la_eval \
        --expect-size 3059925389 --expect-size-vram 2316912557
    BASELINE_LOG_DIR=~/c2_logs python local_alone_eval.py check --split eval --run-id la_eval
"""
import argparse
import collections
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data_loader  # noqa: E402
import grader  # noqa: E402
import guarded_backend as gb  # noqa: E402
import local_alone_vote as lav  # noqa: E402
import ollama_guard  # noqa: E402
import run_shared_c2 as rs  # noqa: E402
import runlog  # noqa: E402

LAV_SHA256 = "c740e354db4611d4ba6487ae01e06c93bc6a7e32a8eccc09ad08d088191e1c50"
MAX_FAILURES_IN_ROW, MAX_FAILURES_TOTAL = 3, 10


class VoteWorker(gb.GuardedOllamaWorker):
    """The pipeline's guarded worker with the vote schema instead of the job schema, and no log-probs.
    The residency check is the same: after every call, before the reply is used."""

    def call(self, prompt, logprobs=False):
        out = gb.guarded_chat(self.model, prompt, self.num_ctx, self.num_predict, fmt=lav.VOTE_SCHEMA,
                              logprobs=False, post=self.post, retries=self.attempts, sleep=self.sleep,
                              timeout=self.timeout)
        size, vram = self.residency()
        if not vram:
            raise gb.GpuHalt(f"the model is not in GPU memory after this call (size {size}, size_vram {vram})")
        if self.reference is None:
            self.reference = {"model_size": size, "model_size_vram": vram}
        elif (size, vram) != (self.reference["model_size"], self.reference["model_size_vram"]):
            raise gb.GpuHalt(f"GPU residency changed: size {size}, size_vram {vram}; reference "
                             f"size {self.reference['model_size']}, size_vram {self.reference['model_size_vram']}")
        return {**out, "guard": gb.GUARD_TAG, **self.env, "model_size": size, "model_size_vram": vram}


def lav_problem():
    with open(lav.__file__, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    return None if got == LAV_SHA256 else f"local_alone_vote.py {got[:8]}, pushed {LAV_SHA256[:8]}"


def run_question(q_index, rec, worker):
    chunks = data_loader.chunk_text("\n\n---\n\n".join(rec["documents"]), data_loader.CHUNK_CHARS)
    per_chunk, letters = [], []
    for ci, chunk in enumerate(chunks):
        r = worker.call(lav.chunk_prompt(rec["question"], rec["options"], chunk))
        kind, letter = lav.parse_vote(r["text"])
        if kind == "letter":
            letters.append(letter)
        per_chunk.append({"chunk": ci, "vote": kind, "letter": letter, "text": r["text"],
                          "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "done_reason": r["done_reason"], "reply_cut": r.get("reply_cut"),
                          "attempts": r.get("attempts"), "chars_per_token": r["chars_per_token"],
                          "wall_time_s": r["wall_time_s"], "model_size": r["model_size"],
                          "model_size_vram": r["model_size_vram"]})
    pred, counts, tied = lav.vote(letters, q_index)
    n_none = sum(1 for c in per_chunk if c["vote"] == "none")
    n_bad = sum(1 for c in per_chunk if c["vote"] == "malformed")
    return {"q_index": q_index, "patient": rec["patient"], "q_idx": rec["q_idx"], "q_no": rec["q_no"],
            "protocol": lav.PROTOCOL, "n_chunks": len(chunks), "requests": len(per_chunk), "votes": counts,
            "n_none": n_none, "n_malformed": n_bad, "tied": tied,
            "tie_seed": lav.SEED + q_index if tied else None, "all_none": pred is None,
            "predicted": pred, "gold": rec["gold"], "match": grader.matches(pred, rec["gold"], rec["options"]),
            "chunks": per_chunk}


def read(path):
    return runlog.read_rows(path) if os.path.exists(path) else []


def cmd_run(args, make_worker=None):
    problem = lav_problem()
    if problem:
        print("refused: the frozen 4B alone script differs from the pushed one: " + problem)
        return 5
    recs = rs.load_records(args.split)
    if args.limit:
        recs = recs[:args.limit]
    bad = [(r["patient"], r["q_idx"], rs.record_problems(r)) for r in recs if rs.record_problems(r)]
    if bad:
        print(f"refused: padding check failed before any call: {bad[:5]}")
        return 5
    path = os.path.join(runlog.LOG_DIR, f"local_alone_4b_{args.split}_{args.run_id}.jsonl")
    lock = rs.take_lock(os.path.join(rs.HERE, ".gpu.lock"))
    if lock:
        print("refused: " + lock)
        return 5
    worker = (make_worker or (lambda: VoteWorker(rs.WORKER_MODEL, rs.NUM_CTX, rs.NUM_PREDICT)))()
    if args.expect_size_vram:
        worker.reference = {"model_size": args.expect_size, "model_size_vram": args.expect_size_vram}
    rows = read(path)
    done, failures, final, in_row, total = rs.log_state(rows)
    launch = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    print(f"{'resuming' if rows else 'logging to'} {path}: {len(done)} done, {len(final)} final failures; "
          f"GPU reference {worker.reference or 'from the first call'}", flush=True)
    for q_index, rec in enumerate(recs):
        key = (rec["patient"], rec["q_idx"])
        tag = f"[{q_index + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}"
        if key in done or key in final:
            continue
        base = {"split": args.split, "run_id": args.run_id, "launch": launch, "patient": rec["patient"],
                "q_idx": rec["q_idx"], "attempt": failures.get(key, 0) + 1, "ollama": worker.env,
                "worker": f"{rs.WORKER_MODEL}, ctx {rs.NUM_CTX}, think off, guarded {gb.GUARD_TAG}",
                "padding_n": rec["padding_n"], "padding_patients": rec["padding_patients"],
                **rs.input_hashes(rec)}
        t0 = time.time()
        try:
            row = run_question(q_index, rec, worker)
        except (gb.Halt, ollama_guard.GuardError, ollama_guard.Rejected) as e:
            runlog.log_row(path, {**base, "status": "halt", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: HALT ({type(e).__name__}: {rs.scrub(str(e))}). Find the cause before relaunching.")
            return 3
        except (KeyError, TypeError, AttributeError, IndexError, NameError, ZeroDivisionError, ValueError) as e:
            # a bug, not an outage: a halt row, never a failed attempt (amendment 2)
            runlog.log_row(path, {**base, "status": "halt", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: HALT (code error {type(e).__name__}). Fix the code first.")
            return 3
        except Exception as e:  # noqa: BLE001
            failures[key] = failures.get(key, 0) + 1
            in_row += 1
            total += 1
            runlog.log_row(path, {**base, "status": "failed", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: FAILED ({type(e).__name__}); failure {failures[key]} of 2", flush=True)
            if in_row >= MAX_FAILURES_IN_ROW or total > MAX_FAILURES_TOTAL:
                print(f"stopping: {in_row} failures in a row, {total} in this run")
                return 4
            continue
        in_row = 0
        runlog.log_row(path, {**base, **row, "status": "ok", "wall_time_s": round(time.time() - t0, 1)})
        print(f"{tag}: chunks {row['n_chunks']}, letter votes {sum(row['votes'].values())}, none {row['n_none']}, "
              f"malformed {row['n_malformed']}, tied {row['tied']}, all none {row['all_none']}", flush=True)
    print("done. Counts only; the log holds the predictions for the analysis.")
    return 0


def cmd_check(args):
    rows = read(os.path.join(runlog.LOG_DIR, f"local_alone_4b_{args.split}_{args.run_id}.jsonl"))
    ok = [r for r in rows if r.get("status") == "ok"]
    fails = []

    def check(cond, text):
        print(("PASS  " if cond else "FAIL  ") + text)
        if not cond:
            fails.append(text)

    keys = collections.Counter((r["patient"], r["q_idx"]) for r in ok)
    chunks = [c for r in ok for c in r["chunks"]]
    print(f"run {args.run_id}: {len(rows)} rows {dict(collections.Counter(r.get('status') for r in rows))}")
    check(all(n == 1 for n in keys.values()), "no question answered twice")
    if args.expect_questions:
        failed = collections.Counter((r["patient"], r["q_idx"]) for r in rows if r.get("status") == "failed")
        final = {k for k, n in failed.items() if n >= 2 and k not in keys}
        check(len(keys) + len(final) == args.expect_questions,
              f"every question answered or final ({len(keys)} answered, {len(final)} final, of {args.expect_questions})")
    check(all(r["requests"] == r["n_chunks"] == len(r["chunks"]) for r in ok), f"one request per chunk ({len(chunks)} chunks)")
    check(all(r["n_none"] + r["n_malformed"] + sum(r["votes"].values()) == r["n_chunks"] for r in ok),
          "letter votes, none and malformed add up to the chunks")
    check(all(c["done_reason"] == "stop" or c.get("reply_cut") for c in chunks), "every call a normal stop or a reply cut")
    res = collections.Counter((c["model_size"], c["model_size_vram"]) for c in chunks)
    check(len(res) == 1 and all(v for _, v in res), f"one GPU residency, never zero {dict(res)}")
    if args.expect_size_vram:
        check(set(res) == {(args.expect_size, args.expect_size_vram)}, "GPU residency equals the reference")
    check(all(c["prompt_tokens"] + rs.NUM_PREDICT <= rs.NUM_CTX for c in chunks), "every prompt plus 300 fits 3,072")
    print(f"      all none {sum(r['all_none'] for r in ok)}, ties {sum(r['tied'] for r in ok)}, "
          f"malformed {sum(r['n_malformed'] for r in ok)}, reply cuts {sum(bool(c.get('reply_cut')) for c in chunks)}, "
          f"retried calls {sum((c.get('attempts') or 1) > 1 for c in chunks)}")
    if args.compare_log:
        old = {(r["patient"], r["q_idx"]): r.get("predicted") for r in read(args.compare_log)}
        both = [k for k in keys if k in old]
        same = sum(1 for r in ok if (r["patient"], r["q_idx"]) in old and r["predicted"] == old[(r["patient"], r["q_idx"])])
        print(f"      same prediction as {os.path.basename(args.compare_log)}: {same} of {len(both)} questions")
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
    return 0 if not fails else 1


def main(argv=None, make_worker=None):
    ap = argparse.ArgumentParser(description="4B alone, guarded (per-chunk vote)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["warmup", "eval"], required=True)
        p.add_argument("--run-id", required=True)
        p.add_argument("--expect-size", type=int)
        p.add_argument("--expect-size-vram", type=int)
    sub.choices["run"].add_argument("--limit", type=int)
    sub.choices["check"].add_argument("--expect-questions", type=int)
    sub.choices["check"].add_argument("--compare-log")
    args = ap.parse_args(argv)
    if bool(args.expect_size) != bool(args.expect_size_vram):
        ap.error("--expect-size and --expect-size-vram go together")
    if args.cmd == "run" and args.split == "eval" and not args.expect_size_vram:
        ap.error("the eval needs the GPU residency reference: --expect-size and --expect-size-vram")
    return cmd_run(args, make_worker) if args.cmd == "run" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
