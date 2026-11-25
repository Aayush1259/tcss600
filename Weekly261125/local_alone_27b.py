"""27B alone on the lab machine: the per-chunk vote of 4B alone, with qwen3.8:27b (November 23).

A copy of local_alone_eval.py (4B alone, pushed November 16) for Prof. Cai's decision of November 23
(Option B first). Only these change: the worker is qwen3.8:27b at a context of 16,384 (as in the
November 6 supervised 27B run); the log is local_alone_27b_<split>_<run id>.jsonl; the fit check
uses that context; the journal check requires exactly one model load (the run starts from an unloaded
model, so a check that matched no load line cannot pass) and also counts "llama runner process has
terminated" as a model process crash. Everything else is the 4B alone file unchanged: the chunks, the
prompt, the vote and the tie rule come from local_alone_vote.py (pushed November 12, hash checked),
every call goes through guarded_backend.guarded_chat ("truncate": false, thinking off, temperature 0,
300-token replies), the GPU residency check after every call, resume, failed and halt rows, counts
only on the screen. Note of November 23 (note-2026-11-23-27b-alone.md) fixes the design.

    ollama stop qwen3.8:27b
    BASELINE_LOG_DIR=~/c27_logs python local_alone_27b.py run --split warmup --limit 1 --run-id la27_trace
    BASELINE_LOG_DIR=~/c27_logs python local_alone_27b.py check --split warmup --run-id la27_trace \
        --ollama-journal journal.txt
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
WORKER_MODEL, NUM_CTX, NUM_PREDICT = "qwen3.8:27b", 16384, 300
LOG_PREFIX = "local_alone_27b_"


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
    path = os.path.join(runlog.LOG_DIR, f"{LOG_PREFIX}{args.split}_{args.run_id}.jsonl")
    lock = rs.take_lock(os.path.join(rs.HERE, ".gpu.lock"))
    if lock:
        print("refused: " + lock)
        return 5
    worker = (make_worker or (lambda: VoteWorker(WORKER_MODEL, NUM_CTX, NUM_PREDICT)))()
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
                "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think off, guarded {gb.GUARD_TAG}",
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


def journal_checks(path, rows, chunks, check):
    """Ollama's service log for the run's window, read by the rules of amendment 2 section 3, as
    check_shared.py reads it for the pipeline: no cut prompt; one layer split for every model load; a
    model process crash accepted only if a reload comes before the next answered request; every HTTP 500
    retried or recorded. A question's chunk calls are logged only when the question ends, so a stopped
    launch, a failed or a halted question can leave up to one question's chunks sent but not logged."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    truncating = sum(line.count("truncating input prompt") for line in lines)
    check(truncating == 0, f"no 'truncating input prompt' in the Ollama journal ({truncating})")
    splits = collections.Counter(line.split("load_tensors:", 1)[-1].strip()
                                 for line in lines if "offloaded" in line and "layers to GPU" in line)
    check(len(splits) == 1, f"exactly one layer split, and at least one model load, in the journal {dict(splits)}")
    lost = sum(line.count("GPU discovery watchdog timed out") for line in lines)
    print(f"      GPU discovery timeouts in the journal: {lost} (reported; the residency check decides)")
    crashes, pending, served_before_reload = 0, False, 0
    for line in lines:
        if ("CUDA error" in line or "llama-server terminated" in line or "signal: aborted" in line
                or "llama runner process has terminated" in line):
            if not pending:
                crashes += 1
            pending = True
        elif "offloaded" in line and "layers to GPU" in line:
            pending = False
        elif pending and "/api/chat" in line and "| 500 |" not in line:
            served_before_reload += 1
    check(served_before_reload == 0, f"model process crashes in the journal: {crashes}, each followed by a reload "
                                     f"before the next answered request ({served_before_reload} answered before a reload)")
    chats = sum(1 for line in lines if "/api/chat" in line)
    errors_500 = sum(1 for line in lines if "/api/chat" in line and "| 500 |" in line)
    retries = sum((c.get("attempts") or 1) - 1 for c in chunks)
    stopped = sum(1 for r in rows if r.get("status") in ("failed", "halt"))
    launches = len({r.get("launch") for r in rows if r.get("launch")})
    check(errors_500 <= retries + stopped, f"every request that got an HTTP 500 was retried or recorded "
                                           f"({errors_500} errors, {retries} retries, {stopped} failed or halt rows)")
    most = max((r["n_chunks"] for r in rows if r.get("status") == "ok"), default=0)
    extra, allowed = chats - len(chunks), retries + (launches + stopped) * most
    check(0 <= extra <= allowed, f"chat requests in the journal {chats}, chunk calls logged {len(chunks)}: {extra} "
                                 f"unlogged, allowed {allowed} (retries {retries}; {launches} launches and {stopped} "
                                 f"failed or halt rows, each up to {most} chunks)")


def cmd_check(args):
    rows = read(os.path.join(runlog.LOG_DIR, f"{LOG_PREFIX}{args.split}_{args.run_id}.jsonl"))
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
    cut = [c for c in chunks if c["done_reason"] == "length" and c.get("reply_cut")
           and c.get("completion_tokens") == NUM_PREDICT]
    check(all(c["done_reason"] == "stop" or c in cut for c in chunks), "every call a normal stop or a reply cut")
    rate = len(cut) / len(chunks) if chunks else 0.0
    check(rate <= 0.01, f"reply cuts at {NUM_PREDICT} tokens: {len(cut)} of {len(chunks)} calls "
                        f"({100 * rate:.2f} percent, limit 1; amendment 2 section 3)")
    res = collections.Counter((c["model_size"], c["model_size_vram"]) for c in chunks)
    check(len(res) == 1 and all(v for _, v in res), f"one GPU residency, never zero {dict(res)}")
    if args.expect_size_vram:
        check(set(res) == {(args.expect_size, args.expect_size_vram)}, "GPU residency equals the reference")
    check(all(c["prompt_tokens"] + NUM_PREDICT <= NUM_CTX for c in chunks), f"every prompt plus {NUM_PREDICT} fits {NUM_CTX:,}")
    print(f"      all none {sum(r['all_none'] for r in ok)}, ties {sum(r['tied'] for r in ok)}, "
          f"malformed {sum(r['n_malformed'] for r in ok)}, reply cuts {sum(bool(c.get('reply_cut')) for c in chunks)}, "
          f"retried calls {sum((c.get('attempts') or 1) > 1 for c in chunks)}")
    if args.compare_log:
        old = {(r["patient"], r["q_idx"]): r.get("predicted") for r in read(args.compare_log)}
        both = [k for k in keys if k in old]
        same = sum(1 for r in ok if (r["patient"], r["q_idx"]) in old and r["predicted"] == old[(r["patient"], r["q_idx"])])
        print(f"      same prediction as {os.path.basename(args.compare_log)}: {same} of {len(both)} questions")
    if args.ollama_journal:
        journal_checks(args.ollama_journal, rows, chunks, check)
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
    return 0 if not fails else 1


def main(argv=None, make_worker=None):
    ap = argparse.ArgumentParser(description="27B alone, guarded (per-chunk vote)")
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
    sub.choices["check"].add_argument("--ollama-journal", help="journalctl -u ollama for the run's window")
    args = ap.parse_args(argv)
    if bool(args.expect_size) != bool(args.expect_size_vram):
        ap.error("--expect-size and --expect-size-vram go together")
    if args.cmd == "run" and args.split == "eval" and not args.expect_size_vram:
        ap.error("the eval needs the GPU residency reference: --expect-size and --expect-size-vram")
    return cmd_run(args, make_worker) if args.cmd == "run" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
