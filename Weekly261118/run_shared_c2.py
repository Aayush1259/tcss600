"""The 4B shared parts with the guarded backends (code change C2, November 13).

Replaces `run_pipeline.py shared` for the 4B worker. It imports the frozen helper modules
(data_loader, pipeline_prompts, shared_parts, cache, runlog, backends, cloud_only, ollama_guard)
unchanged and never imports run_pipeline.py or eval_protocols.py, whose 4B worker is the
unguarded one at 2,048. The shared parts themselves are computed by the frozen
shared_parts.run_shared, with the agreement passes off (paused on November 5).

What it adds:
- the guarded worker (context 3,072) and the version-pinned supervisor from guarded_backend.py;
- a run id: the shared log is shared_4b_<split>_<run id>.jsonl and the calls log
  calls_4b_<split>_<run id>.jsonl, so a run never mixes with older logs;
- a config id that holds the SHA-256 of every module it runs, the Ollama version and the model
  digest; the frozen modules must have their pushed hashes, and a log is resumed only with the
  same config id;
- caps on fresh supervisor calls and dollars for the whole run, checked before every fresh call
  and seeded from the run's own calls log on every launch;
- a failed row for a question whose call ends in an error (one rerun on a later launch, then final
  and wrong), a halt row and exit code 3 for a guard stop, a refused request, a wrong model version
  or a cached reply without the marks; failure counts seeded from the log; the run stops after 3
  failures in a row or more than 10 failure and halt rows;
- per-question SHA-256 of the padded documents list and of the question with its options;
- a lock file holding the process id, so two GPU runs cannot start at once;
- counts only on screen: no accuracy, no answers.

Usage on the laptop, from ~/pipeline-4b-c2 (logs go to BASELINE_LOG_DIR):
    python run_shared_c2.py sizecheck --split eval
    python run_shared_c2.py run --split warmup --limit 1 --run-id trace1 \\
        --cache-dir ~/cache_c2_warmup --max-usd 0.10 --max-calls 30
Exit codes: 0 done, 2 cap reached, 3 halt, 4 too many failures, 5 refused to start.
"""
import argparse
import atexit
import hashlib
import json
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cache as cache_mod  # noqa: E402
import data_loader  # noqa: E402
import guarded_backend as gb  # noqa: E402
import ollama_guard  # noqa: E402
import pipeline_prompts  # noqa: E402
import runlog  # noqa: E402
import shared_parts  # noqa: E402

WORKER_MODEL, NUM_CTX, NUM_PREDICT = "qwen3:4b", 3072, 300
SUPERVISOR_MODEL = "gemini-3.8-flash"
PAD_N = 2
GRADING = "JSON-mode aggregation letter + rule-based fallback + hand-check queue"
MAX_FAILURES_IN_ROW, MAX_FAILURES_TOTAL = 3, 10

# SHA-256 of the frozen modules as pushed (Weekly261104; ollama_guard.py in Weekly261111).
FROZEN = {
    "backends": "6be750f73aee3de192616b9e418213d1622dc0a353e7746a9298ab3368c12856",
    "cache": "9ca436057d3b2f9c8d26a75c924de8dd4f31ce8e0bff81c0603dcd0a4ffeceb3",
    "cloud_only": "1e5cdd18628410ec73bbebeebe9ce6dc2666a533a89aa3ca5c0b09a2a4f81eac",
    "data_loader": "394b558e8647ba6eabfde4446f56a38b7b6c381999b0d802049c4cee69a68c59",
    "grader": "b784c5b1f2b8a035f72729fbc4f902bd07fdb79ba6fa0674d5a05f8eaaf6608f",
    "pipeline_prompts": "0ca01425a35509dd3efd52783ccb1a8bf68a11d3d1e9f80bafbfb85a13bfd1b3",
    "runlog": "22c2a6a695e7b6061788da3369a2fdee05b03c4f56ccb99b29b3de3cdbd883cf",
    "shared_parts": "3b19bc361bf8b6c9e4ccdd0bdb0736ca60d3fba1909f5c70204245f682e00367",
    "signals": "de2c93bfb7292af55b3299b285e3d1151d5354c89bd224685bbe143ea691b450",
    "ollama_guard": "3d2bbe020e2e84f12101b5c171ca880b7245fc7035619dceb8eecc1f1df8b68b",
}
NEW = ("guarded_backend", "run_shared_c2")


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def module_hashes():
    """SHA-256 of every module this run uses, read from the file each was actually imported from."""
    import cloud_only  # noqa: F401  (imported here so its file is the one in use)
    hashes = {}
    for name in list(FROZEN) + list(NEW):
        mod = sys.modules.get(name) or __import__(name)
        hashes[name] = sha256_of(mod.__file__)
    return hashes


def frozen_problems(hashes):
    return [f"{name}: {hashes[name][:8]}, pushed {want[:8]}" for name, want in FROZEN.items()
            if hashes.get(name) != want]


def runner_problems():
    return [name for name in ("run_pipeline", "eval_protocols") if name in sys.modules]


def pid_alive(pid):
    if os.name == "nt":  # on Windows signal 0 is Ctrl-C, so ask the kernel instead
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def take_lock(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                pid = int(f.read().strip() or 0)
        except ValueError:
            pid = 0
        if pid and pid != os.getpid() and pid_alive(pid):
            return f"another run holds the GPU lock {path} (process {pid})"
        os.remove(path)                      # the holder is gone: a stale lock
    try:                                     # create it atomically, so two launches cannot both take it
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return f"another run took the GPU lock {path} at the same moment"
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(path) and os.remove(path))
    return None


def input_hashes(rec):
    docs = json.dumps(rec["documents"], ensure_ascii=False)
    qo = json.dumps({"question": rec["question"], "options": rec["options"]}, sort_keys=True, ensure_ascii=False)
    return {"documents_sha256": hashlib.sha256(docs.encode("utf-8")).hexdigest(),
            "question_sha256": hashlib.sha256(qo.encode("utf-8")).hexdigest()}


def record_problems(rec):
    pads = rec.get("padding_patients") or []
    problems = []
    if rec.get("padding_n") != PAD_N:
        problems.append(f"padding_n {rec.get('padding_n')}")
    if len(pads) != PAD_N or len(set(pads)) != PAD_N or rec.get("patient") in pads:
        problems.append(f"padding patients {pads}")
    return problems


def scrub(message):
    key = os.environ.get("GEMINI_API_KEY")
    return (message.replace(key, "[key]") if key else message)[:300]


def log_state(rows):
    """Only "failed" rows count as attempts. A "halt" row means the run stopped for a cause that is
    fixed before relaunch (GPU lost, a mark check); it is set aside and never makes a question final."""
    done, failures = set(), {}
    for r in rows:
        key = (r.get("patient"), r.get("q_idx"))
        if r.get("status") == "ok":
            done.add(key)
        elif r.get("status") == "failed":
            failures[key] = failures.get(key, 0) + 1
    final = {k for k, n in failures.items() if n >= 2 and k not in done}
    in_row = 0
    for r in reversed(rows):
        if r.get("status") == "halt":
            continue
        if r.get("status") == "failed":
            in_row += 1
        else:
            break
    return done, failures, final, in_row, sum(failures.values())


def jobs_in(sp):
    return sum(len(s["out"]) + len(s["reviewed"] or []) for s in sp["steps"])


def load_records(split):
    return data_loader.load_split(split, pad_n=PAD_N)


def residency_from_log(path):
    """The GPU residency of the run's first logged worker call, or None."""
    if not os.path.exists(path):
        return None
    for r in runlog.read_rows(path):
        if r.get("kind") == "ollama" and r.get("model_size_vram") and not r.get("cached"):
            return {"model_size": r.get("model_size"), "model_size_vram": r.get("model_size_vram")}
    return None


def make_backends(args, cache, budget, paths):
    """The real guarded backends. The tests replace this function."""
    from google import genai
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set in this shell.")
    raw_w = gb.GuardedOllamaWorker(WORKER_MODEL, NUM_CTX, NUM_PREDICT)
    raw_s = gb.PinnedGeminiSupervisor(SUPERVISOR_MODEL, genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
                                      args.expect_version)
    return raw_w, raw_s


def cmd_sizecheck(args):
    """Offline: the largest job prompt per question, with a 300-character subtask, against the limit."""
    recs = load_records(args.split)
    if args.limit:
        recs = recs[:args.limit]
    over, sizes = [], []
    for rec in recs:
        chunks = data_loader.chunk_text("\n\n---\n\n".join(rec["documents"]), args.chunk_chars)
        longest = max(len(pipeline_prompts.job_prompt("x" * 300, rec["question"], c)) for c in chunks)
        est = longest / 3 + NUM_PREDICT
        sizes.append((longest, est, rec["patient"], rec["q_idx"], len(chunks)))
        if est >= NUM_CTX:
            over.append((rec["patient"], rec["q_idx"], longest, round(est)))
    sizes.sort(reverse=True)
    print(f"{len(recs)} questions; chunks per question {min(s[4] for s in sizes)} to {max(s[4] for s in sizes)}")
    print("largest job prompts (characters, characters / 3 + 300):")
    for longest, est, p, q, _ in sizes[:5]:
        print(f"  {p} q{q}: {longest} characters, estimate {est:.0f} of {NUM_CTX}")
    print(f"over the estimate: {len(over)} {over[:5]}")
    print("This is an estimate only; the guard measures every real prompt. No model was called.")
    return 0


def cmd_run(args):
    if runner_problems():
        print(f"refused: the frozen runner is imported ({runner_problems()})")
        return 5
    hashes = module_hashes()
    bad = frozen_problems(hashes)
    if bad:
        print("refused: a frozen module differs from the pushed one: " + "; ".join(bad))
        return 5
    recs = load_records(args.split)
    if args.limit:
        recs = recs[:args.limit]
    bad = [(r["patient"], r["q_idx"], record_problems(r)) for r in recs if record_problems(r)]
    if bad:
        print(f"refused: padding check failed before any call: {bad[:5]}")
        return 5

    log_dir = runlog.LOG_DIR
    paths = {"shared": os.path.join(log_dir, f"shared_4b_{args.split}_{args.run_id}.jsonl"),
             "calls": os.path.join(log_dir, f"calls_4b_{args.split}_{args.run_id}.jsonl"),
             "heartbeat": os.path.join(log_dir, f"heartbeat_{args.run_id}.json"),
             "lock": os.path.join(HERE, ".gpu.lock")}
    problem = take_lock(paths["lock"])
    if problem:
        print("refused: " + problem)
        return 5

    cache = cache_mod.DiskCache(os.path.expanduser(args.cache_dir))
    budget = gb.Budget.from_calls_log(paths["calls"], args.max_calls, args.max_usd)
    raw_w, raw_s = make_backends(args, cache, budget, paths)
    given = ({"model_size": args.expect_size, "model_size_vram": args.expect_size_vram}
             if args.expect_size_vram else None)
    logged = residency_from_log(paths["calls"])
    if given and logged and given != logged:
        print(f"refused: the GPU residency given {given} differs from this run's log {logged}")
        return 5
    raw_w.reference = given or logged
    print(f"GPU residency reference: {raw_w.reference or 'taken from the first worker call'}"
          f"{' (from the command line)' if given else ' (from this run' + chr(39) + 's log)' if logged else ''}",
          flush=True)
    worker = gb.GuardedCachedBackend(raw_w, cache, paths["calls"], budget, args.run_id,
                                     heartbeat=paths["heartbeat"])
    supervisor = gb.GuardedCachedBackend(raw_s, cache, paths["calls"], budget, args.run_id,
                                         expect_version=args.expect_version)
    wlabel = f"{WORKER_MODEL}, ctx {NUM_CTX}, think off, guarded {gb.GUARD_TAG}"
    cfg = runlog.config_id(worker=wlabel, supervisor=SUPERVISOR_MODEL, expect_version=args.expect_version,
                           chunk_chars=args.chunk_chars, padding=PAD_N, agreement=False,
                           prompts=pipeline_prompts.fingerprint(), code=hashes, ollama=raw_w.env)

    rows = runlog.read_rows(paths["shared"]) if os.path.exists(paths["shared"]) else []
    other = {r.get("config_id") for r in rows} - {cfg}
    if other:
        print(f"refused: {paths['shared']} holds rows of another configuration {sorted(map(str, other))}; "
              f"this one is {cfg}. A change of code or Ollama needs a new run id.")
        return 5
    done, failures, final, in_row, total = log_state(rows)
    print(f"{'resuming' if rows else 'logging to'} {paths['shared']} (config {cfg}; {len(done)} done, "
          f"{len(final)} final failures; caps for the whole run: {budget.calls} calls and "
          f"{budget.usd:.4f} dollars used of {args.max_calls} and {args.max_usd})", flush=True)

    chunk_fn = lambda t: data_loader.chunk_text(t, args.chunk_chars)  # noqa: E731
    # one id per launch on every row: after a stop and a relaunch, the check counts each launch's calls apart
    launch = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    base = {"config_id": cfg, "run_id": args.run_id, "launch": launch, "split": args.split, "worker": wlabel,
            "supervisor": SUPERVISOR_MODEL, "expect_version": args.expect_version, "grading": GRADING}
    for i, rec in enumerate(recs):
        key = (rec["patient"], rec["q_idx"])
        tag = f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}"
        if key in done:
            print(f"{tag}: done, skipping")
            continue
        if key in final:
            print(f"{tag}: failed twice, final (counts as wrong), skipping")
            continue
        context = {"patient": rec["patient"], "q_idx": rec["q_idx"], "attempt": failures.get(key, 0) + 1,
                   "launch": launch}
        worker.context = supervisor.context = context
        ident = {**base, "patient": rec["patient"], "q_idx": rec["q_idx"], **input_hashes(rec)}
        t0, w0, s0, h0 = time.time(), worker.fresh, supervisor.fresh, worker.hits
        try:
            sp = shared_parts.run_shared(rec, worker, supervisor, chunk_fn, with_agreement=False)
        except gb.CapReached as e:
            print(f"{tag}: CAP REACHED ({e}); no row written; progress is kept in the cache.")
            return 2
        except (gb.Halt, ollama_guard.GuardError, ollama_guard.Rejected) as e:
            runlog.log_row(paths["shared"], {**ident, "status": "halt", "attempt": context["attempt"],
                                             "error_type": type(e).__name__, "error_message": scrub(str(e))})
            print(f"{tag}: HALT ({type(e).__name__}: {scrub(str(e))}). Find the cause before relaunching.")
            return 3
        except Exception as e:  # noqa: BLE001  (any other error: a failed row, then the next question)
            failures[key] = failures.get(key, 0) + 1
            in_row += 1
            total += 1
            runlog.log_row(paths["shared"], {**ident, "status": "failed", "attempt": context["attempt"],
                                             "error_type": type(e).__name__, "error_message": scrub(str(e))})
            print(f"{tag}: FAILED ({type(e).__name__}: {scrub(str(e))}); failure {failures[key]} of 2", flush=True)
            if in_row >= MAX_FAILURES_IN_ROW or total > MAX_FAILURES_TOTAL:
                print(f"stopping: {in_row} failures in a row, {total} in this run")
                return 4
            continue
        in_row = 0
        n_jobs, wall = jobs_in(sp), time.time() - t0
        fresh_w = worker.fresh - w0
        sp.update({**ident, "status": "ok", "attempt": context["attempt"], "n_jobs": n_jobs,
                   "fresh_worker_calls": fresh_w, "cached_worker_calls": worker.hits - h0,
                   "fresh_supervisor_calls": supervisor.fresh - s0, "wall_time_s": round(wall, 1)})
        runlog.log_row(paths["shared"], sp)
        redo = sum(1 for s in sp["steps"] if s["review"]["verdict"] == "redo")
        per_job = f"{wall / fresh_w:.1f} s per fresh job" if fresh_w else "all jobs cached"
        print(f"{tag}: K={len(sp['steps'])} chunks={sp['n_chunks']} redo={redo} jobs={n_jobs} "
              f"fresh w/s={fresh_w}/{supervisor.fresh - s0} {per_job}; run dollars {budget.usd:.4f}", flush=True)
    print(f"done. this launch: fresh worker {worker.fresh}, fresh supervisor {supervisor.fresh}, "
          f"cached worker {worker.hits}, cached supervisor {supervisor.hits}; run dollars {budget.usd:.4f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="4B shared parts with the guarded backends (C2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sizecheck")
    s.add_argument("--split", choices=["warmup", "eval"], default="eval")
    s.add_argument("--limit", type=int)
    s.add_argument("--chunk-chars", type=int, default=data_loader.CHUNK_CHARS)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["warmup", "eval"], required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--run-id", required=True)
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--max-calls", type=int, required=True)
    r.add_argument("--expect-version", default=SUPERVISOR_MODEL)
    r.add_argument("--expect-size", type=int, help="the model size from /api/ps (bytes), with --expect-size-vram")
    r.add_argument("--expect-size-vram", type=int, help="the reference bytes in GPU memory (the eval uses warmup4's)")
    r.add_argument("--chunk-chars", type=int, default=data_loader.CHUNK_CHARS)
    args = ap.parse_args(argv)
    if args.cmd == "run" and bool(args.expect_size) != bool(args.expect_size_vram):
        ap.error("--expect-size and --expect-size-vram go together")
    if args.cmd == "run" and args.split == "eval" and not args.expect_size_vram:
        ap.error("the eval needs the GPU residency reference from the warmup: --expect-size and --expect-size-vram")
    return cmd_sizecheck(args) if args.cmd == "sizecheck" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
