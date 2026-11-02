"""Driver for the shared-parts pipeline: shared -> evaluate -> oracle.

    # 1. plan + worker outputs + reviews + both signals, cached (Week 4)
    python3 run_pipeline.py shared   --split warmup --worker 4b --limit 1
    python3 run_pipeline.py shared   --split warmup --worker 4b
    # 2. thresholds from the warmup, then the protocols (Week 5)
    python3 run_pipeline.py evaluate --shared <warmup shared log> --calibrate
    python3 run_pipeline.py evaluate --shared <eval shared log> \\
        --threshold-file <threshold json>
    # 3. oracle bound; default sample is 50, eval D3 must pass --sample 200
    python3 run_pipeline.py oracle   --shared <eval shared log> --sample 200

Dry run, no models, no key, no benchmark (numbers are SIMULATED):
    python3 run_pipeline.py shared   --mock
    python3 run_pipeline.py evaluate --mock --shared <that log> --calibrate

Every row carries protocol / worker / supervisor / grading labels. Every
model response is cached (--cache-dir), so a rerun, a crash-resume, or the
oracle search never pays for the same call twice; a daily-quota stop keeps
all progress. The calls log (calls_<worker>_<split>.jsonl, fresh calls
only) feeds cost_model.py --calls-log with MEASURED tokens per call.

Live runs: workers need Ollama (laptop for 4b, bosgame for 27b); the
supervisor needs GEMINI_API_KEY. Live backends were tested in the
warmup shared runs, with log-probabilities returned per chunk.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader  # noqa: E402
import oracle  # noqa: E402
import pipeline_prompts  # noqa: E402
import protocols  # noqa: E402
import runlog  # noqa: E402
import shared_parts  # noqa: E402
import signals  # noqa: E402
from backends import CachedBackend, GeminiSupervisor, OllamaWorker  # noqa: E402
from cache import DiskCache  # noqa: E402

WORKERS = {  # model, num_ctx, num_predict (a job reply is a short JSON object)
    "4b": ("qwen3:4b", 2048, 300),
    "27b": ("qwen3.8:27b", 16384, 400),
}
SUPERVISOR_MODEL = "gemini-3.8-flash"
GRADING = "JSON-mode aggregation letter + rule-based fallback + hand-check queue"


class _Stop(Exception):
    pass


def _cap_hit(supervisor, args):
    """True if a fresh supervisor cap is reached. Counts fresh calls
    and fresh dollars only; cached calls cost 0 and do not count.
    Local worker calls are not supervisor calls and do not count."""
    if getattr(args, "max_calls", None) is not None and supervisor.fresh >= args.max_calls:
        return True
    if getattr(args, "max_usd", None) is not None and getattr(supervisor, "fresh_usd", 0.0) >= args.max_usd:
        return True
    return False


def _cap_msg(supervisor, args):
    return (f"CAP REACHED: fresh supervisor calls {supervisor.fresh}"
            f"{' / ' + str(args.max_calls) if args.max_calls is not None else ''}, "
            f"fresh $ {getattr(supervisor, 'fresh_usd', 0.0):.4f}"
            f"{' / ' + str(args.max_usd) if args.max_usd is not None else ''}. "
            "Progress is kept, no partial row is written, rerun with a higher "
            "cap only as a documented freeze change before the affected run.")


def build(args, need_worker):
    """-> (worker CachedBackend or None, supervisor CachedBackend, worker label)."""
    cache_dir = args.cache_dir or os.path.expanduser(
        "~/pipeline_cache_mock" if args.mock else "~/pipeline_cache")
    cache = DiskCache(cache_dir)
    calls_log = os.path.join(runlog.LOG_DIR, f"calls_{args.worker}_{args.split}"
                             + ("_mock" if args.mock else "") + ".jsonl")
    if args.mock:
        from mocks import MockSupervisor, MockWorker
        raw_w, raw_s = MockWorker(f"mock-{args.worker}"), MockSupervisor()
        label = f"MOCK {args.worker}"
    else:
        model, ctx, npred = WORKERS[args.worker]
        raw_w = OllamaWorker(model, ctx, npred) if need_worker else None
        from google import genai
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY is not set in this shell.")
        raw_s = GeminiSupervisor(args.supervisor_model, genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]))
        label = f"{model}, ctx {ctx}, think off"
    worker = CachedBackend(raw_w, cache, calls_log) if raw_w else None
    return worker, CachedBackend(raw_s, cache, calls_log), label


def _quota_exception():
    try:
        from cloud_only import DailyQuotaExhausted
        return DailyQuotaExhausted
    except Exception:  # noqa: BLE001  (no google-genai: mock/dry run)
        return _Stop


def cmd_shared(args):
    worker, supervisor, wlabel = build(args, need_worker=True)
    if args.mock:
        import mocks
        recs, chunk_fn = mocks.synthetic_records(), mocks.mock_chunker
    else:
        recs = data_loader.load_split(args.split, pad_n=args.pad_n)
        chunk_fn = lambda t: data_loader.chunk_text(t, args.chunk_chars)  # noqa: E731
    if args.limit:
        recs = recs[:args.limit]
    prefix = f"shared_{args.worker}_{args.split}" + ("_mock" if args.mock else "")
    cfg = runlog.config_id(
        worker=wlabel, supervisor=supervisor.model,
        chunk_chars=None if args.mock else args.chunk_chars,
        padding=None if args.mock else args.pad_n,
        agreement=not args.no_agreement, prompts=pipeline_prompts.fingerprint())
    path, done, _ = runlog.open_run(prefix, cfg, fresh=args.fresh,
                                    allow_legacy=False)
    cap_hit = False
    try:
        for i, rec in enumerate(recs):
            if (rec["patient"], rec["q_idx"]) in done:
                print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: done, skipping")
                continue
            if _cap_hit(supervisor, args):
                cap_hit = True
                break
            sp = shared_parts.run_shared(rec, worker, supervisor, chunk_fn,
                                         with_agreement=not args.no_agreement)
            sp.update({"config_id": cfg, "worker": wlabel,
                       "supervisor": supervisor.model, "split": args.split,
                       "grading": GRADING})
            runlog.log_row(path, sp)
            flagged = sum(1 for s in sp["steps"] if s["review"]["verdict"] == "redo")
            print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
                  f"K={len(sp['steps'])} chunks={sp['n_chunks']} redo={flagged} "
                  f"fresh calls w/s={worker.fresh}/{supervisor.fresh}", flush=True)
            if _cap_hit(supervisor, args):
                cap_hit = True
                break
    except _quota_exception() as e:
        print(f"\nDAILY QUOTA EXHAUSTED; cached progress is kept ({path}). "
              f"Rerun the same command after the reset.\n{str(e)[:200]}")
        sys.exit(2)
    if cap_hit:
        print(f"\n{_cap_msg(supervisor, args)} Progress is kept ({path}).")
        sys.exit(2)
    print(f"done. cache hits w/s: {worker.hits}/{supervisor.hits}")


def _read_shared(path):
    return runlog.read_rows(path)


def _threshold(args, rows):
    if args.threshold is not None:
        return args.threshold
    if args.threshold_file:
        with open(args.threshold_file, encoding="utf-8") as f:
            return json.load(f)["threshold"]
    return None


def cmd_evaluate(args):
    rows = _read_shared(args.shared)
    if args.calibrate:
        # Thresholds are set ONLY on the 20 warmup questions (study plan,
        # Week 5): calibrating on eval rows would leak the evaluation set
        # into the design before the pre-registration is frozen.
        splits = sorted({str(r.get("split")) for r in rows})
        if splits != ["warmup"]:
            sys.exit(f"refusing to calibrate on split(s) {splits}: thresholds are "
                     "set only on the warmup questions")
        scores = [s["logprob_score"] for r in rows for s in r["steps"]]
        thr = signals.calibrate_threshold(scores, args.flag_rate)
        out = os.path.splitext(args.shared)[0] + "_threshold.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"threshold": thr, "flag_rate": args.flag_rate,
                       "n_steps_scored": sum(1 for x in scores if x is not None),
                       "from": os.path.basename(args.shared),
                       "frozen": False}, f, indent=2)
        print(f"log-prob threshold {thr:.4f} (target flag rate {args.flag_rate:.0%}) -> {out}")
        return
    thr = _threshold(args, rows)
    if thr is None:
        print("note: no --threshold/--threshold-file given; skipping triggered_logprob")
    _, supervisor, _ = build(args, need_worker=False)
    path = runlog.new_log(f"protocols_{args.worker}_{args.split}"
                          + ("_mock" if args.mock else ""))
    names = protocols.PROTOCOLS
    agg = {n: {"n": 0, "correct": 0, "reviews": 0, "usd": 0.0, "latency": 0.0,
               "passes": 0, "flag": 0} for n in names}
    cap_hit = False
    try:
        for r in rows:
            if _cap_hit(supervisor, args):
                cap_hit = True
                break
            rec = {"question": r["question"], "options": r["options"]}
            for name in names:
                if name == "triggered_logprob" and thr is None:
                    continue
                row = protocols.run_protocol(rec, r, supervisor, name, thr)
                row.update({"worker": r["worker"], "supervisor": r["supervisor"],
                            "grading": GRADING, "threshold": thr})
                runlog.log_row(path, row)
                a = agg[name]
                a["n"] += 1
                a["correct"] += row["correct"]
                a["reviews"] += row["n_reviews"]
                a["usd"] += row["cloud_usd"]
                a["latency"] += row["latency_s"]
                a["passes"] += row["local_passes"]
                a["flag"] += row["needs_handcheck"]
                if _cap_hit(supervisor, args):
                    cap_hit = True
                    break
            if cap_hit:
                break
    except _quota_exception() as e:
        print(f"\nDAILY QUOTA EXHAUSTED; rerun after the reset (cached calls are free).\n{str(e)[:200]}")
        sys.exit(2)
    first = rows[0] if rows else {}
    print(f"worker: {first.get('worker')} | supervisor: {first.get('supervisor')} | "
          f"grading: {GRADING}")
    print(f"{'protocol':<22}{'n':>4}{'acc':>8}{'rev/q':>7}{'cloud $/q':>11}"
          f"{'latency s/q':>13}{'local passes/q':>16}{'handchk':>9}")
    for name, a in agg.items():
        if not a["n"]:
            continue
        n = a["n"]
        print(f"{name:<22}{n:>4}{a['correct']/n:>8.1%}{a['reviews']/n:>7.2f}"
              f"{a['usd']/n:>11.4f}{a['latency']/n:>13.1f}{a['passes']/n:>16.1f}"
              f"{a['flag']:>9}")
    print(f"rows -> {path}")
    if cap_hit:
        print(f"\n{_cap_msg(supervisor, args)} Progress is kept ({path}).")
        sys.exit(2)


def pick_sample(rows, n):
    """Fixed, reproducible sample: order by SHA-256 of the question id, take n.
    (Pre-registration item: record this rule and the resulting id list.)"""
    key = lambda r: hashlib.sha256(  # noqa: E731
        f"{r['patient']} q{r['q_idx']}".encode()).hexdigest()
    return sorted(rows, key=key)[:n]


def cmd_oracle(args):
    rows = pick_sample(_read_shared(args.shared), args.sample)
    _, supervisor, _ = build(args, need_worker=False)
    path = runlog.new_log(f"oracle_{args.worker}_{args.split}"
                          + ("_mock" if args.mock else ""))
    found = aggs = 0
    search_usd = bound_usd = 0.0
    cap_hit = False
    try:
        for r in rows:
            if _cap_hit(supervisor, args):
                cap_hit = True
                break
            rec = {"question": r["question"], "options": r["options"]}
            res = oracle.oracle_search(rec, r, supervisor, args.max_size)
            res.update({"worker": r["worker"], "supervisor": r["supervisor"],
                        "grading": GRADING, "protocol": "oracle (bound)"})
            runlog.log_row(path, res)
            found += res["found"]
            aggs += res["n_aggregations"]
            search_usd += res["search_agg_usd"]
            if res["found"]:
                bound_usd += res["bound_row"]["cloud_usd"]
            if _cap_hit(supervisor, args):
                cap_hit = True
                break
    except _quota_exception() as e:
        print(f"\nDAILY QUOTA EXHAUSTED; rerun after the reset.\n{str(e)[:200]}")
        sys.exit(2)
    n = len(rows)
    print(f"oracle on {n} questions | worker: {rows[0]['worker'] if rows else None} | "
          f"supervisor: {rows[0]['supervisor'] if rows else None}")
    print(f"bound accuracy {found}/{n} = {found/max(n,1):.1%} | "
          f"mean aggregations/q {aggs/max(n,1):.1f} | "
          f"mean bound cloud $/q (winners) {bound_usd/max(found,1):.4f} | "
          f"search aggregation $ total {search_usd:.4f} | rows -> {path}")
    if cap_hit:
        print(f"\n{_cap_msg(supervisor, args)} Progress is kept ({path}).")
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--worker", choices=list(WORKERS), default="4b")
        p.add_argument("--split", choices=["warmup", "eval"], default="warmup")
        p.add_argument("--mock", action="store_true",
                       help="dry run with simulated backends")
        p.add_argument("--cache-dir")
        p.add_argument("--supervisor-model", default=SUPERVISOR_MODEL)
        p.add_argument("--max-calls", type=int, default=None,
                       help="stop cleanly after this many fresh supervisor calls")
        p.add_argument("--max-usd", type=float, default=None,
                       help="stop cleanly after this many fresh supervisor dollars")

    s = sub.add_parser("shared")
    common(s)
    s.add_argument("--limit", type=int)
    s.add_argument("--fresh", action="store_true")
    s.add_argument("--no-agreement", action="store_true",
                   help="skip the 2 paraphrase passes (agreement signal)")
    s.add_argument("--chunk-chars", type=int, default=data_loader.CHUNK_CHARS)
    s.add_argument("--pad-n", type=int, default=2,
                   help="padding patients (frozen setting: 2)")
    e = sub.add_parser("evaluate")
    common(e)
    e.add_argument("--shared", required=True)
    e.add_argument("--calibrate", action="store_true")
    e.add_argument("--flag-rate", type=float, default=0.5)
    e.add_argument("--threshold", type=float)
    e.add_argument("--threshold-file")
    o = sub.add_parser("oracle")
    common(o)
    o.add_argument("--shared", required=True)
    o.add_argument("--sample", type=int, default=50)
    o.add_argument("--max-size", type=int, default=2)
    args = ap.parse_args()
    {"shared": cmd_shared, "evaluate": cmd_evaluate, "oracle": cmd_oracle}[args.cmd](args)


if __name__ == "__main__":
    main()
