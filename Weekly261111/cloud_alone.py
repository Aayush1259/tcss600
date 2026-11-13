"""Cloud alone on the padded record, with the model version recorded and pinned.

This is the cloud-only arm of the study for the evaluation set. The frozen cloud_only.py (Weekly261104)
stays unchanged: I import its prompt (build_prompt), its call code with retries and the call cap
(ask_supervisor) and its settings, so every request is the one the frozen script sends. What this script
adds, decided on Nov 12 before any evaluation call of this arm:

1. Padding is fixed at 2, the frozen setting. There is no padding flag, and every record is checked before
   any call: padding_n is 2, two distinct padding patients, neither of them the question's own patient.
2. The model version the API reports is logged on every row. A missing version, a version other than
   --expect-version, or a change within the run stops the run before that row is written.
3. A question whose call raises an error gets a failure row (failed true, the error type, the first 200
   characters of the message) and the run goes on. On the next launch it is asked once more; after a
   second failure it is final and counts as wrong. The run stops after 3 failures in a row, or after more
   than 10 failure rows in total (5 percent of 200). An empty or unreadable reply is not a failure: it is
   graded as the frozen rule says (wrong, in the hand-check queue).
4. The caps cover the whole run: on resume, the dollars and API attempts already in the log count toward
   --max-usd and --max-calls.
5. Its own log name (cloud_alone_{split}_pad2_json), so it never resumes a log of the frozen script. A log
   is resumed only if every answered row has the expected model version, every row has padding 2, and
   every row was written by the same code (same SHA-256 of this script and the four frozen modules).
6. Every row records those SHA-256 hashes, the google-genai and Python versions, and the machine name.
7. --dry-run builds and checks every prompt, with no API call and no key. With --ref-log, it compares each
   prompt byte for byte with the prompt logged for the same question in an earlier run, and projects the
   cost from that log's tokens.
8. --check LOG prints the counts of a log (rows, questions, failures, versions, padding, code, attempts,
   dollars) and no accuracy.
9. 2 seconds between calls. The frozen script waits 15 seconds, which was for the free tier; the account
   is prepay now, and a 429 still goes to the frozen retry loop. The pause does not change any answer.

Labels on every row: protocol cloud-only | worker none | supervisor gemini-3.8-flash, thinking low |
grading json + hand-check queue | script cloud_alone.py.

Usage, from a folder that also holds cloud_only.py, data_loader.py, grader.py and runlog.py:
    python3 cloud_alone.py --split warmup --dry-run --ref-log OLD_LOG
        # all 20 warmup prompts, compared with the logged prompts of the Oct 24 run
    python3 cloud_alone.py --split eval --dry-run --ref-log OLD_LOG
        # all 200 eval prompts checked; cost projected from the old log's tokens
    python3 cloud_alone.py --split warmup --limit 1 --max-calls 3 --max-usd 0.10
        # one question
    python3 cloud_alone.py --check LOG
Needs GEMINI_API_KEY, except with --dry-run and --check.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import platform
import sys
import time

import cloud_only
import data_loader
import grader
import runlog

PROTOCOL = "cloud-only"
SCRIPT = "cloud_alone.py"
PAD_N = 2
PACE_S = 2
MAX_FAILURES_IN_ROW = 3
MAX_FAILURES_TOTAL = 10
FROZEN_MODULES = (cloud_only, data_loader, grader, runlog)


class VersionRecorder:
    """Passes calls to the real client and keeps the last response, so the model version it reports can be logged."""

    def __init__(self, client):
        self._client = client
        self.last = None
        self.models = self

    def generate_content(self, **kwargs):
        self.last = self._client.models.generate_content(**kwargs)
        return self.last


def make_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    return cloud_only.genai.Client(api_key=key)


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_hashes():
    hashes = {SCRIPT: sha256_of(os.path.abspath(__file__))}
    for module in FROZEN_MODULES:
        hashes[os.path.basename(module.__file__)] = sha256_of(module.__file__)
    return hashes


def environment():
    return {
        "google_genai": getattr(cloud_only.genai, "__version__", None),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def log_prefix(split):
    return f"cloud_alone_{split}_pad{PAD_N}_json"


def latest_log_for(split):
    cands = sorted(glob.glob(os.path.join(runlog.LOG_DIR, log_prefix(split) + "_*.jsonl")))
    return cands[-1] if cands else None


def record_problems(rec):
    """Empty list if the record has the frozen padding; otherwise what is wrong with it."""
    pads = rec.get("padding_patients") or []
    problems = []
    if rec.get("padding_n") != PAD_N:
        problems.append(f"padding_n is {rec.get('padding_n')}")
    if len(pads) != PAD_N or len(set(pads)) != PAD_N:
        problems.append(f"padding patients {pads}")
    if rec.get("patient") in pads:
        problems.append("the question's own patient is used as padding")
    return problems


def log_state(rows):
    """Questions answered, failure rows per question, and questions that failed twice (final)."""
    done = set()
    failures = collections.Counter()
    for r in rows:
        key = (r.get("patient"), r.get("q_idx"))
        if r.get("failed"):
            failures[key] += 1
        else:
            done.add(key)
    final_failed = {k for k, n in failures.items() if n >= 2 and k not in done}
    return done, failures, final_failed


def resume_problems(rows, hashes, expect_version):
    problems = []
    answered = [r for r in rows if not r.get("failed")]
    versions = {r.get("supervisor_model_version") for r in answered}
    if None in versions:
        problems.append("an answered row has no model version")
    if len(versions) > 1:
        problems.append(f"more than one model version: {sorted(map(str, versions))}")
    other = versions - {expect_version, None}
    if other:
        problems.append(f"model version {sorted(other)} is not the expected {expect_version}")
    if any(r.get("padding_n") != PAD_N for r in rows):
        problems.append(f"a row without padding {PAD_N}")
    if any(r.get("code_sha256") != hashes for r in rows):
        problems.append("rows written by different code (the SHA-256 of a script or module differs)")
    return problems


def scrub(message):
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        message = message.replace(key, "[key]")
    return message[:200]


def dry_run(recs, ref_log):
    ref = {}
    if ref_log:
        for r in runlog.read_rows(ref_log):
            if r.get("prompt") is not None and not r.get("failed"):
                ref[(r.get("patient"), r.get("q_idx"))] = r
    same = differ = 0
    lengths = []
    for i, rec in enumerate(recs):
        prompt = cloud_only.build_prompt(rec)
        lengths.append(len(prompt))
        note = ""
        old = ref.get((rec["patient"], rec["q_idx"]))
        if old is not None:
            if old["prompt"] == prompt:
                same += 1
                note = ", same prompt as the reference log"
            else:
                differ += 1
                note = ", DIFFERENT from the reference log"
        print(f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: padding {rec['padding_patients']}, "
              f"prompt {len(prompt)} characters{note}")
    print(f"\n{len(recs)} prompts built; every record has padding {PAD_N}, two distinct padding patients, "
          f"and never its own patient as padding. No API call was made.")
    if lengths:
        print(f"Prompt characters: largest {max(lengths)}, smallest {min(lengths)}, total {sum(lengths)}.")
    if ref:
        print(f"Reference log {ref_log}: {same + differ} of these questions are in it; "
              f"{same} identical prompts, {differ} different.")
        rows = list(ref.values())
        chars = sum(len(r["prompt"]) for r in rows)
        tokens = sum(r.get("supervisor_prompt_tokens") or 0 for r in rows)
        out_tokens = sum((r.get("supervisor_completion_tokens") or 0) + (r.get("supervisor_thinking_tokens") or 0)
                         for r in rows)
        if tokens:
            chars_per_token = chars / tokens
            mean_out_usd = out_tokens / len(rows) / 1e6 * runlog.FLASH_OUT_PER_1M
            projected = sum(n / chars_per_token / 1e6 * runlog.FLASH_IN_PER_1M + mean_out_usd for n in lengths)
            print(f"From the reference log: {chars_per_token:.2f} characters per prompt token, "
                  f"{mean_out_usd:.4f} dollars of output per question. Projected cost of these {len(recs)} "
                  f"questions: {projected:.2f} dollars at list price.")
    return 1 if differ else 0


def check_log(path):
    rows = runlog.read_rows(path)
    answered = [r for r in rows if not r.get("failed")]
    done, failures, final_failed = log_state(rows)
    waiting = set(failures) - done - final_failed
    answered_twice = [k for k, n in collections.Counter((r.get("patient"), r.get("q_idx")) for r in answered).items()
                      if n > 1]
    print(path)
    print(f"rows {len(rows)}: answered {len(answered)}, failure rows {len(rows) - len(answered)}")
    print(f"questions: {len({(r.get('patient'), r.get('q_idx')) for r in rows})} in the log, {len(done)} answered, "
          f"{len(final_failed)} failed twice (final, counted wrong), {len(waiting)} waiting for their one rerun, "
          f"{len(answered_twice)} answered more than once")
    print(f"patients: {len({r.get('patient') for r in rows})}")
    print(f"splits: {dict(collections.Counter(r.get('split') for r in rows))}")
    print(f"model versions (answered rows): {dict(collections.Counter(r.get('supervisor_model_version') for r in answered))}")
    print(f"padding_n: {dict(collections.Counter(r.get('padding_n') for r in rows))}; "
          f"rows with a bad padding list: {sum(1 for r in rows if record_problems(r))}")
    print(f"code versions: {len({json.dumps(r.get('code_sha256'), sort_keys=True) for r in rows})}; "
          f"environments: {len({json.dumps(r.get('environment'), sort_keys=True) for r in rows})}")
    print(f"unreadable answers (hand-check): {sum(1 for r in answered if r.get('needs_handcheck'))}; "
          f"empty replies: {sum(1 for r in answered if r.get('empty_reply'))}")
    print(f"API attempts: {sum(r.get('attempts') or 0 for r in rows)}; "
          f"estimated cost: {sum(r.get('est_cost_usd') or 0.0 for r in rows):.4f} dollars")
    return 0


def run(recs, args):
    hashes = code_hashes()
    env = environment()
    log_path = None if args.fresh else latest_log_for(args.split)
    rows = runlog.read_rows(log_path) if log_path else []
    if rows:
        problems = resume_problems(rows, hashes, args.expect_version)
        if problems:
            sys.exit(f"will not resume {log_path}: " + "; ".join(problems))
        print(f"resuming {log_path} ({len(rows)} rows already)")
    else:
        log_path = runlog.new_log(log_prefix(args.split))
        print(f"logging to {log_path}")

    done, failures, final_failed = log_state(rows)
    total_failures = sum(failures.values())
    spent = sum(r.get("est_cost_usd") or 0.0 for r in rows)
    cloud_only.ATTEMPTS = sum(r.get("attempts") or 0 for r in rows)
    cloud_only.MAX_CALLS = args.max_calls
    print(f"caps for the whole run: max-calls={args.max_calls} max-usd={args.max_usd}; already used: "
          f"{cloud_only.ATTEMPTS} attempts, {spent:.4f} dollars; expected model version {args.expect_version}",
          flush=True)

    client = VersionRecorder(make_client())
    in_row = 0
    for i, rec in enumerate(recs):
        key = (rec["patient"], rec["q_idx"])
        tag = f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}"
        if key in done:
            print(f"{tag}: already done, skipping", flush=True)
            continue
        if key in final_failed:
            print(f"{tag}: failed twice, final (counts as wrong), skipping", flush=True)
            continue
        if args.max_usd is not None and spent >= args.max_usd:
            print(f"budget cap reached: {spent:.4f} dollars for the whole run; stopping. "
                  f"Relaunch with a higher cap to go on.", flush=True)
            break
        prompt = cloud_only.build_prompt(rec)
        base = {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": PROTOCOL,
            "worker": None,
            "supervisor": f"{cloud_only.SUPERVISOR_MODEL}, thinking {cloud_only.THINKING_LEVEL}",
            "supervisor_thinking_level": cloud_only.THINKING_LEVEL,
            "grading": "json + hand-check queue",
            "output_format": "json",
            "padding_n": rec["padding_n"],
            "padding_patients": rec["padding_patients"],
            "script": SCRIPT,
            "code_sha256": hashes,
            "environment": env,
            "gold": rec["gold"],
        }
        client.last = None
        before = cloud_only.ATTEMPTS
        try:
            out = cloud_only.ask_supervisor(client, prompt)
        except cloud_only.BudgetStop as e:
            print(f"budget cap reached: {e}; stopping. Relaunch with a higher cap to go on.", flush=True)
            break
        except Exception as e:  # noqa: BLE001  (any error: a failure row, then the run goes on)
            failures[key] += 1
            total_failures += 1
            in_row += 1
            message = scrub(str(e))
            runlog.log_row(log_path, {
                **base,
                "failed": True,
                "failure_number": failures[key],
                "error_type": type(e).__name__,
                "error_message": message,
                "supervisor_model_version": None,
                "predicted": None,
                "match": False,
                "needs_handcheck": False,
                "attempts": cloud_only.ATTEMPTS - before,
                "est_cost_usd": 0.0,
                "prompt": prompt,
                "answer": None,
            })
            print(f"{tag}: FAILED ({type(e).__name__}: {message}); failure {failures[key]} of 2 for this question",
                  flush=True)
            if in_row >= MAX_FAILURES_IN_ROW:
                sys.exit(f"{MAX_FAILURES_IN_ROW} failures in a row; stopping to find the cause")
            if total_failures > MAX_FAILURES_TOTAL:
                sys.exit(f"more than {MAX_FAILURES_TOTAL} failure rows in this log; stopping to find the cause")
            time.sleep(PACE_S)
            continue

        version = getattr(client.last, "model_version", None)
        if version is None:
            sys.exit(f"{tag}: the API reported no model version; that answer was not logged")
        if version != args.expect_version:
            sys.exit(f"{tag}: model version {version}, expected {args.expect_version}; that answer was not logged")
        in_row = 0
        g = grader.grade_json(out["text"], rec["gold"], rec["options"])
        cost = runlog.supervisor_cost_usd(out["prompt_tokens"], out["completion_tokens"], out["thinking_tokens"])
        spent += cost
        runlog.log_row(log_path, {
            **base,
            "failed": False,
            "supervisor_model_version": version,
            "predicted": g["letter"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "grading_method": g["method"],
            "empty_reply": out["text"] is None,
            "wall_time_s": out["wall_time_s"],
            "supervisor_prompt_tokens": out["prompt_tokens"],
            "supervisor_completion_tokens": out["completion_tokens"],
            "supervisor_thinking_tokens": out["thinking_tokens"],
            "attempts": cloud_only.ATTEMPTS - before,
            "est_cost_usd": round(cost, 6),
            "prompt": prompt,
            "answer": out["text"],
        })
        print(f"{tag}: pred={g['letter']} gold={rec['gold']} match={g['match']} "
              f"tok={out['prompt_tokens']}/{out['completion_tokens']}(+{out['thinking_tokens']} think) "
              f"${cost:.4f} (run total ${spent:.4f}, {cloud_only.ATTEMPTS} attempts) version {version}", flush=True)
        time.sleep(PACE_S)

    runlog.summarize(log_path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cloud alone on the padded record (padding 2), model version pinned.")
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None, help="API attempts for the whole run, retries included")
    ap.add_argument("--max-usd", type=float, default=None, help="estimated dollars for the whole run")
    ap.add_argument("--expect-version", default=cloud_only.SUPERVISOR_MODEL,
                    help="the model version the API must report (default: %(default)s)")
    ap.add_argument("--fresh", action="store_true", help="start a new log instead of resuming the latest")
    ap.add_argument("--dry-run", action="store_true", help="build and check every prompt; no API call, no key")
    ap.add_argument("--ref-log", help="with --dry-run: a log whose prompts and tokens are compared")
    ap.add_argument("--check", metavar="LOG", help="print the counts of a log and exit")
    args = ap.parse_args(argv)

    if args.check:
        return check_log(args.check)
    if args.ref_log and not args.dry_run:
        sys.exit("--ref-log is used only with --dry-run")

    recs = data_loader.load_split(args.split, pad_n=PAD_N)
    if args.limit:
        recs = recs[:args.limit]
    bad = [(r["patient"], r["q_idx"], record_problems(r)) for r in recs if record_problems(r)]
    if bad:
        sys.exit(f"padding check failed before any call: {bad[:5]}")
    if args.dry_run:
        return dry_run(recs, args.ref_log)
    return run(recs, args)


if __name__ == "__main__":
    sys.exit(main())
