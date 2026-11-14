"""JSONL run logger for the Week 3 baselines.

Every row carries the full labeling Cai requires (protocol, worker,
supervisor, grading method) plus timing, tokens, and cost.

Oct 17 lesson, enforced here: the pilot saved only final_answer[:500],
which destroyed verifiability (patient_04 q1 could not be hand-checked).
This logger saves FULL prompts and FULL answers. Disk is cheap; reruns
are not.

Cost math uses gemini-3.8-flash intro pricing verified Oct 15, 2026:
$0.75 per 1M input tokens, $3.75 per 1M output tokens through Dec 31,
2026. Thinking tokens bill as output and are recorded separately.

Oct 18: resume helpers (latest_log_for, done_ids) live here instead of
being copy-pasted into each script. A run only resumes a log whose rows
were written with the same config_id (model, ctx, token budget, prompt
wording), so rows from two different prompts never mix in one log. Rows
written before config_id existed count as compatible, so the in-progress
Oct 17-18 logs can still be resumed. BASELINE_LOG_DIR overrides the log
folder (used by the tests).
"""
import glob
import hashlib
import json
import os
import time

HOME = os.path.expanduser("~")
LOG_DIR = os.environ.get("BASELINE_LOG_DIR") or os.path.join(HOME, "baseline_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Verified 2026-10-15. Doubles Jan 1, 2027; the study ends Dec 11.
FLASH_IN_PER_1M = 0.75
FLASH_OUT_PER_1M = 3.75


def new_log(name):
    """Returns the path of a fresh timestamped JSONL log file."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{name}_{ts}.jsonl")
    return path


def config_id(**settings):
    """Short stable fingerprint of the settings that define a run.

    Pass everything that changes the numbers: model, ctx, num_predict,
    think, and a prompt fingerprint (see prompt_fingerprint).
    """
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def prompt_fingerprint(build_fn, *extra_args):
    """Hash of a prompt builder's wording, via a dummy record.

    Catches any edit to the template (a reworded instruction changes the
    id; a code comment does not). `build_fn(rec, *extra_args)` is called
    with a record made of distinctive placeholder strings.
    """
    rec = {
        "question": "QQQ",
        "options": {L: f"OPT{L}" for L in "ABCDE"},
        "documents": ["DOC1", "DOC2"],
    }
    return hashlib.sha256(build_fn(rec, *extra_args).encode("utf-8")).hexdigest()[:10]


def supervisor_cost_usd(prompt_tokens, completion_tokens, thinking_tokens=0):
    pt = prompt_tokens or 0
    out = (completion_tokens or 0) + (thinking_tokens or 0)
    return pt / 1e6 * FLASH_IN_PER_1M + out / 1e6 * FLASH_OUT_PER_1M


def log_row(path, row):
    """Appends one JSON row. `row` must include the labeling fields:
    protocol, worker, supervisor, grading. Full 'prompt' and 'answer'
    text are stored, never truncated."""
    row = dict(row)
    row["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_rows(path):
    """All parseable rows. A half-written last line (crash) is skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def latest_log_for(prefix, cfg_id=None, allow_legacy=True):
    """Newest non-regraded log named `<prefix>_*.jsonl` that can be resumed.

    With cfg_id given, a log is resumable only if every row has that
    config_id, or (allow_legacy) none at all. Returns None if no log fits.
    """
    cands = sorted(glob.glob(os.path.join(LOG_DIR, f"{prefix}_*.jsonl")))
    for path in reversed([c for c in cands if "regraded" not in c]):
        if cfg_id is None:
            return path
        ids = {r.get("config_id") for r in read_rows(path)}
        if ids <= ({None, cfg_id} if allow_legacy else {cfg_id}):
            return path
    return None


def done_ids(path):
    """(patient, q_idx) pairs already logged in `path`."""
    if not path:
        return set()
    return {(r["patient"], r["q_idx"]) for r in read_rows(path)
            if "patient" in r and "q_idx" in r}


def open_run(prefix, cfg_id, fresh=False, allow_legacy=True):
    """Pick the log to write to. Returns (path, done_ids, resumed).

    Resumes the newest compatible log unless fresh=True. Prints which.
    allow_legacy=False refuses logs written before config_id existed
    (use it where an old log may hold rows from a different setting).
    """
    path = None if fresh else latest_log_for(prefix, cfg_id, allow_legacy)
    done = done_ids(path)
    if path and done:
        print(f"resuming {path} ({len(done)} already done, config {cfg_id})")
        return path, done, True
    path = new_log(prefix)
    print(f"logging to {path} (config {cfg_id})")
    return path, set(), False


def summarize(path):
    """Prints accuracy and cost totals for a finished log."""
    n = correct = flagged = 0
    cost = 0.0
    for r in read_rows(path):
        n += 1
        correct += 1 if r.get("match") else 0
        flagged += 1 if r.get("needs_handcheck") else 0
        cost += r.get("est_cost_usd") or 0.0
    print(f"{path}: {correct}/{n} correct, {flagged} need hand-check, "
          f"est cost ${cost:.4f}")
    return {"n": n, "correct": correct, "flagged": flagged, "cost": cost}
