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
"""
import json
import os
import time

HOME = os.path.expanduser("~")
LOG_DIR = os.path.join(HOME, "baseline_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Verified 2026-10-15. Doubles Jan 1, 2027; the study ends Dec 11.
FLASH_IN_PER_1M = 0.75
FLASH_OUT_PER_1M = 3.75


def new_log(name):
    """Returns the path of a fresh timestamped JSONL log file."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{name}_{ts}.jsonl")
    return path


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
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def summarize(path):
    """Prints accuracy and cost totals for a finished log."""
    n = correct = flagged = 0
    cost = 0.0
    for line in open(path):
        r = json.loads(line)
        n += 1
        correct += 1 if r.get("match") else 0
        flagged += 1 if r.get("needs_handcheck") else 0
        cost += r.get("est_cost_usd") or 0.0
    print(f"{path}: {correct}/{n} correct, {flagged} need hand-check, "
          f"est cost ${cost:.4f}")
    return {"n": n, "correct": correct, "flagged": flagged, "cost": cost}
