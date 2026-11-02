"""Tally supervisor costs from a pipeline calls log, no new API calls.

A calls log (run_pipeline.py, CachedBackend) holds fresh calls only.
Cached calls are not logged again, so a resumed run does not double
count calls that were already paid for in an earlier session. Each row
has: component, model, prompt_tokens, completion_tokens,
thinking_tokens, wall_time_s, logged_at.

Supervisor cost rule, from runlog.py, verified Oct 15, 2026:
  gemini-3.8-flash, $0.75 per 1M prompt tokens,
  $3.75 per 1M output tokens, where output = completion + thinking.
  Thinking bills as output. Prices double Jan 1, 2027; the study ends
  Dec 11, 2026, so the Oct 2026 prices apply here.

Counting rules, stated so the tally can be checked:
  Supervisor rows are rows with model "gemini-3.8-flash". Worker rows
  (local model, for example qwen3.8:27b) cost $0 in cloud terms and
  are reported for token and wall time reference only, never added
  to the supervisor cost.
  Component counts are reported per component (plan, review, worker,
  aggregate, oracle_aggregate, and any other value found), so an
  unexpected component is visible instead of being hidden in a total.
  Rows with missing token fields are counted separately and treated
  as 0 tokens in sums, so a malformed row cannot silently inflate
  the total.
  logged_at first to last is the calendar span of the log file. That
  span includes dead periods between resume sessions, so it is NOT
  run duration. Active time is the sum of wall_time_s, because calls
  in the shared run are sequential. Report both, labeled, and do not
  present the calendar span as run time.

This tally is a log-estimated cost, at list price, not a console
billed figure. Weekly reports carry both, labeled, per the standing
spend rule. Console billed figures come from the Cloud console,
Billing, Reports and Transactions, and will lag the log total until
recent calls post.

Usage:
  python3 calls_tally.py /home/localai/baseline_logs/calls_27b_warmup.jsonl

Stdlib only. Reads the log file only, writes nothing.
"""

import json
import sys
from collections import Counter, defaultdict

FLASH_IN_PER_1M = 0.75
FLASH_OUT_PER_1M = 3.75
SUPERVISOR_MODEL = "gemini-3.8-flash"


def supervisor_cost_usd(prompt_tokens, completion_tokens, thinking_tokens):
    pt = prompt_tokens or 0
    out = (completion_tokens or 0) + (thinking_tokens or 0)
    return pt / 1e6 * FLASH_IN_PER_1M + out / 1e6 * FLASH_OUT_PER_1M


def main():
    if len(sys.argv) != 2:
        print("usage: python3 calls_tally.py <calls log path>")
        sys.exit(2)
    path = sys.argv[1]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    comp_count = Counter()
    comp_sup_cost = defaultdict(float)
    comp_sup_rows = Counter()
    comp_wall = defaultdict(float)
    models = Counter()
    missing_tokens = 0
    sup_rows = 0
    sup_in = 0
    sup_out = 0
    sup_think = 0
    sup_cost = 0.0
    sup_wall = 0.0
    work_rows = 0
    work_in = 0
    work_out = 0
    work_wall = 0.0
    logged = []
    for r in rows:
        comp = str(r.get("component", "?"))
        model = str(r.get("model", "?"))
        comp_count[comp] += 1
        models[model] += 1
        wall = r.get("wall_time_s") or 0.0
        comp_wall[comp] += wall
        if r.get("logged_at"):
            logged.append(str(r["logged_at"]))
        pt = r.get("prompt_tokens")
        ct = r.get("completion_tokens")
        tt = r.get("thinking_tokens")
        if pt is None or ct is None or tt is None:
            missing_tokens += 1
        pt = pt or 0
        ct = ct or 0
        tt = tt or 0
        if model == SUPERVISOR_MODEL:
            sup_rows += 1
            comp_sup_rows[comp] += 1
            sup_in += pt
            sup_out += ct
            sup_think += tt
            cost = supervisor_cost_usd(pt, ct, tt)
            sup_cost += cost
            comp_sup_cost[comp] += cost
            sup_wall += wall
        else:
            work_rows += 1
            work_in += pt
            work_out += ct
            work_wall += wall

    print(f"Calls tally: {total} rows in {path}")
    print(f"Models: {dict(models)}")
    print(f"Rows per component: {dict(comp_count)}")
    print(f"Wall time per component (seconds): "
          f"{ {k: round(v, 1) for k, v in sorted(comp_wall.items())} }")
    print(f"Rows with a missing token field: {missing_tokens} "
          f"(counted as 0 tokens in sums)")
    print()
    print(f"Supervisor rows (model {SUPERVISOR_MODEL}): {sup_rows}")
    print(f"Supervisor rows per component: {dict(comp_sup_rows)}")
    print(f"Supervisor tokens: prompt {sup_in}, completion {sup_out}, "
          f"thinking {sup_think}")
    print(f"Supervisor cost at list price: prompt "
          f"${sup_in / 1e6 * FLASH_IN_PER_1M:.4f} + output "
          f"${(sup_out + sup_think) / 1e6 * FLASH_OUT_PER_1M:.4f} = "
          f"${sup_cost:.4f}")
    print(f"Supervisor cost per component: "
          f"{ {k: round(v, 4) for k, v in sorted(comp_sup_cost.items())} }")
    print(f"Supervisor wall time total (seconds): {sup_wall:.1f}")
    print()
    print(f"Worker rows (local, cloud cost $0): {work_rows}")
    print(f"Worker tokens: prompt {work_in}, completion {work_out}")
    print(f"Worker wall time total (seconds): {work_wall:.1f}")
    print()
    if logged:
        print(f"logged_at first: {min(logged)}, last: {max(logged)}")
        print("Note: first to last is the calendar span of the log file, "
              "including any dead periods between resume sessions. "
              "It is not run duration. Active time is the wall time "
              "sum above, because shared-run calls are sequential.")
    else:
        print("No logged_at values found.")
    print("Note: this is a log-estimated cost at list price, not a "
          "console billed figure. Weekly reports carry both, labeled.")


if __name__ == "__main__":
    main()
