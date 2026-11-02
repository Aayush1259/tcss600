"""Regrade a baseline JSONL with the current grader, using the saved answers.

The log stores full prompts and answers, so grader improvements never
require rerunning the model. Options come from the benchmark via
patient/q_idx; on a machine without the benchmark, rows that saved their
prompt (27B and cloud logs) fall back to the options parsed from it.

Machine: bosgame or laptop (needs the benchmark + ID files for 4B logs).

Usage (from the harness folder):
    ~/is-env/bin/python regrade.py /home/localai/baseline_logs/<file>.jsonl
    python3 regrade.py ~/baseline_logs/local_only_4b_chunked_warmup_<ts>.jsonl
    python3 regrade.py <file>.jsonl --out-dir <folder> --bench <benchmark_v5.json>

Writes <file>_regraded.jsonl (next to the source, or in --out-dir) and
prints a before/after summary plus every row whose verdict changed, so a
regrade is never a silent change.

4B rows (chunked) are regraded from their saved per-chunk votes (majority,
ties broken by the earliest chunk). Their 'answer' field is the text
"per-chunk votes: [...]", which has no letter; running the text grader on
it turns every 4B row into a hand-check (0/20 correct, 20/20 flagged on
the Oct 18 warmup log). 4B rows saved no prompt, so they need the benchmark.

This file depends only on data_loader.load_benchmark / get_question and
grader.grade / matches, so it runs unchanged with grader v2.1 and v3.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader  # noqa: E402
import grader  # noqa: E402


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


def options_from_prompt(prompt):
    """{'A': text, ...} from the '(A) text' lines of a saved prompt.

    The 27B and cloud prompts put the question and options before the
    records, so lines inside the records are not read.
    """
    head = re.split(r"\n\nAnswer with|\n\nMedical records:", prompt or "",
                    maxsplit=1)[0]
    opts = {}
    for letter in "ABCDE":
        m = re.search(rf"^\({letter}\) (.*)$", head, re.M)
        opts[letter] = m.group(1).strip() if m else ""
    return opts


def majority_vote(votes):
    """Majority over per-chunk letters; ties go to the letter whose first
    vote came earliest (the frozen 4B design). None if nobody voted."""
    counts, first = Counter(), {}
    for i, v in enumerate(votes):
        if isinstance(v, str) and len(v) == 1 and v in "ABCDE":
            counts[v] += 1
            first.setdefault(v, i)
    if not counts:
        return None
    best = max(counts.values())
    return min((L for L, n in counts.items() if n == best), key=first.__getitem__)


def regrade_row(row, options):
    """Returns (letter, match, needs_handcheck, label) for one logged row."""
    if row.get("chunk_votes") is not None:
        letter = majority_vote(row["chunk_votes"])
        label = "majority-vote over saved chunk_votes + hand-check queue [regraded]"
        return (letter, grader.matches(letter, row["gold"], options),
                letter is None, label)
    g = grader.grade(row.get("answer") or "", row["gold"], options)
    label = f"rule-based ({g['method']}) + hand-check queue [regraded]"
    return g["letter"], g["match"], g["needs_handcheck"], label


def options_for(row, bench):
    """Benchmark options if the benchmark is loaded, else parsed from the prompt."""
    if bench is not None:
        return data_loader.get_question(bench, row["patient"], row["q_idx"])["options"]
    if row.get("prompt"):
        return options_from_prompt(row["prompt"])
    raise SystemExit(f"{row['patient']} q{row['q_idx']}: no benchmark and no saved "
                     "prompt, so the options are unknown. Pass --bench.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--bench", default=data_loader.BENCH_PATH)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = args.src
    bench = (data_loader.load_benchmark(args.bench)
             if os.path.exists(args.bench) else None)
    rows = read_rows(src)
    before_hit = sum(1 for r in rows if r.get("match"))
    before_flag = sum(1 for r in rows if r.get("needs_handcheck"))
    changed = []
    for r in rows:
        old = (r.get("predicted"), r.get("match"))
        letter, match, flag, label = regrade_row(r, options_for(r, bench))
        r["predicted"], r["match"] = letter, match
        r["needs_handcheck"], r["grading"] = flag, label
        if (letter, match) != old:
            changed.append((r["patient"], r["q_idx"], old, (letter, match)))
    base = os.path.basename(src)
    base = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
    if base.endswith("_regraded"):
        base = base[:-len("_regraded")]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(src))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, base + "_regraded.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    after_hit = sum(1 for r in rows if r["match"])
    after_flag = sum(1 for r in rows if r["needs_handcheck"])
    print(f"{os.path.basename(src)} -> {dst}")
    print(f"correct: {before_hit} -> {after_hit} | "
          f"hand-check: {before_flag} -> {after_flag}")
    for p, q, old, new in changed:
        print(f"  changed {p} q{q}: (pred, match) {old} -> {new}")


if __name__ == "__main__":
    main()
