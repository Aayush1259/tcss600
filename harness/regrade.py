"""Regrade a baseline JSONL with the current grader, using the saved answers.

The log stores full prompts and answers, so grader improvements never
require rerunning the model. Options are reloaded from the benchmark via
patient/q_idx, so old logs regrade cleanly.

Machine: bosgame (needs the benchmark + ID files). Run from ~.

Usage:
    ~/is-env/bin/python regrade.py /home/localai/baseline_logs/<file>.jsonl

Writes <file>_regraded.jsonl and prints a before/after summary.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader  # noqa: E402
import grader  # noqa: E402


def main():
    src = sys.argv[1]
    bench = data_loader.load_benchmark()
    rows = [json.loads(line) for line in open(src)]
    before_hit = sum(1 for r in rows if r["match"])
    before_flag = sum(1 for r in rows if r["needs_handcheck"])
    for r in rows:
        rec = data_loader.get_question(bench, r["patient"], r["q_idx"])
        g = grader.grade(r["answer"] or "", r["gold"], rec["options"])
        r["predicted"] = g["letter"]
        r["match"] = g["match"]
        r["needs_handcheck"] = g["needs_handcheck"]
        r["grading"] = f"rule-based ({g['method']}) + hand-check queue [regraded]"
    dst = src.replace(".jsonl", "_regraded.jsonl")
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    after_hit = sum(1 for r in rows if r["match"])
    after_flag = sum(1 for r in rows if r["needs_handcheck"])
    print(f"{os.path.basename(src)} -> {os.path.basename(dst)}")
    print(f"correct: {before_hit} -> {after_hit} | "
          f"hand-check: {before_flag} -> {after_flag}")


if __name__ == "__main__":
    main()
