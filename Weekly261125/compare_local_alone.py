"""27B alone against 4B alone on the evaluation set, worker against worker (note of November 23, section 4).

Read only: it reads two local alone logs and writes one JSON file; it calls no model. Descriptive, no
verdict: 27B alone (qwen3.8:27b on the lab machine) and 4B alone (qwen3:4b on the laptop) are two
workers, not one setting. It uses analyze_a.py's outcomes() and compare() unchanged (the patient
cluster interval of my amendment of November 13: 10,000 replicates, seed 20261105), on the 91
record-dependent questions and on the 200. A question is correct when its log row has match true.

    python compare_local_alone.py --first <27B alone eval log> --second <4B alone eval log> \
        --set record_dependent_set.json --out local_alone_27b_vs_4b.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_a as aa  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="27B alone against 4B alone (read only)")
    ap.add_argument("--first", required=True, help="27B alone eval log")
    ap.add_argument("--second", required=True, help="4B alone eval log")
    ap.add_argument("--first-name", default="27B alone")
    ap.add_argument("--second-name", default="4B alone")
    ap.add_argument("--set", required=True, help="record_dependent_set.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if aa.sha256(aa.pt.__file__) != aa.PAIRED_TESTS_SHA256:
        print("refused: paired_tests.py differs from the pushed one")
        return 5
    if not aa.sha256(args.set).startswith(aa.SET_SHA256_PREFIX):
        print(f"refused: the set file is not the pushed set {aa.SET_SHA256_PREFIX}")
        return 5
    with open(args.set, encoding="utf-8") as f:
        the_set = sorted((q["patient"], q["q_idx"]) for q in json.load(f)["set"])

    rows = {"first": aa.rows_of(args.first), "second": aa.rows_of(args.second)}
    res = {n: aa.outcomes(rs, "match") for n, rs in rows.items()}
    universe = sorted(res["first"])
    if len(universe) != 200 or set(res["second"]) != set(universe) or len(the_set) != 91 \
            or not set(the_set) <= set(universe):
        print(f"refused: questions {len(res['first'])} and {len(res['second'])}, set {len(the_set)}")
        return 5
    # both workers must have read the same padded record for every question
    docs = {n: {(r["patient"], r["q_idx"]): (r.get("documents_sha256"), tuple(r.get("padding_patients") or ()))
                for r in rs if r.get("status") in (None, "ok")} for n, rs in rows.items()}
    differ = [k for k in universe if docs["first"].get(k) != docs["second"].get(k)]
    if differ:
        print(f"refused: the two logs read different records for {len(differ)} questions, e.g. {differ[:3]}")
        return 5

    out = {"rule": "descriptive, no verdict; patient cluster interval of the November 13 amendment, "
                   "10,000 replicates, seed 20261105; difference = first minus second, in points",
           "first": args.first_name, "second": args.second_name,
           "inputs": {n: {"file": os.path.basename(p), "sha256": aa.sha256(p)}
                      for n, p in (("first", args.first), ("second", args.second))},
           "set_sha256": aa.sha256(args.set)}
    for label, keys in (("on_91", the_set), ("on_200", universe)):
        c = aa.compare(res["first"], res["second"], keys, verdict=False)
        out[label] = {"n_questions": c["n_questions"], "n_patients": c["n_patients"],
                      "first_correct": c["arm_correct"], "second_correct": c["cloud_alone_correct"],
                      "difference_points": c["difference_points"], "interval_points": c["interval_points"],
                      "question_interval_points": c["question_interval_points"],
                      "only_first_right": c["only_arm_right"], "only_second_right": c["only_cloud_alone_right"],
                      "failed_first": c["failed_questions"]}
        o = out[label]
        print(f"{label}: {o['n_questions']} questions, {o['n_patients']} patients; "
              f"{args.first_name} {o['first_correct']}, {args.second_name} {o['second_correct']}; "
              f"difference {o['difference_points']} points, patient interval {o['interval_points']}, "
              f"question interval {o['question_interval_points']}; only {args.first_name} right "
              f"{o['only_first_right']}, only {args.second_name} right {o['only_second_right']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
