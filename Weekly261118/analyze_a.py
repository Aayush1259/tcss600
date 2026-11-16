"""Question A: each arm against cloud alone, by the rule of the November 12 amendment (C6, November 14).

Read-only: it reads logs and writes one JSON result file; it calls no model.

The rule (amendment of November 12, section 4; agreed by Prof. Cai on November 12):
- primary: plan once and check every step against cloud alone, 4B worker, on the 91
  record-dependent questions (record_dependent_set.json, SHA-256 prefix d288c671); each judged on
  its own; exploratory, no correction for two comparisons;
- the difference is the arm's accuracy minus cloud alone's on the same questions, paired by
  question; a failed question (final after its one rerun) counts as wrong; only patients complete
  in both arms are used;
- the interval is Section 8's, unchanged: patient cluster bootstrap, 20 patients resampled with
  replacement, each sampled patient's questions kept together, 10,000 replicates, seed 20261105,
  2.5th and 97.5th percentiles by nearest rank. The bootstrap functions are imported unchanged from
  paired_tests.py (pushed November 4; its hash is checked). The question-level bootstrap is the
  sensitivity check;
- close: the lower end is above minus 10 points; worse: the upper end is below minus 10 points;
  otherwise inconclusive. Never loosened after a result is seen;
- with each comparison: the questions only the arm got right and only cloud alone got right;
- more than 5 percent failed questions in an arm: the arm is reported as partial;
- secondary: the same on all 200 and on the 109 outside the set.

Also, from amendment 2: the triggered log-prob arm and 4B alone as context; the cost ratio (the
arm's protocol accounting cloud dollars over cloud alone's on the same questions) with the fresh
dollars beside it when a calls log is given; the laptop's seconds per question; "who does the work"
(correct answers whose chosen steps had no worker finding).

From the pre-registration (Section 7, frozen November 3): check every step against plan once by the
reading rule (better only if the interval lies entirely above 0); rescued and broken for each protocol
against plan once and against 4B alone; triggered against check every step within 5 points, descriptive.

It refuses to run while any arm still has questions without a final result, unless
--complete-patients-only is given (the fallback of Section 11, after the deadline), so a partial
look at the evaluation cannot happen by accident.

    python analyze_a.py --cloud-alone <cloud alone eval log> --scores <score_c3 eval log> \
        --shared <eval shared log> --local-alone <local_alone_eval log> --set record_dependent_set.json \
        --out question_a_results.json
"""
import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paired_tests as pt  # noqa: E402

import datetime  # noqa: E402

# The fallback of Section 11 (complete patients only) may be used only from this time on (amendment 2).
PARTIAL_CUTOFF = datetime.datetime(2026, 11, 19, 12, 0)
PAIRED_TESTS_SHA256 ="986c56ca281679dc3deab5db57c3fd04b9ca7d86306da89aab135791213f35d1"
SET_SHA256_PREFIX = "d288c671"
MARGIN_POINTS = 10.0
PRIMARY = ("plan_once", "check_every_step")
CONTEXT = ("triggered_logprob_mean", "local_alone")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rows_of(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def outcomes(rows, correct_field, ok=lambda r: True):
    """{(patient, q_idx): True/False/"failed"}. Two failed rows and no ok row: final, counted wrong."""
    res, failed = {}, collections.Counter()
    for r in rows:
        key = (r["patient"], r["q_idx"])
        if r.get("status") == "failed":
            failed[key] += 1
        elif r.get("status") in (None, "ok") and ok(r):
            v = bool(r.get(correct_field))
            if key in res and res[key] != v:
                raise SystemExit(f"refused: two answered rows disagree for {key}")
            res[key] = v
    for key, n in failed.items():
        if n >= 2 and key not in res:
            res[key] = "failed"
    return res


def compare(arm, cloud, keys, verdict=True):
    """The frozen Section 8 interval on the complete patients of both arms. `cloud` is the reference
    arm (cloud alone, or 4B alone with verdict=False: no margin was fixed for that comparison)."""
    patients = sorted({k[0] for k in keys})
    complete = [p for p in patients
                if all(k in arm and k in cloud for k in keys if k[0] == p)]
    use = [k for k in keys if k[0] in complete]
    if not use:
        return {"n_questions": 0, "n_patients": 0, "patients_left_out": patients, "verdict": None,
                "difference_points": None, "interval_points": None, "partial": False,
                "only_arm_right": 0, "only_cloud_alone_right": 0}
    a = {k: arm[k] is True for k in use}          # "failed" counts as wrong
    c = {k: cloud[k] is True for k in use}
    values = [int(a[k]) - int(c[k]) for k in use]
    clusters = pt.by_patient([(k[0], int(a[k]) - int(c[k])) for k in use])
    lo, hi = pt.interval(pt.bootstrap_patients(clusters, pt.REPLICATES, pt.SEED))
    q_lo, q_hi = pt.interval(pt.bootstrap_questions(values, pt.REPLICATES, pt.SEED))
    lo_p, hi_p = 100 * lo, 100 * hi
    if verdict:
        verdict = "close" if lo_p > -MARGIN_POINTS else "worse" if hi_p < -MARGIN_POINTS else "inconclusive"
    else:
        verdict = None
    n_failed = sum(1 for k in use if arm[k] == "failed")
    return {"n_questions": len(use), "n_patients": len(complete),
            "patients_left_out": [p for p in patients if p not in complete],
            "questions_per_patient": dict(collections.Counter(k[0] for k in use)),
            "arm_correct": sum(a.values()), "cloud_alone_correct": sum(c.values()),
            "difference_points": round(100 * sum(values) / len(values), 2) if values else None,
            "interval_points": [round(lo_p, 2), round(hi_p, 2)],
            "question_interval_points": [round(100 * q_lo, 2), round(100 * q_hi, 2)],
            "verdict": verdict, "only_arm_right": sum(1 for k in use if a[k] and not c[k]),
            "only_cloud_alone_right": sum(1 for k in use if c[k] and not a[k]),
            "failed_questions": n_failed, "partial": bool(use) and n_failed / len(use) > 0.05}


def against(arm, ref, keys):
    """compare() with a reference other than cloud alone, the fields named for it: rescued means the
    reference was wrong and the arm right, broken the reverse (the definition of my November 4 report)."""
    c = compare(arm, ref, keys, verdict=False)
    return {"n_questions": c["n_questions"], "n_patients": c["n_patients"],
            "arm_correct": c.get("arm_correct"), "reference_correct": c.get("cloud_alone_correct"),
            "difference_points": c["difference_points"], "interval_points": c["interval_points"],
            "question_interval_points": c.get("question_interval_points"),
            "rescued": c["only_arm_right"], "broken": c["only_cloud_alone_right"]}


def switch(c):
    """Pre-registration Section 7, the reading rule, unchanged: check every step counts as better than
    plan once only if the patient cluster 95 percent interval of check minus plan lies entirely above 0.
    (A nonzero interval end is at least one question over the resample, far above the 0.01 rounding.)"""
    if not c["n_questions"]:
        return None
    return "better" if c["interval_points"][0] > 0 else "not better"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Question A by the November 12 rule (read-only)")
    ap.add_argument("--cloud-alone", required=True)
    ap.add_argument("--scores", required=True, help="score_c3.py log on the evaluation split")
    ap.add_argument("--shared", required=True, help="the 4B shared parts log on the evaluation split")
    ap.add_argument("--local-alone")
    ap.add_argument("--set", required=True, help="record_dependent_set.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--complete-patients-only", action="store_true")
    ap.add_argument("--calls", nargs="*", default=[], help="calls logs of the shared run and the scoring run (fresh dollars)")
    args = ap.parse_args(argv)
    if args.complete_patients_only and datetime.datetime.now() < PARTIAL_CUTOFF:
        print(f"refused: complete patients only is allowed from {PARTIAL_CUTOFF:%B %d %H:%M} on (amendment 2)")
        return 6

    if sha256(pt.__file__) != PAIRED_TESTS_SHA256:
        print("refused: paired_tests.py differs from the pushed one")
        return 5
    if not sha256(args.set).startswith(SET_SHA256_PREFIX):
        print(f"refused: the set file is not the pushed set {SET_SHA256_PREFIX}")
        return 5
    with open(args.set, encoding="utf-8") as f:
        the_set = {(q["patient"], q["q_idx"]) for q in json.load(f)["set"]}

    cloud_rows = rows_of(args.cloud_alone)
    # a cloud alone failure row counts only after its one rerun also failed (November 12 rule)
    cloud = outcomes([{**r, "status": "failed" if r.get("failed") else "ok"}
                      for r in cloud_rows if r.get("split") == "eval"], "match")
    universe = sorted(cloud)
    if len(universe) != 200 or not the_set <= set(universe) or len(the_set) != 91:
        print(f"refused: cloud alone has {len(universe)} questions; the set {len(the_set)}")
        return 5

    score_rows = rows_of(args.scores)
    shared_rows = rows_of(args.shared)
    shared_final = {k for k, v in outcomes(shared_rows, "status").items() if v == "failed"}
    # the scores must come from this shared log and one fixed threshold
    ok_scores = [r for r in score_rows if r.get("status") != "failed"]
    logs = {r.get("shared_log") for r in ok_scores}
    if ok_scores and logs != {os.path.basename(args.shared)}:
        print(f"refused: the score rows come from {sorted(map(str, logs))}, not {os.path.basename(args.shared)}")
        return 5
    thresholds = {r.get("threshold") for r in ok_scores}
    if len(thresholds) > 1:
        print(f"refused: the score rows use more than one threshold {sorted(map(str, thresholds))}")
        return 5

    # every arm must have read the same padded record for each question
    pads, docs = collections.defaultdict(set), collections.defaultdict(set)
    sources = [("cloud alone", cloud_rows), ("shared", shared_rows)]
    if args.local_alone:
        sources.append(("4B alone", rows_of(args.local_alone)))
    for label, rs in sources:
        for r in rs:
            if r.get("status") not in (None, "ok"):
                continue
            if r.get("padding_patients") is not None:
                pads[(r["patient"], r["q_idx"])].add(tuple(r["padding_patients"]))
            if r.get("documents_sha256"):              # shared and 4B alone rows carry the record's hash
                docs[(r["patient"], r["q_idx"])].add(r["documents_sha256"])
    differ = [k for k, v in pads.items() if len(v) > 1] + [k for k, v in docs.items() if len(v) > 1]
    if differ:
        print(f"refused: the arms read different padded records for {len(differ)} questions, e.g. {differ[:3]}")
        return 5
    arms = {}
    for name in ("plan_once", "check_every_step", "triggered_logprob_mean"):
        res = outcomes([r for r in score_rows if r.get("arm") == name or r.get("status") == "failed"], "correct")
        for k in shared_final:                 # shared parts final failure: wrong in every pipeline arm
            res.setdefault(k, "failed")
        arms[name] = res
    if args.local_alone:
        arms["local_alone"] = outcomes(rows_of(args.local_alone), "match")

    open_q = {n: [k for k in universe if k not in res] for n, res in arms.items()}
    if any(open_q.values()) and not args.complete_patients_only:
        print("refused: some arms still have questions without a final result "
              f"{ {n: len(v) for n, v in open_q.items()} }; run after the arms finish, or use "
              "--complete-patients-only after the deadline (Section 11)")
        return 6

    # Section 11: one subset for every arm: the patients complete in cloud alone and in every arm given
    keep = {p for p in {k[0] for k in universe}
            if all(k in cloud and all(k in res for res in arms.values()) for k in universe if k[0] == p)}
    universe = [k for k in universe if k[0] in keep]
    the_set = {k for k in the_set if k[0] in keep}
    rest = [k for k in universe if k not in the_set]
    out = {"patients_used": sorted(keep),
           "rule_note": "complete patients only, same subset for every arm" if args.complete_patients_only else "all patients",
           **{}}
    out.update({"rule": "November 12 amendment, section 4; margin -10 points; Section 8 interval, seed 20261105",
           "set_sha256": sha256(args.set), "inputs": {k: os.path.basename(v) for k, v in vars(args).items()
                                                     if k in ("cloud_alone", "scores", "shared", "local_alone") and v},
           "comparisons": {}})
    for name, res in arms.items():
        primary = name in PRIMARY                 # the November 12 verdict is for the two main arms only
        out["comparisons"][name] = {
            "role": "primary" if primary else "context",
            "on_91": compare(res, cloud, sorted(the_set), primary),
            "on_200": compare(res, cloud, universe, primary),
            "on_109": compare(res, cloud, rest, primary)}

    # cloud alone is graded on its JSON answer only; the pipeline arms also accept the rule-based reading.
    # Sensitivity: the same comparison with every answer read only by the fallback counted wrong.
    out["fallback"] = {}
    for name in ("plan_once", "check_every_step", "triggered_logprob_mean"):
        fb = {(r["patient"], r["q_idx"]) for r in score_rows
              if r.get("arm") == name and r.get("agg_parse") == "rule-fallback"}
        strict = {k: (False if k in fb and v is True else v) for k, v in arms[name].items()}
        out["fallback"][name] = {"answers_read_by_fallback_on_91": len(fb & the_set),
                                 "correct_by_fallback_on_91": sum(1 for k in fb & the_set if arms[name].get(k) is True),
                                 "on_91_counted_wrong": compare(strict, cloud, sorted(the_set), name in PRIMARY)}
    # Prof. Cai, November 13: plan once and check every step against cloud alone and against 4B alone.
    # The rule of November 12 decides only the comparison with cloud alone; against 4B alone the
    # difference and the same interval are reported, with no verdict.
    if "local_alone" in arms:
        out["against_4b_alone"] = {name: {"on_91": compare(arms[name], arms["local_alone"], sorted(the_set), False),
                                          "on_200": compare(arms[name], arms["local_alone"], universe, False)}
                                   for name in PRIMARY}

    # Pre-registration Section 7 (frozen November 3): plan once against check every step by the switch,
    # on the 91 and on all 200 (the pre-registration was written for the 200; the 91 are the primary set
    # since November 11); both are reported, neither is chosen after the result.
    sets = (("on_91", sorted(the_set)), ("on_200", universe))
    out["check_against_plan"] = {}
    for label, keys in sets:
        c = against(arms["check_every_step"], arms["plan_once"], keys)
        out["check_against_plan"][label] = {**c, "switch": switch(c)}
    # Rescued and broken for each protocol (Prof. Cai, October 28): against plan once, and against 4B alone
    out["rescued_broken"] = {}
    for name in ("check_every_step", "triggered_logprob_mean"):
        out["rescued_broken"][f"{name}_against_plan_once"] = {
            label: {k: v for k, v in against(arms[name], arms["plan_once"], keys).items() if k in ("n_questions", "rescued", "broken")}
            for label, keys in sets}
    if "local_alone" in arms:
        for name in ("plan_once", "check_every_step", "triggered_logprob_mean"):
            out["rescued_broken"][f"{name}_against_4b_alone"] = {
                label: {k: v for k, v in against(arms[name], arms["local_alone"], keys).items() if k in ("n_questions", "rescued", "broken")}
                for label, keys in sets}
    # Triggered against check every step within 5 points (Section 7), descriptive: Question B is
    # exploratory and the Holm tests are paused (amendment 2). Within 5 points means the interval lies
    # entirely above minus 5; it says something about the signal only if the switch says check is better.
    out["triggered_against_check"] = {}
    for label, keys in sets:
        c = against(arms["triggered_logprob_mean"], arms["check_every_step"], keys)
        within = None if not c["n_questions"] else c["interval_points"][0] > -5.0
        out["triggered_against_check"][label] = {
            **c, "within_5_points": within,
            "meaningful": out["check_against_plan"][label]["switch"] == "better",
            "note": "descriptive; read only if check every step is better than plan once by the switch"}

    by_arm =collections.defaultdict(dict)            # one row per question and arm (a rescored question once)
    for r in score_rows:
        if r.get("status") in (None, "ok"):
            by_arm[r["arm"]][(r["patient"], r["q_idx"])] = r
    cloud_usd = {(r["patient"], r["q_idx"]): r.get("est_cost_usd") or 0.0 for r in cloud_rows}
    cost = {}
    for name, d in by_arm.items():
        rs = list(d.values())
        for label, keys in (("on_91", the_set), ("on_200", set(universe))):
            sel = [r for r in rs if (r["patient"], r["q_idx"]) in keys]
            arm_usd = sum(r.get("cloud_usd") or 0.0 for r in sel)
            ref_usd = sum(cloud_usd[(r["patient"], r["q_idx"])] for r in sel)
            cost.setdefault(name, {})[label] = {
                "questions": len(sel), "arm_cloud_usd": round(arm_usd, 4), "cloud_alone_usd": round(ref_usd, 4),
                "ratio": round(arm_usd / ref_usd, 4) if ref_usd else None,
                "laptop_seconds_per_question": round(sum(r.get("local_gpu_s") or 0 for r in sel) / len(sel), 1) if sel else None,
                "correct_with_no_worker_finding": sum(1 for r in sel if r.get("correct") and not r.get("worker_findings"))}
    out["cost_and_work"] = cost
    # fresh dollars actually paid, from the calls logs (appendix; the headline is protocol accounting)
    fresh = 0.0
    for path in sorted(set(args.calls)):
        fresh += sum(r.get("usd") or 0.0 for r in rows_of(path) if r.get("kind") == "gemini" and not r.get("cached"))
    out["fresh_dollars_paid"] = {"calls_logs": [os.path.basename(p) for p in args.calls], "usd": round(fresh, 4)}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for name, c in out["comparisons"].items():
        p = c["on_91"]
        verdict = (p["verdict"] or "no verdict (context arm)").upper() if p["n_questions"] else "NO USABLE QUESTION"
        print(f"{name} ({c['role']}), on the 91: difference {p['difference_points']} points, "
              f"95% patient cluster interval {p['interval_points']}, {verdict}"
              f"{', PARTIAL' if p['partial'] else ''}; only arm right {p['only_arm_right']}, "
              f"only cloud alone right {p['only_cloud_alone_right']}; {p['n_questions']} questions, "
              f"{p['n_patients']} patients")
    for name, c in out.get("against_4b_alone", {}).items():
        p = c["on_91"]
        print(f"{name} against 4B alone, on the 91: difference {p['difference_points']} points, "
              f"95% patient cluster interval {p['interval_points']} (no margin, no verdict); only {name} right "
              f"{p['only_arm_right']}, only 4B alone right {p['only_cloud_alone_right']}")
    for label in ("on_91", "on_200"):
        s = out["check_against_plan"][label]
        print(f"check every step against plan once, {label[3:]}: difference {s['difference_points']} points, "
              f"95% patient cluster interval {s['interval_points']}, switch: {str(s['switch']).upper()}; "
              f"rescued {s['rescued']}, broken {s['broken']}")
    for key, v in out["rescued_broken"].items():
        print(f"{key}, on the 91: rescued {v['on_91']['rescued']}, broken {v['on_91']['broken']}")
    t = out["triggered_against_check"]["on_91"]
    print(f"triggered against check every step, on the 91: difference {t['difference_points']} points, interval "
          f"{t['interval_points']}, within 5 points {t['within_5_points']} (descriptive"
          f"{'' if t['meaningful'] else '; says nothing about the signal, check is not better'})")
    for name, c in cost.items():
        print(f"{name}: cloud dollars ratio on the 91 {c['on_91']['ratio']}, laptop seconds per question "
              f"{c['on_91']['laptop_seconds_per_question']}, correct with no worker finding "
              f"{c['on_91']['correct_with_no_worker_finding']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
