"""Paired tests for the eval protocols, patient cluster bootstrap.

Reads the logs written by eval_protocols.py and oracle.py and the MinionS
runners. Log files only: no API calls, no worker calls, standard library only.

Primary tests, pre-registration Section 6, 8 tests, family alpha 0.05:
  for each worker (27b, 4b) and each signal (triggered_logprob_mean,
  triggered_agreement), against plan_once and against check_every_step.
  Against plan_once: difference zero, two sided.
  Against check_every_step: null difference <= -0.05, one sided.
  The Holm adjusted p-value decides. The interval is descriptive, and any
  disagreement between the two is reported.

Bootstrap, Section 8: resample patients with replacement, keep each patient's
questions together, paired within question, 10,000 replicates, seed 20261105,
one generator per comparison starting from that seed.
  Interval: 2.5th and 97.5th percentiles, nearest rank.
  Two sided p: 2 * min(share <= 0, share >= 0), capped at 1.
  One sided p: share <= -0.05.
  No smoothing. The resolution at 10,000 replicates is 0.0001.
McNemar, exact, and the question paired bootstrap interval are sensitivity
figures, never used for Holm. McNemar does not test the -0.05 margin.

Reading rule, Section 7: check_every_step counts as better than plan_once only
if its interval against plan_once lies entirely above 0. If it does not, the
margin test says nothing about the signal, and the random control comparison
below is the one that counts.

Random controls, Section 9, exploratory: rescued, broken and net for each
triggered protocol and its control, net difference, and a patient bootstrap
interval for that difference.

Oracle, Section 9, exploratory: gap to plan_once and rescued count. A rescued
count of 0 reports the review step as the bottleneck, unless plan_once has no
wrong answers, which leaves nothing to rescue and shows nothing about reviews.
A count above 0 reports the recovery rate with a patient bootstrap interval, and
each triggered protocol's distance to the oracle.

MinionS, exploratory: score only.

Questions that are missing or failed, Section 11. --on-missing chooses the rule,
and the run stops with a list of them if a rule is needed and none was chosen:
  wrong: the failure rule. A failed question counts as wrong for every protocol
    of the worker whose run failed, and is listed. Worker versus worker scores
    are also given without those questions.
  complete-patients: the fallback if eval cannot finish. Only patients with all
    their questions complete for every worker given, the same subset for every
    protocol and worker. Partial patients are never analyzed.

--interim: the 4B memo before the 27B protocols exist. 4B logs only, raw p-values
and intervals, no Holm correction and no significance claim.

--warmup-check: compares warmup logs with the frozen warmup key table and exits 1
on any difference.

Usage:
  python paired_tests.py --split eval \
    --protocols eval_protocols_27b_eval.jsonl eval_protocols_4b_eval.jsonl \
    --oracle oracle_27b_eval.jsonl oracle_4b_eval.jsonl \
    --minions minions_27b_eval.jsonl minions_4b_eval.jsonl \
    --output paired_tests_eval.json
"""

import argparse
import hashlib
import json
import math
import random
import sys

SEED = 20261105
REPLICATES = 10000
MARGIN = 0.05
ALPHA = 0.05

PATIENTS = tuple(f"patient_{i:02d}" for i in range(1, 21))
QUESTIONS = {"warmup": (2,), "eval": tuple(range(3, 13))}

PROTOCOLS = (
    "plan_once",
    "check_every_step",
    "triggered_agreement",
    "triggered_logprob_mean",
    "random_logprob",
    "random_agreement",
    "sweep_25th",
    "sweep_75th",
)
SIGNALS = (("triggered_logprob_mean", "random_logprob"), ("triggered_agreement", "random_agreement"))
PRIMARY = tuple(
    (worker, signal, comparator, kind)
    for worker in ("27b", "4b")
    for signal, _ in SIGNALS
    for comparator, kind in (("plan_once", "two_sided"), ("check_every_step", "margin"))
)

# A log's worker label starts with the model name, then the context and thinking settings.
WORKER_MODELS = {"qwen3:4b": "4b", "qwen3.8:27b": "27b"}

# (correct, reviews) per protocol, flips (rescue, breakage, neutral), oracle and MinionS correct.
EXPECTED_WARMUP = {
    "27b": {
        "plan_once": (20, 0),
        "check_every_step": (19, 75),
        "triggered_logprob_mean": (20, 30),
        "triggered_agreement": (20, 30),
        "random_logprob": (20, 30),
        "random_agreement": (20, 30),
        "sweep_25th": (20, 15),
        "sweep_75th": (20, 45),
        "flips": (0, 1, 74),
        "oracle": 20,
        "minions": 11,
    },
    "4b": {
        "plan_once": (15, 0),
        "check_every_step": (13, 69),
        "triggered_logprob_mean": (15, 31),
        "triggered_agreement": (13, 49),
        "random_logprob": (15, 31),
        "random_agreement": (13, 49),
        "sweep_25th": (15, 14),
        "sweep_75th": (13, 45),
        "flips": (0, 4, 65),
        "oracle": 15,
        "minions": 10,
    },
}


def fail(message):
    print(f"STOPPED: {message}", file=sys.stderr)
    sys.exit(2)


def worker_id(label):
    model = str(label).split(",")[0].strip()
    if model not in WORKER_MODELS:
        fail(f"unknown worker label {label!r}")
    return WORKER_MODELS[model]


def read_rows(path, notes):
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad:
        notes.append(f"{path}: {bad} unreadable line(s) were not used")
    return rows


def file_hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def row_value(row, field):
    """True or False, or None if the row reports an error or has no value."""
    if row.get("error"):
        return None
    value = row.get(field)
    return None if value is None else bool(value)


def read_protocols(paths, notes):
    """Return results {worker: {protocol: {key: value}}}, reviews, flips and labels."""
    results, reviews, flips, labels = {}, {}, {}, {}
    for path in paths:
        for row in read_rows(path, notes):
            worker = worker_id(row.get("worker", ""))
            labels.setdefault(worker, set()).add(row["worker"])
            name = row.get("display_protocol") or row.get("protocol")
            key = (row.get("patient"), row.get("q_idx"))
            if name == "flip":
                flip_key = key + (row.get("flip_step"),)
                seen = flips.setdefault(worker, {})
                if flip_key in seen:
                    fail(f"{worker} flip {flip_key} appears twice; give one complete log per worker")
                seen[flip_key] = row.get("label")
                continue
            if name not in PROTOCOLS:
                notes.append(f"{path}: rows for unknown protocol {name!r} were not used")
                continue
            seen = results.setdefault(worker, {}).setdefault(name, {})
            if key in seen:
                fail(f"{worker} {name} {key} appears twice; give one complete log per worker")
            seen[key] = row_value(row, "correct")
            reviews.setdefault(worker, {}).setdefault(name, {})[key] = row.get("n_reviews") or 0
    for worker, found in labels.items():
        if len(found) > 1:
            fail(f"{worker} has more than one setting in its logs: {sorted(found)}")
    return results, reviews, flips, {w: sorted(v)[0] for w, v in labels.items()}


def read_baseline(paths, field, notes):
    """Rows of a baseline log, {worker: {key: value}}. Oracle uses found, MinionS uses match."""
    out = {}
    for path in paths:
        for row in read_rows(path, notes):
            worker = worker_id(row.get("worker", ""))
            key = (row.get("patient"), row.get("q_idx"))
            seen = out.setdefault(worker, {})
            if key in seen:
                fail(f"{worker} {key} appears twice in the {field} logs; give one complete log per worker")
            seen[key] = row_value(row, field)
    return out


def expected_keys(split, patients=PATIENTS):
    return [(p, q) for p in patients for q in QUESTIONS[split]]


def failed_questions(results, split):
    """Per worker, questions missing or failed in any protocol."""
    out = {}
    for worker, protocols in results.items():
        bad = set()
        for key in expected_keys(split):
            for name in PROTOCOLS:
                if protocols.get(name, {}).get(key) is None:
                    bad.add(key)
        out[worker] = sorted(bad)
    return out


def choose_subset(results, split, on_missing, notes):
    """Return (keys in patient order, failed per worker), applying the Section 11 rule."""
    failed = failed_questions(results, split)
    if not any(failed.values()):
        return expected_keys(split), failed
    listing = "; ".join(f"{w}: {len(v)} {v[:6]}" for w, v in sorted(failed.items()) if v)
    if on_missing is None:
        fail(f"missing or failed questions, choose --on-missing wrong or complete-patients. {listing}")
    if on_missing == "wrong":
        notes.append(f"failed questions counted wrong for that worker: {listing}")
        return expected_keys(split), failed
    bad_patients = {key[0] for keys in failed.values() for key in keys}
    patients = [p for p in PATIENTS if p not in bad_patients]
    if not patients:
        fail("no patient is complete for every worker given")
    notes.append(f"complete patients only: {len(patients)} of {len(PATIENTS)} kept, dropped {sorted(bad_patients)}")
    return expected_keys(split, patients), {w: [] for w in failed}


def is_correct(results, worker, name, key):
    """A failed or missing question counts as wrong."""
    return bool(results[worker].get(name, {}).get(key))


def by_patient(values):
    """values is [(patient, number)]. Returns [(sum, count)] per patient, in patient order."""
    totals = {}
    for patient, value in values:
        total, count = totals.get(patient, (0, 0))
        totals[patient] = (total + value, count + 1)
    return [totals[p] for p in sorted(totals)]


def bootstrap_patients(clusters, replicates, seed):
    rng = random.Random(seed)
    n = len(clusters)
    diffs = []
    for _ in range(replicates):
        total = count = 0
        for _ in range(n):
            t, c = clusters[rng.randrange(n)]
            total += t
            count += c
        diffs.append(total / count)
    diffs.sort()
    return diffs


def bootstrap_questions(values, replicates, seed):
    rng = random.Random(seed)
    n = len(values)
    diffs = sorted(sum(rng.choices(values, k=n)) / n for _ in range(replicates))
    return diffs


def interval(sorted_diffs):
    """2.5th and 97.5th percentiles, nearest rank."""
    n = len(sorted_diffs)
    return sorted_diffs[math.ceil(0.025 * n) - 1], sorted_diffs[math.ceil(0.975 * n) - 1]


def p_value(sorted_diffs, kind):
    n = len(sorted_diffs)
    if kind == "margin":
        return sum(1 for d in sorted_diffs if d <= -MARGIN) / n
    low = sum(1 for d in sorted_diffs if d <= 0) / n
    high = sum(1 for d in sorted_diffs if d >= 0) / n
    return min(1.0, 2 * min(low, high))


def mcnemar_exact(b, c):
    """Exact two sided McNemar, from the binomial with p = 0.5 on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(raw_ps):
    """Holm step down. Adjusted p-values, in the order given."""
    m = len(raw_ps)
    order = sorted(range(m), key=lambda i: raw_ps[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * raw_ps[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare(results, worker, name_a, name_b, keys, kind, replicates, seed):
    pairs = [(k, is_correct(results, worker, name_a, k), is_correct(results, worker, name_b, k)) for k in keys]
    values = [int(a) - int(b) for _, a, b in pairs]
    clusters = by_patient([(k[0], int(a) - int(b)) for k, a, b in pairs])
    diffs = bootstrap_patients(clusters, replicates, seed)
    lo, hi = interval(diffs)
    q_lo, q_hi = interval(bootstrap_questions(values, replicates, seed))
    b = sum(1 for _, x, y in pairs if x and not y)
    c = sum(1 for _, x, y in pairs if y and not x)
    return {
        "worker": worker,
        "signal": name_a,
        "comparator": name_b,
        "kind": kind,
        "n": len(keys),
        "n_patients": len(clusters),
        "correct_signal": sum(1 for _, x, _ in pairs if x),
        "correct_comparator": sum(1 for _, _, y in pairs if y),
        "diff": sum(values) / len(values),
        "interval_lo": lo,
        "interval_hi": hi,
        "question_interval_lo": q_lo,
        "question_interval_hi": q_hi,
        "p_raw": p_value(diffs, kind),
        "mcnemar_p": mcnemar_exact(b, c),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "no_discordant_pairs": b + c == 0,
    }


def primary_tests(results, keys, replicates, seed, interim):
    tests = [t for t in PRIMARY if t[0] == "4b"] if interim else list(PRIMARY)
    missing = [t for t in tests if t[0] not in results or t[1] not in results[t[0]] or t[2] not in results[t[0]]]
    if missing:
        fail(f"primary tests cannot run, protocols or workers are missing: {missing}")
    rows = [compare(results, w, s, c, keys, kind, replicates, seed) for w, s, c, kind in tests]
    if interim:
        for row in rows:
            row["label"] = "interim, raw p-value, no significance claim"
        return rows
    for row, adjusted in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = adjusted
        row["significant"] = adjusted < ALPHA
        if row["kind"] == "margin":
            row["interval_says"] = row["interval_lo"] > -MARGIN
        else:
            row["interval_says"] = row["interval_lo"] > 0 or row["interval_hi"] < 0
        row["holm_and_interval_disagree"] = row["significant"] != row["interval_says"]
    return rows


def reading_rule(results, keys, replicates, seed):
    out = []
    for worker in sorted(results):
        row = compare(results, worker, "check_every_step", "plan_once", keys, "two_sided", replicates, seed)
        out.append({
            "worker": worker,
            "interval_lo": row["interval_lo"],
            "interval_hi": row["interval_hi"],
            "check_better": row["interval_lo"] > 0,
        })
    return out


def net_counts(results, worker, name, keys):
    rescued = broken = 0
    for key in keys:
        plan = is_correct(results, worker, "plan_once", key)
        other = is_correct(results, worker, name, key)
        rescued += (not plan) and other
        broken += plan and not other
    return rescued, broken


def random_controls(results, keys, replicates, seed):
    out = []
    for worker in sorted(results):
        for signal, control in SIGNALS:
            s_r, s_b = net_counts(results, worker, signal, keys)
            c_r, c_b = net_counts(results, worker, control, keys)
            row = compare(results, worker, signal, control, keys, "two_sided", replicates, seed)
            n = len(keys)
            if row["interval_lo"] > 0:
                reading = "beats random on net"
            elif row["interval_hi"] < 0:
                reading = "worse than random on net"
            else:
                reading = "does not beat random on net"
            out.append({
                "worker": worker,
                "signal": signal,
                "control": control,
                "signal_rescued": s_r,
                "signal_broken": s_b,
                "signal_net": s_r - s_b,
                "control_rescued": c_r,
                "control_broken": c_b,
                "control_net": c_r - c_b,
                "net_difference": (s_r - s_b) - (c_r - c_b),
                "net_interval_lo": row["interval_lo"] * n,
                "net_interval_hi": row["interval_hi"] * n,
                "reading": reading,
            })
    return out


def oracle_summary(results, oracle, keys, replicates, seed, notes):
    out = {}
    for worker in sorted(oracle):
        if worker not in results:
            notes.append(f"oracle rows for {worker} were not used, that worker has no protocol log")
            continue
        found = oracle[worker]
        absent = [k for k in keys if found.get(k) is None]
        if absent:
            notes.append(f"oracle {worker}: {len(absent)} questions missing or failed, counted as not found: {absent[:6]}")
        plan = {k: is_correct(results, worker, "plan_once", k) for k in keys}
        correct = {k: bool(found.get(k)) for k in keys}
        rescued = sum(1 for k in keys if correct[k] and not plan[k])
        broken = sum(1 for k in keys if plan[k] and not correct[k])
        plan_wrong = sum(1 for k in keys if not plan[k])
        n = len(keys)
        entry = {
            "correct": sum(correct.values()),
            "n": n,
            "plan_correct": n - plan_wrong,
            "gap": (sum(correct.values()) - (n - plan_wrong)) / n,
            "rescued": rescued,
            "broken": broken,
            "plan_wrong": plan_wrong,
            "nothing_to_rescue": plan_wrong == 0,
            "review_step_is_bottleneck": rescued == 0 and plan_wrong > 0,
        }
        if broken:
            notes.append(f"oracle {worker}: {broken} plan correct questions were not found, which should not happen")
        if rescued:
            clusters = {}
            for k in keys:
                r, w = clusters.get(k[0], (0, 0))
                clusters[k[0]] = (r + (correct[k] and not plan[k]), w + (not plan[k]))
            items = [clusters[p] for p in sorted(clusters)]
            rng = random.Random(seed)
            rates = []
            skipped = 0
            for _ in range(replicates):
                total_r = total_w = 0
                for _ in range(len(items)):
                    r, w = items[rng.randrange(len(items))]
                    total_r += r
                    total_w += w
                if total_w:
                    rates.append(total_r / total_w)
                else:
                    skipped += 1
            rates.sort()
            entry["recovery_rate"] = rescued / plan_wrong
            entry["recovery_lo"], entry["recovery_hi"] = interval(rates)
            entry["recovery_replicates_skipped"] = skipped
            entry["distance_to_oracle"] = {
                signal: (sum(correct.values()) - sum(is_correct(results, worker, signal, k) for k in keys)) / n
                for signal, _ in SIGNALS
            }
        out[worker] = entry
    return out


def minions_summary(minions, keys, notes):
    out = {}
    for worker in sorted(minions):
        rows = minions[worker]
        failed = [k for k in keys if rows.get(k) is None]
        if failed:
            notes.append(f"MinionS {worker}: {len(failed)} questions missing or failed, counted wrong: {failed[:6]}")
        out[worker] = {"correct": sum(1 for k in keys if rows.get(k)), "n": len(keys), "failed": len(failed)}
    return out


def cross_worker(results, keys, failed):
    """Worker versus worker scores, with and without the questions that failed on either worker."""
    if len(results) < 2:
        return None
    bad = {k for v in failed.values() for k in v}
    clean = [k for k in keys if k not in bad]
    out = {"n_all": len(keys), "n_without_failed": len(clean), "protocols": {}}
    for name in PROTOCOLS:
        out["protocols"][name] = {
            worker: {
                "correct_all": sum(is_correct(results, worker, name, k) for k in keys),
                "correct_without_failed": sum(is_correct(results, worker, name, k) for k in clean),
            }
            for worker in sorted(results)
        }
    return out


def score_table(results, reviews, keys):
    table = {}
    for worker in sorted(results):
        table[worker] = {}
        for name in PROTOCOLS:
            if name not in results[worker]:
                continue
            rescued, broken = net_counts(results, worker, name, keys)
            table[worker][name] = {
                "correct": sum(is_correct(results, worker, name, k) for k in keys),
                "n": len(keys),
                "reviews": sum(reviews[worker][name].get(k, 0) for k in keys),
                "rescued": rescued,
                "broken": broken,
            }
    return table


def flip_summary(flips, keys):
    wanted = set(keys)
    out = {}
    for worker, rows in flips.items():
        counts = {"rescue": 0, "breakage": 0, "neutral": 0}
        for key, label in rows.items():
            if key[:2] in wanted and label in counts:
                counts[label] += 1
        out[worker] = counts
    return out


def warmup_check(result):
    problems = []
    for worker in sorted(result["scores"]):
        expected = EXPECTED_WARMUP[worker]
        for name, row in result["scores"][worker].items():
            if (row["correct"], row["reviews"]) != expected[name]:
                problems.append(f"{worker} {name}: got {(row['correct'], row['reviews'])}, expected {expected[name]}")
        counts = result["flips"].get(worker, {})
        got = (counts.get("rescue"), counts.get("breakage"), counts.get("neutral"))
        if got != expected["flips"]:
            problems.append(f"{worker} flips: got {got}, expected {expected['flips']}")
        if worker in result.get("oracle", {}) and result["oracle"][worker]["correct"] != expected["oracle"]:
            problems.append(f"{worker} oracle: got {result['oracle'][worker]['correct']}, expected {expected['oracle']}")
        if worker in result.get("minions", {}) and result["minions"][worker]["correct"] != expected["minions"]:
            problems.append(f"{worker} MinionS: got {result['minions'][worker]['correct']}, expected {expected['minions']}")
    return problems


def analyze(args, notes):
    results, reviews, flips, labels = read_protocols(args.protocols, notes)
    if not results:
        fail("no protocol rows found")
    if args.interim and set(results) != {"4b"}:
        fail(f"--interim reads 4B logs only, got {sorted(results)}")
    keys, failed = choose_subset(results, args.split, args.on_missing, notes)
    if args.interim and args.on_missing == "complete-patients":
        notes.append("27B logs were not given, so completeness here covers 4B only")
    result = {
        "split": args.split,
        "mode": "interim, 4B only, no Holm" if args.interim else "final, Holm across 8 tests",
        "on_missing": args.on_missing,
        "worker_settings": labels,
        "replicates": args.replicates,
        "seed": args.seed,
        "n_questions": len(keys),
        "n_patients": len({k[0] for k in keys}),
        "failed_questions": {w: [list(k) for k in v] for w, v in failed.items()},
        "input_sha256": {p: file_hash(p) for p in args.protocols + args.oracle + args.minions},
        "scores": score_table(results, reviews, keys),
        "flips": flip_summary(flips, keys),
        "primary": primary_tests(results, keys, args.replicates, args.seed, args.interim),
        "reading_rule": reading_rule(results, keys, args.replicates, args.seed),
        "random_controls": random_controls(results, keys, args.replicates, args.seed),
    }
    if args.oracle:
        oracle = read_baseline(args.oracle, "found", notes)
        result["oracle"] = oracle_summary(results, oracle, keys, args.replicates, args.seed, notes)
    if args.minions:
        result["minions"] = minions_summary(read_baseline(args.minions, "match", notes), keys, notes)
    result["worker_versus_worker"] = cross_worker(results, keys, failed)
    if args.replicates != REPLICATES or args.seed != SEED:
        notes.append(f"replicates or seed differ from the frozen {REPLICATES} and {SEED}")
    result["notes"] = notes
    return result


def print_result(result):
    print(f"Split {result['split']}, {result['mode']}: {result['n_questions']} questions, "
          f"{result['n_patients']} patients, {result['replicates']} replicates, seed {result['seed']}")
    print(f"Workers: {result['worker_settings']}")
    for worker, table in result["scores"].items():
        print(f"\nWorker {worker}, flips {result['flips'].get(worker)}")
        for name, row in table.items():
            print(f"  {name}: {row['correct']}/{row['n']}, reviews {row['reviews']}, "
                  f"rescued {row['rescued']}, broken {row['broken']}")
    print("\nPrimary tests:")
    for row in result["primary"]:
        decision = (f"p_holm {row['p_holm']:.4f}, significant {row['significant']}, "
                    f"disagree with interval {row['holm_and_interval_disagree']}" if "p_holm" in row else row["label"])
        print(f"  {row['worker']} {row['signal']} vs {row['comparator']} ({row['kind']}): "
              f"{row['correct_signal']} vs {row['correct_comparator']} of {row['n']}, diff {row['diff']:+.4f}, "
              f"interval [{row['interval_lo']:+.4f}, {row['interval_hi']:+.4f}], "
              f"question interval [{row['question_interval_lo']:+.4f}, {row['question_interval_hi']:+.4f}], "
              f"p_raw {row['p_raw']:.4f}, {decision}, "
              f"McNemar p {row['mcnemar_p']:.4f} (b {row['mcnemar_b']}, c {row['mcnemar_c']})"
              + (", no discordant pairs, so the two protocols gave the same results" if row["no_discordant_pairs"] else ""))
    print("\nReading rule, check versus plan:")
    for row in result["reading_rule"]:
        print(f"  {row['worker']}: interval [{row['interval_lo']:+.4f}, {row['interval_hi']:+.4f}], "
              f"check better {row['check_better']}")
    print("\nRandom controls, exploratory:")
    for row in result["random_controls"]:
        print(f"  {row['worker']} {row['signal']}: rescued {row['signal_rescued']}, broken {row['signal_broken']}, "
              f"net {row['signal_net']}; control rescued {row['control_rescued']}, broken {row['control_broken']}, "
              f"net {row['control_net']}; net difference {row['net_difference']:+d}, "
              f"interval [{row['net_interval_lo']:+.2f}, {row['net_interval_hi']:+.2f}]: {row['reading']}")
    if "oracle" in result:
        print(f"\nOracle, exploratory: {result['oracle']}")
    if "minions" in result:
        print(f"\nMinionS, exploratory: {result['minions']}")
    if result["notes"]:
        print("\nNotes:")
        for note in result["notes"]:
            print(f"  {note}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Paired tests from eval protocol logs.")
    ap.add_argument("--split", choices=sorted(QUESTIONS), required=True)
    ap.add_argument("--protocols", nargs="+", required=True)
    ap.add_argument("--oracle", nargs="*", default=[])
    ap.add_argument("--minions", nargs="*", default=[])
    ap.add_argument("--on-missing", choices=["wrong", "complete-patients"], default=None)
    ap.add_argument("--interim", action="store_true")
    ap.add_argument("--warmup-check", action="store_true")
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    if args.warmup_check and args.split != "warmup":
        ap.error("--warmup-check applies to --split warmup only")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = analyze(args, [])
    print_result(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"\nJSON: {args.output}")
    if args.warmup_check:
        problems = warmup_check(result)
        if problems:
            print("\nWARMUP CHECK FAILED:")
            for problem in problems:
                print(f"  {problem}")
            sys.exit(1)
        print("\nWARMUP CHECK PASSED: scores, reviews, flips, oracle and MinionS match the frozen warmup table.")


if __name__ == "__main__":
    main()
