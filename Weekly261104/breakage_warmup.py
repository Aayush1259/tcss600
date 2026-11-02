"""Print breakage flips from a flips log, joined to the shared log.

Reads only, no API, no key. For each flip row with label breakage,
prints patient, step, plan answer, flip answer, gold, and the shared
step's supervisor verdict (redo or accept) and whether a reviewed
job exists. Also prints the plan_once wrong questions and a count
check for rescue rows.

Usage, from any folder on the machine holding the logs:
  python3 breakage_warmup.py --flips <flips log> --shared <shared log>
"""
import argparse
import json


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flips", required=True)
    ap.add_argument("--shared", required=True)
    args = ap.parse_args()
    flips = load(args.flips)
    shared = load(args.shared)
    by_q = {}
    for r in shared:
        by_q[(r.get("patient"), r.get("q_idx"))] = r
    plans = [r for r in flips if r.get("label") == "plan_once"]
    wrong_plans = [(r.get("patient"), r.get("predicted"), r.get("gold"))
                   for r in plans if not r.get("correct")]
    print(f"plan_once rows: {len(plans)}, wrong: {len(wrong_plans)}")
    for p, pred, gold in wrong_plans:
        print(f"  wrong plan: {p} plan={pred} gold={gold}")
    breaks = [r for r in flips if r.get("label") == "breakage"]
    rescues = [r for r in flips if r.get("label") == "rescue"]
    print(f"breakage flips: {len(breaks)}, rescue flips: {len(rescues)}")
    for r in breaks:
        key = (r.get("patient"), r.get("q_idx"))
        srow = by_q.get(key, {})
        steps = srow.get("steps", [])
        i = r.get("flip_step")
        step = steps[i] if isinstance(i, int) and 0 <= i < len(steps) else {}
        verdict = step.get("review", {}).get("verdict", "?")
        has_reviewed = step.get("reviewed") is not None
        print(f"  BREAK {r.get('patient')} q{r.get('q_idx')} step {i} "
              f"plan={r.get('plan_predicted')} flip={r.get('predicted')} "
              f"gold={r.get('gold')} shared_verdict={verdict} "
              f"reviewed_present={has_reviewed}")
    for r in rescues:
        print(f"  RESCUE {r.get('patient')} q{r.get('q_idx')} "
              f"step {r.get('flip_step')} plan={r.get('plan_predicted')} "
              f"flip={r.get('predicted')} gold={r.get('gold')}")


if __name__ == "__main__":
    main()
