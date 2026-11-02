"""Rescued and broken per protocol, from a protocols log, plus oracle.

Reads only, no API, no key. Groups protocols rows by patient, uses
plan_once as the baseline for that question, and counts, per protocol:
  rescued: plan_once wrong, protocol correct
  broken:  plan_once correct, protocol wrong
  neutral: otherwise (same correctness as plan_once)
Also prints the rescued and broken patient lists, and, if an oracle
log is given, the oracle bound per question and its rescued list.

Usage, from any folder on the machine holding the logs:
  python3 rescued_broken_warmup.py --protocols <protocols log>
      [--oracle <oracle log>]
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
    ap.add_argument("--protocols", required=True)
    ap.add_argument("--oracle", default=None)
    args = ap.parse_args()
    rows = load(args.protocols)
    by_q = {}
    for r in rows:
        by_q.setdefault((r.get("patient"), r.get("q_idx")), {})[r.get("protocol")] = r
    plans = {k: v.get("plan_once") for k, v in by_q.items() if v.get("plan_once")}
    print(f"questions with plan_once: {len(plans)}")
    protocols = sorted({r.get("protocol") for r in rows if r.get("protocol") != "plan_once"})
    for name in protocols:
        rescued = broken = neutral = n = correct = reviews = 0
        cloud = 0.0
        rescued_q = []
        broken_q = []
        for key, plan in plans.items():
            row = by_q.get(key, {}).get(name)
            if row is None:
                continue
            n += 1
            correct += 1 if row.get("correct") else 0
            reviews += row.get("n_reviews", 0)
            cloud += row.get("cloud_usd", 0.0)
            if (not plan.get("correct")) and row.get("correct"):
                rescued += 1
                rescued_q.append(key[0])
            elif plan.get("correct") and (not row.get("correct")):
                broken += 1
                broken_q.append(key[0])
            else:
                neutral += 1
        print(f"{name}: n={n} correct={correct} rescued={rescued} "
              f"broken={broken} neutral={neutral} reviews_total={reviews} "
              f"cloud_usd_total={cloud:.4f}")
        if rescued_q:
            print(f"  rescued questions: {', '.join(rescued_q)}")
        if broken_q:
            print(f"  broken questions: {', '.join(broken_q)}")
    if args.oracle:
        orows = load(args.oracle)
        found = sum(1 for r in orows if r.get("found"))
        print(f"oracle: questions={len(orows)} bound_correct={found} "
              f"bound_accuracy={found / max(len(orows), 1):.1%}")
        for r in orows:
            key = (r.get("patient"), r.get("q_idx"))
            plan = plans.get(key)
            if plan is None:
                continue
            if (not plan.get("correct")) and r.get("found"):
                brow = r.get("bound_row") or {}
                print(f"  oracle rescued: {key[0]} set={r.get('set')} "
                      f"bound_predicted={brow.get('predicted')} "
                      f"aggregations={r.get('n_aggregations')}")
            if plan.get("correct") and (not r.get("found")):
                print(f"  oracle still wrong on plan-correct question: {key[0]} "
                      f"(unexpected, check log)")


if __name__ == "__main__":
    main()
