"""Form the record-dependent set: wrong in at least 3 of the 5 closed-book eval runs (amendment of Nov 11).

Usage: python form_set.py --log-dir DIR [--out FILE]
DIR holds the five closed_book_eval_json_*.jsonl logs; FILE defaults to DIR/record_dependent_set.json.
"""
import argparse, glob, json, collections, hashlib, os
ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", required=True, help="folder with the five closed-book eval logs")
ap.add_argument("--out", default=None, help="where to write the set (default: LOG_DIR/record_dependent_set.json)")
args = ap.parse_args()
log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
logs = sorted(glob.glob(os.path.join(log_dir, "closed_book_eval_json_*.jsonl")))
assert len(logs) == 5, f"need exactly 5 logs, found {len(logs)}"
runs = []
for p in logs:
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    d = {(r["patient"], r["q_idx"]): r for r in rows}
    assert len(rows) == 200 and len(d) == 200, f"{p}: not 200 distinct rows"
    runs.append(d)
keys = sorted(runs[0])
assert all(set(d) == set(keys) for d in runs), "logs do not hold the same questions"
wrong = {k: sum(not d[k]["match"] for d in runs) for k in keys}
S = [k for k in keys if wrong[k] >= 3]
print("logs", [p.split("_")[-2] + "_" + p.split("_")[-1][:6] for p in logs])
print("accuracy per run (all 200):", [sum(d[k]["match"] for k in keys) for d in runs])
print("questions by number of wrong runs (0 to 5):", [sum(w == i for w in wrong.values()) for i in range(6)])
print("record-dependent set:", len(S), "of 200")
print("per patient:", dict(sorted(collections.Counter(k[0] for k in S).items())))
print("patients with none in the set:", sorted({k[0] for k in keys} - {k[0] for k in S}))
print("closed-book accuracy on the set, per run:", [sum(d[k]["match"] for k in S) for d in runs], "of", len(S))
print("closed-book accuracy outside the set, per run:", [sum(d[k]["match"] for k in keys if k not in S) for d in runs], "of", 200 - len(S))
out = {"rule": "wrong in at least 3 of 5 closed-book eval runs", "logs": logs,
       "set": [{"patient": p, "q_idx": q, "wrong_runs": wrong[(p, q)]} for p, q in S]}
txt = json.dumps(out, indent=1)
out_path = args.out or os.path.join(log_dir, "record_dependent_set.json")
open(out_path, "w").write(txt)
print("saved", os.path.basename(out_path) + ", sha256", hashlib.sha256(txt.encode()).hexdigest()[:8])
