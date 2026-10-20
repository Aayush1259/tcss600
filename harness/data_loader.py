"""Data loader for the Week 3 baselines.

Reads LongHealth benchmark_v5.json and the frozen question-set files,
and produces standard records: patient, question, options, gold answer,
documents.

Frozen decisions used here (2026-10-17):
- Question sets: ~/warmup_ids.json (20), ~/eval_ids.json (200),
  SHA-256 hashes recorded, zero overlap with the 10 pilot questions.
- Padding N = 0: each question uses its own patient's records only.

Machine: bosgame. Assumes benchmark at ~/LongHealth/data/benchmark_v5.json
and the frozen ID files at ~/warmup_ids.json, ~/eval_ids.json.
"""
import json
import os

HOME = os.path.expanduser("~")
BENCH_PATH = os.path.join(HOME, "LongHealth", "data", "benchmark_v5.json")
WARMUP_IDS = os.path.join(HOME, "warmup_ids.json")
EVAL_IDS = os.path.join(HOME, "eval_ids.json")

# Frozen 2026-10-17. Padding is a time-allows extension, not the mainline.
PAD_N = 0


def load_benchmark(path=BENCH_PATH):
    with open(path) as f:
        return json.load(f)


def load_id_list(path):
    """'patient_03 q2' strings -> [('patient_03', 2), ...]."""
    ids = []
    for line in json.load(open(path)):
        p, q = line.split()
        ids.append((p, int(q[1:])))
    return ids


def get_question(bench, patient, qidx, pad_n=PAD_N):
    p = bench[patient]
    q = p["questions"][qidx]
    docs = [p["texts"][k] for k in sorted(p["texts"].keys())]
    if pad_n:
        raise NotImplementedError("padding N>0 is a time-allows extension")
    options = {L: q["answer_" + L.lower()] for L in "ABCDE"}
    return {
        "patient": patient,
        "q_idx": qidx,
        "q_no": q["No"],
        "question": q["question"],
        "options": options,
        "gold": q["correct"],
        "documents": docs,
    }


def load_split(split):
    """split: 'warmup' or 'eval'. Returns list of records."""
    bench = load_benchmark()
    path = WARMUP_IDS if split == "warmup" else EVAL_IDS
    return [get_question(bench, p, i) for p, i in load_id_list(path)]


def build_mcq_prompt(rec, documents=None):
    """Standard multiple-choice prompt. Documents default to the record's own."""
    docs = documents if documents is not None else rec["documents"]
    opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
    doc_text = "\n\n---\n\n".join(docs)
    return (
        f"{rec['question']}\n{opts}\n\n"
        "Answer with the letter of the correct option.\n\n"
        f"Medical records:\n{doc_text}"
    )


if __name__ == "__main__":
    recs = load_split("warmup")
    print(f"warmup records: {len(recs)}")
    r0 = recs[0]
    total_chars = sum(len(d) for d in r0["documents"])
    print(f"first: {r0['patient']} q{r0['q_idx']} (No {r0['q_no']}), "
          f"gold={r0['gold']}, doc chars={total_chars}")
    print("prompt chars:", len(build_mcq_prompt(r0)))
