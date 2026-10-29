"""Data loader for the Week 3 baselines, extended for Week 4.

Reads LongHealth benchmark_v5.json and the frozen question-set files,
and produces standard records: patient, question, options, gold answer,
documents.

Frozen decisions used here (2026-10-17):
- Question sets: ~/warmup_ids.json (20), ~/eval_ids.json (200),
  SHA-256 hashes recorded, zero overlap with the 10 pilot questions.
- Padding N = 0: each question uses its own patient's records only.

Week 4 freeze (2026-10-24, see frozen-rules-2026-10-24.md):
- Padding is a parameter. Selection is deterministic: the question's
  own record first, then the records of the next N patients in
  benchmark order, wrapping to the start. Padding patients are
  recorded per row so any run is reproducible from its log.
- Prompts can request JSON-only final answers (json_output=True).

Frozen Week 4 data loader, copied verbatim from the bosgame freeze
(2026-10-24) for the laptop 4B MinionS runner. The laptop's own
data_loader.py refuses padding N>0 ("padding N>0 is a time-allows
extension"), so this module supplies the identical padding semantics
without touching it. Paths are built from ~, so they resolve on the
laptop exactly as on bosgame (~/LongHealth/data/benchmark_v5.json,
~/warmup_ids.json, ~/eval_ids.json).
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
    padding_patients = []
    if pad_n:
        # Deterministic padding: the next pad_n patients in benchmark
        # order after the question's own patient, wrapping to the start.
        # Own record stays first; distractors follow in patient order.
        patients = sorted(bench.keys())
        own = patients.index(patient)
        for j in range(1, pad_n + 1):
            other = patients[(own + j) % len(patients)]
            padding_patients.append(other)
            op = bench[other]
            docs += [op["texts"][k] for k in sorted(op["texts"].keys())]
    options = {L: q["answer_" + L.lower()] for L in "ABCDE"}
    return {
        "patient": patient,
        "q_idx": qidx,
        "q_no": q["No"],
        "question": q["question"],
        "options": options,
        "gold": q["correct"],
        "documents": docs,
        "padding_n": pad_n,
        "padding_patients": padding_patients,
    }


def load_split(split, pad_n=PAD_N):
    """split: 'warmup' or 'eval'. Returns list of records."""
    bench = load_benchmark()
    path = WARMUP_IDS if split == "warmup" else EVAL_IDS
    return [get_question(bench, p, i, pad_n=pad_n)
            for p, i in load_id_list(path)]


def build_mcq_prompt(rec, documents=None, json_output=False):
    """Standard multiple-choice prompt. Documents default to the record's own.

    json_output=True requests a JSON-only final answer (Week 4 freeze):
    {"answer": "X"} with X a single letter. Free-text answers are not
    accepted under the frozen JSON rule; see grader.grade_json.
    """
    docs = documents if documents is not None else rec["documents"]
    opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
    doc_text = "\n\n---\n\n".join(docs)
    if json_output:
        answer_line = ('Respond with JSON only, exactly in this shape, '
                       'with X the single letter of the correct option:\n'
                       '{"answer": "X"}\n\n')
    else:
        answer_line = "Answer with the letter of the correct option.\n\n"
    return (
        f"{rec['question']}\n{opts}\n\n"
        f"{answer_line}"
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
