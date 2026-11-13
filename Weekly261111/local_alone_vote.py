"""4B alone: every chunk answers a letter or "none"; the most common letter wins.

Rules fixed on Nov 9, before any closed-book score (decision log):
- each chunk of the padded record gets the question and the options and
  answers a letter A to E, or "none";
- "none" does not vote. A reply that is not one of these is counted as
  malformed, does not vote either, and is kept for the hand-check;
- the most common letter wins. A tie is broken by a seeded random pick
  among the tied letters, seed 20261109 plus the question's index in the
  frozen question order, so it does not depend on run order;
- a question with no letter vote is scored wrong and never dropped;
- no supervisor, no plan, no review; cloud cost 0.

Every call goes through ollama_guard (truncate and shift off, normal stop,
prompt fits, 3 to 5 characters per token); the run stops at the first
failed check, and every row written before it stays.

Laptop WSL, from ~/pipeline-4b-ctx3072 with ollama_guard.py next to it:
    BASELINE_LOG_DIR=~/baseline_logs_ctx3072 ~/is-env/bin/python local_alone_vote.py --split warmup --dry-run
    BASELINE_LOG_DIR=~/baseline_logs_ctx3072 ~/is-env/bin/python local_alone_vote.py --split warmup --limit 1
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader  # noqa: E402
import grader  # noqa: E402
import ollama_guard  # noqa: E402
import runlog  # noqa: E402

SEED = 20261109
WORKER_MODEL = "qwen3:4b"
NUM_CTX = 3072
NUM_PREDICT = 300
PROTOCOL = "local-alone-per-chunk-vote"
VOTE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D", "E", "none"]}},
    "required": ["answer"],
}


def chunk_prompt(question, options, chunk):
    opts = "\n".join(f"({L}) {options[L]}" for L in "ABCDE")
    return (
        "You are answering a multiple-choice medical question using only this "
        "excerpt of a patient's medical records. If the excerpt does not contain "
        'the answer, reply exactly {"answer": "none"}.\n\n'
        f"Question: {question}\n{opts}\n\n"
        f"Excerpt:\n{chunk}\n\n"
        'Reply with JSON only: {"answer": "X"}, where X is A, B, C, D, E, or "none".'
    )


def parse_vote(text):
    """("letter", L), ("none", None) or ("malformed", None)."""
    try:
        obj = json.loads((text or "").strip())
    except ValueError:
        return "malformed", None
    ans = obj.get("answer") if isinstance(obj, dict) else None
    if isinstance(ans, str):
        a = ans.strip()
        if len(a) == 1 and a.upper() in "ABCDE":
            return "letter", a.upper()
        if a.lower() == "none":
            return "none", None
    return "malformed", None


def vote(letters, q_index):
    """letters: the letter votes only. Returns (winner or None, counts, tied)."""
    counts = dict(sorted(collections.Counter(letters).items()))
    if not counts:
        return None, counts, False
    top = max(counts.values())
    winners = sorted(L for L, c in counts.items() if c == top)
    if len(winners) == 1:
        return winners[0], counts, False
    return random.Random(SEED + q_index).choice(winners), counts, True


def run_question(q_index, rec, post=None):
    chunks = data_loader.chunk_text("\n\n---\n\n".join(rec["documents"]), data_loader.CHUNK_CHARS)
    per_chunk, letters, requests = [], [], 0
    for ci, chunk in enumerate(chunks):
        prompt = chunk_prompt(rec["question"], rec["options"], chunk)
        requests += 1
        try:
            r = ollama_guard.chat(WORKER_MODEL, prompt, NUM_CTX, NUM_PREDICT, fmt=VOTE_SCHEMA, post=post)
        except (ollama_guard.Rejected, ollama_guard.GuardError) as e:
            raise SystemExit(f"HALT at {rec['patient']} q{rec['q_idx']} chunk {ci}: {e}")
        kind, letter = parse_vote(r["text"])
        if kind == "letter":
            letters.append(letter)
        per_chunk.append({"chunk": ci, "vote": kind, "letter": letter, "text": r["text"],
                          "prompt_chars": r["prompt_chars"], "prompt_tokens": r["prompt_tokens"],
                          "completion_tokens": r["completion_tokens"], "done_reason": r["done_reason"],
                          "chars_per_token": r["chars_per_token"], "wall_time_s": r["wall_time_s"]})
    n_none = sum(1 for c in per_chunk if c["vote"] == "none")
    n_bad = sum(1 for c in per_chunk if c["vote"] == "malformed")
    if not (requests == len(chunks) == len(per_chunk) and len(letters) + n_none + n_bad == len(chunks)):
        raise SystemExit(f"HALT: counts do not add up at {rec['patient']} q{rec['q_idx']}")
    pred, counts, tied = vote(letters, q_index)
    return {
        "split": rec.get("split"), "q_index": q_index, "patient": rec["patient"],
        "q_idx": rec["q_idx"], "q_no": rec["q_no"], "protocol": PROTOCOL,
        "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think off, temperature 0",
        "supervisor": None, "grading": "per-chunk vote, JSON answers, grader.matches",
        "padding_n": rec["padding_n"], "padding_patients": rec["padding_patients"],
        "n_chunks": len(chunks), "requests": requests, "votes": counts,
        "n_none": n_none, "n_malformed": n_bad, "tied": tied,
        "tie_seed": SEED + q_index if tied else None, "all_none": pred is None,
        "predicted": pred, "gold": rec["gold"],
        "match": grader.matches(pred, rec["gold"], rec["options"]),
        "needs_handcheck": pred is None or n_bad > 0, "chunks": per_chunk,
    }


def main(argv=None, post=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pad-n", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and the chunk counts; no call")
    args = ap.parse_args(argv)

    recs = data_loader.load_split(args.split, pad_n=args.pad_n)
    todo = list(enumerate(recs))[: args.limit] if args.limit else list(enumerate(recs))
    if args.dry_run:
        for q_index, rec in todo:
            chunks = data_loader.chunk_text("\n\n---\n\n".join(rec["documents"]), data_loader.CHUNK_CHARS)
            longest = max(len(chunk_prompt(rec["question"], rec["options"], c)) for c in chunks)
            print(f"[{q_index}] {rec['patient']} q{rec['q_idx']}: {len(chunks)} chunks, longest prompt {longest} chars")
        first = todo[0][1]
        first_chunk = data_loader.chunk_text("\n\n---\n\n".join(first["documents"]), data_loader.CHUNK_CHARS)[0]
        print("\n--- first prompt ---\n" + chunk_prompt(first["question"], first["options"], first_chunk))
        return None

    log_path = runlog.new_log(f"local_alone_vote_4b_{args.split}")
    print(f"logging to {log_path}; worker {WORKER_MODEL}, ctx {NUM_CTX}; tie seed {SEED} plus question index")
    n = correct = all_none = ties = malformed = 0
    for q_index, rec in todo:
        rec = {**rec, "split": args.split}
        row = run_question(q_index, rec, post=post)
        runlog.log_row(log_path, row)
        n += 1
        correct += row["match"]
        all_none += row["all_none"]
        ties += row["tied"]
        malformed += row["n_malformed"]
        print(f"[{q_index}] {rec['patient']} q{rec['q_idx']}: chunks {row['n_chunks']}, votes {row['votes']}, "
              f"none {row['n_none']}, malformed {row['n_malformed']}, tied {row['tied']}, "
              f"pred {row['predicted']}, gold {row['gold']}, match {row['match']}", flush=True)
    print(f"{log_path}: {correct} of {n} correct; {all_none} all none; {ties} ties; {malformed} malformed chunk replies")
    return log_path


if __name__ == "__main__":
    main()
