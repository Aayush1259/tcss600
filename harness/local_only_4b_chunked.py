"""4B local-only baseline (chunked) on the frozen question sets.
Machine: laptop (WSL2 Ubuntu), where the qwen3:4b worker lives.

Frozen design (2026-10-17, see docs/decision-4b-local-only-design.md):
the 4B context is 2,048 tokens, so each record is cut into ~1,500-token
document chunks (~6,000 chars at ~4 chars/token, prompt headroom ~500).
One worker call per chunk; final answer = majority vote over the
per-chunk letters; ties broken by the earliest chunk. No supervisor.

Labels on every row: protocol local-only (chunked) | worker qwen3:4b,
ctx 2048, think off | supervisor none | grading JSON-mode vote +
rule-based fallback + hand-check queue.

Oct 18: per-chunk votes use native Ollama JSON mode ({"letter": "B"}).
Free-text prompting made the 4B narrate the excerpt instead of answering
(8/8 chunks ungradable on Oct 17); JSON mode constrains it to emit a
letter. This mirrors the Oct 14 fix where structured_output_schema
repaired the 4B Minion flow.

Setup on the laptop (one time):
  1. Put this file plus data_loader.py, grader.py, runlog.py in one
     folder (e.g. ~/harness/) and run from there.
  2. Copy the benchmark and the frozen ID files from bosgame:
     scp localai@100.81.30.97:~/LongHealth/data/benchmark_v5.json ~/LongHealth/data/
     scp localai@100.81.30.97:~/warmup_ids.json ~/
     (data_loader looks for them at ~/LongHealth/data/benchmark_v5.json,
     ~/warmup_ids.json, ~/eval_ids.json)
  3. Python needs the `ollama` package: pip install ollama

Usage:
    python3 local_only_4b_chunked.py --split warmup --limit 1  # smoke test
    python3 local_only_4b_chunked.py --split warmup            # all 20
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import ollama

import data_loader
import grader
import runlog

WORKER_MODEL = "qwen3:4b"  # frozen 2026-10-17 (ollama list)
NUM_CTX = 2048
CHUNK_CHARS = 6000  # ~1,500 tokens at ~4 chars/token; ~500 tok prompt headroom


def chunk_text(text, size=CHUNK_CHARS):
    """Split text into <=size char chunks at whitespace boundaries."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        start = end
    return chunks


def build_chunk_prompt(rec, chunk):
    """Chunk prompt for JSON-mode voting.

    The 4B worker narrates its reasoning and never states a letter in
    free text (seen Oct 17-18: 16/16 chunks ungradable across two runs,
    even with a "letter first" instruction). JSON mode constrains it to
    emit {"letter": "X"}. (The 27B/cloud prompts use the shared free-text
    wording; both prompts are quoted verbatim in the weekly report.)
    """
    opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
    return (
        "You are answering a multiple-choice medical question using ONLY "
        "the records excerpt below. Reply with a JSON object containing "
        "the letter of the correct option, like {\"letter\": \"B\"}.\n\n"
        f"{rec['question']}\n{opts}\n\nRecords excerpt:\n{chunk}"
    )


def parse_vote(text):
    """Extract the voted letter: JSON mode first, rule-based fallback."""
    try:
        letter = json.loads(text)["letter"].strip().upper().strip("(). ")
        if letter in "ABCDE":
            return letter, "json"
    except Exception:
        pass
    letter, _ = grader.extract_letter_fixed(text or "")
    return letter, "rule-fallback"


def ask_chunk(chunk_prompt, retries=3):
    """One chunk vote, with retry on transport errors.

    Oct 18: the laptop Ollama server dropped a connection mid-run
    ("Server disconnected without sending a response"), killing a 20q
    run at 4/20. Retry with backoff survives transient drops.
    """
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            # NOTE (Oct 17): `think` is a TOP-LEVEL chat parameter. Inside
            # `options` it is silently ignored and the model thinks by default.
            resp = ollama.chat(
                model=WORKER_MODEL,
                messages=[{"role": "user", "content": chunk_prompt}],
                think=False,
                format={"type": "object",
                        "properties": {"letter": {"type": "string"}},
                        "required": ["letter"]},
                options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 200},
            )
            dt = time.time() - t0
            letter, how = parse_vote(resp.message.content)
            return {
                "letter": letter,
                "vote_method": how,
                "text": resp.message.content,
                "prompt_tokens": getattr(resp, "prompt_eval_count", None),
                "completion_tokens": getattr(resp, "eval_count", None),
                "wall_time_s": round(dt, 1),
            }
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    chunk call failed ({type(e).__name__}, "
                  f"attempt {attempt+1}/{retries}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise last_err


def latest_log_for(split):
    """Newest non-regraded log for this split, for resume after a crash."""
    cands = sorted(glob.glob(os.path.join(
        runlog.LOG_DIR, f"local_only_4b_chunked_{split}_*.jsonl")))
    cands = [c for c in cands if "regraded" not in c]
    return cands[-1] if cands else None


def done_ids(log_path):
    done = set()
    if log_path:
        for line in open(log_path):
            try:
                r = json.loads(line)
                done.add((r["patient"], r["q_idx"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = data_loader.load_split(args.split)
    if args.limit:
        recs = recs[:args.limit]

    log_path = latest_log_for(args.split)
    done = done_ids(log_path)
    if log_path and done:
        print(f"resuming {log_path} ({len(done)} already done)")
    else:
        log_path = runlog.new_log(f"local_only_4b_chunked_{args.split}")
        print(f"logging to {log_path}")

    for i, rec in enumerate(recs):
        if (rec["patient"], rec["q_idx"]) in done:
            print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
                  f"already done, skipping", flush=True)
            continue
        full_doc = "\n\n---\n\n".join(rec["documents"])
        chunks = chunk_text(full_doc)
        votes, pt_sum, ct_sum, dt_sum = [], 0, 0, 0.0
        chunk_texts = []
        first_of = {}
        for c in chunks:
            out = ask_chunk(build_chunk_prompt(rec, c))
            chunk_texts.append(out["text"])
            if out["letter"] and out["letter"] not in first_of:
                first_of[out["letter"]] = len(votes)
            votes.append(out["letter"])
            pt_sum += out["prompt_tokens"] or 0
            ct_sum += out["completion_tokens"] or 0
            dt_sum += out["wall_time_s"]

        counts = Counter(v for v in votes if v)
        if counts:
            top = counts.most_common()
            best_n = top[0][1]
            tied = sorted([L for L, n in top if n == best_n],
                          key=lambda L: first_of[L])
            final = tied[0]
            method = f"majority-vote ({len(chunks)} chunks)"
        else:
            final, method = None, "manual-review"
        match = grader.matches(final, rec["gold"], rec["options"])

        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": "local-only (chunked)",
            "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think off",
            "supervisor": None,
            "grading": f"JSON-mode vote + rule fallback + hand-check queue; aggregation {method}",
            "predicted": final,
            "gold": rec["gold"],
            "match": match,
            "needs_handcheck": final is None,
            "n_chunks": len(chunks),
            "chunk_votes": votes,
            "wall_time_s": round(dt_sum, 1),
            "worker_prompt_tokens": pt_sum,
            "worker_completion_tokens": ct_sum,
            "est_cost_usd": 0.0,
            "answer": f"per-chunk votes: {votes}",
            "chunk_answers": chunk_texts,
        })
        print(f"[{i+1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: "
              f"pred={final} gold={rec['gold']} match={match} "
              f"votes={votes} {round(dt_sum,1)}s", flush=True)

    runlog.summarize(log_path)


if __name__ == "__main__":
    main()
