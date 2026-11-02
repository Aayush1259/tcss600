#!/usr/bin/env python3
"""Join quote verification with redo and changed labels, shared log.

Reads only. No API calls. No key needed. Writes nothing to disk.
Stdlib only.

Usage:
  python quote_join_27b.py <shared_log.jsonl> <benchmark_v5.json>

Chunk reproduction is the same as quote_check_27b.py: documents
in stored padding order, joined with "\n\n--\n\n", chunked with
chunk_text size 6000 copied from data_loader.py. Chunk count
and padding order are checked per question.

Per step, for out jobs with a non-empty quote and normalised
length at least 20:
  quote_flag_norm_ws = any job fails normalised whitespace match
  quote_flag_norm_punct = any job fails punctuation removed match
  quote_flag_piece = any job fails piece_all match
Redo = review verdict is redo, or reviewed is not None when
verdict is missing. Changed = reviewed jobs differ from out
jobs on at least one chunk, answers normalised by lower case,
punctuation stripped except word characters, space and equals,
whitespace collapsed; accept steps have reviewed None, so
changed is False by definition.

Output is per step lines for steps with at least one testable
job, plus summary flag rates by redo and by changed, for
testable steps only, plus counts for steps with no testable
job and redo among them. No single definition is chosen as
primary here; all three are reported side by side.
"""
import json
import re
import sys


CHUNK_CHARS = 6000


def chunk_text(text, size=CHUNK_CHARS):
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


def norm_ws(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def norm_punct(text):
    return re.sub(r"[^a-z0-9 ]+", "", norm_ws(text))


def norm_answer(text):
    if text is None:
        return "none"
    t = str(text).lower()
    t = re.sub(r"[^\w\s=]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def piece_check(quote, chunk_norm):
    parts = []
    for chunk_part in str(quote).split("..."):
        for line in chunk_part.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if line.startswith("-"):
                line = line[1:].strip()
            norm_line = norm_ws(line)
            if len(norm_line) >= 20:
                parts.append(norm_line)
    if not parts:
        return None
    return all(p in chunk_norm for p in parts)


def main():
    if len(sys.argv) != 3:
        print("usage: quote_join_27b.py <shared_log.jsonl> <benchmark_v5.json>")
        return 2
    shared_path, bench_path = sys.argv[1], sys.argv[2]
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    rows = []
    with open(shared_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    steps_out = []
    chunk_mismatch = 0
    padding_mismatch = 0
    no_testable_steps = 0
    no_testable_redo = 0

    for row in rows:
        patient = row.get("patient")
        qidx = row.get("q_idx")
        pad_n = row.get("padding_n")
        stored_padding = row.get("padding_patients") or []
        try:
            p = bench[patient]
            docs = [p["texts"][k] for k in sorted(p["texts"].keys())]
            for other in stored_padding:
                op = bench[other]
                docs += [op["texts"][k] for k in sorted(op["texts"].keys())]
        except KeyError as e:
            print("ERROR missing bench entry for", patient, "key", e)
            continue
        patients_sorted = sorted(bench.keys())
        own = patients_sorted.index(patient)
        recomputed = []
        if pad_n:
            for j in range(1, pad_n + 1):
                recomputed.append(patients_sorted[(own + j) % len(patients_sorted)])
        if recomputed != stored_padding:
            padding_mismatch += 1
        joined = "\n\n--\n\n".join(docs)
        chunks = chunk_text(joined)
        if len(chunks) != row.get("n_chunks"):
            chunk_mismatch += 1

        for si, step in enumerate(row.get("steps", [])):
            review = step.get("review") or {}
            verdict = review.get("verdict")
            reviewed = step.get("reviewed")
            if verdict is not None:
                redo = (verdict == "redo")
            else:
                redo = (reviewed is not None)
            # Changed: reviewed jobs differ from out jobs on any chunk.
            changed = False
            if reviewed is not None:
                out_by_chunk = {}
                for job in step.get("out", []):
                    out_by_chunk[job.get("chunk_idx")] = norm_answer(job.get("answer"))
                for job in reviewed:
                    if norm_answer(job.get("answer")) != out_by_chunk.get(job.get("chunk_idx"), "none"):
                        changed = True
                        break
            # Quote flags, testable jobs only.
            flags = {"norm_ws": None, "norm_punct": None, "piece": None}
            n_testable = 0
            fails = {"norm_ws": False, "norm_punct": False, "piece": False}
            has = {"norm_ws": False, "norm_punct": False, "piece": False}
            for job in step.get("out", []):
                quote = job.get("quote")
                if quote is None or str(quote).strip() == "":
                    continue
                if len(norm_ws(quote)) < 20:
                    continue
                chunk_idx = job.get("chunk_idx")
                if chunk_idx is None or chunk_idx < 0 or chunk_idx >= len(chunks):
                    continue
                n_testable += 1
                chunk = chunks[chunk_idx]
                chunk_ws = norm_ws(chunk)
                chunk_punct = norm_punct(chunk)
                if norm_ws(quote) not in chunk_ws:
                    fails["norm_ws"] = True
                if norm_punct(quote) not in chunk_punct:
                    fails["norm_punct"] = True
                pc = piece_check(quote, chunk_ws)
                if pc is not None:
                    has["piece"] = True
                    if not pc:
                        fails["piece"] = True
                has["norm_ws"] = True
                has["norm_punct"] = True
            if n_testable == 0:
                no_testable_steps += 1
                if redo:
                    no_testable_redo += 1
                continue
            for key in flags:
                if has[key]:
                    flags[key] = fails[key]
            steps_out.append((patient, si, redo, changed, n_testable, flags))
            print(patient, "step", si, "redo", redo, "changed", changed, "n_testable", n_testable,
                  "quote_flag_norm_ws", flags["norm_ws"], "quote_flag_norm_punct", flags["norm_punct"],
                  "quote_flag_piece", flags["piece"])

    print("SUMMARY testable_steps", len(steps_out), "no_testable_steps", no_testable_steps, "no_testable_redo", no_testable_redo)
    print("SUMMARY chunk_mismatch_questions", chunk_mismatch, "padding_mismatch_questions", padding_mismatch)
    for label, key in (("redo", 2), ("changed", 3)):
        pass
    # Flag rates by redo and by changed, per quote definition.
    for qkey in ("norm_ws", "norm_punct", "piece"):
        for group_name, group_idx, group_val in (("redo", 2, True), ("accept", 2, False), ("changed", 3, True), ("unchanged", 3, False)):
            subset = [s for s in steps_out if s[group_idx] == group_val and s[5][qkey] is not None]
            flagged = sum(1 for s in subset if s[5][qkey])
            print("FLAG_RATE", qkey, group_name, "flagged", flagged, "total", len(subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
