#!/usr/bin/env python3
"""Quote verification diagnostic for shared 27B warmup log.

Reads only. No API calls. No key needed. Writes nothing to disk.
Stdlib only.

Usage:
  python quote_check_27b.py <shared_log.jsonl> <benchmark_v5.json>

What it checks, per out job with a non-empty quote:
  - chunk text is reproduced exactly as in the run:
    documents = own patient texts sorted by key, then padding
    patients in the stored padding_patients order, each with texts
    sorted by key; joined with "\n\n--\n\n"; chunked with
    chunk_text copied from data_loader.py lines 117-128, size 6000.
    Chunk count per question must match the stored n_chunks.
  - exact: raw quote string appears in raw chunk text.
  - norm_ws: lower case, whitespace collapsed to single spaces,
    quote appears in chunk.
  - norm_punct: lower case, whitespace collapsed, punctuation
    removed (only letters, digits, spaces kept), quote appears
    in chunk.
  - piece: quote split on "..." and on newlines, pieces trimmed
    and normalised by whitespace and case; pieces shorter than
    20 normalised characters are ignored; piece_all means every
    kept piece appears in the normalised chunk; piece_frac is
    the fraction of kept pieces that appear.

Quotes with normalised length under 20 characters are counted
separately as too_short, because a short phrase matches by chance.

Empty quote jobs (answer None) are excluded, not counted as fake.

Output is per job lines plus summary counts, plus per step
flag counts for each definition. No single definition is chosen
as primary here; this diagnostic reports all definitions side
by side, so the primary rule can be decided after seeing them.
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
    t = norm_ws(text)
    return re.sub(r"[^a-z0-9 ]+", "", t)


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
        return None, 0, 0
    hits = sum(1 for p in parts if p in chunk_norm)
    return (hits == len(parts)), hits, len(parts)


def main():
    if len(sys.argv) != 3:
        print("usage: quote_check_27b.py <shared_log.jsonl> <benchmark_v5.json>")
        return 2
    shared_path, bench_path = sys.argv[1], sys.argv[2]
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    rows = []
    with open(shared_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total_jobs = 0
    testable = 0
    too_short = 0
    exact_pass = 0
    norm_ws_pass = 0
    norm_punct_pass = 0
    piece_all_pass = 0
    piece_jobs = 0
    chunk_mismatch = 0
    padding_mismatch = 0
    step_stats = []

    for row in rows:
        patient = row.get("patient")
        qidx = row.get("q_idx")
        pad_n = row.get("padding_n")
        stored_padding = row.get("padding_patients") or []
        # Rebuild documents in stored padding order, not recomputed order.
        try:
            p = bench[patient]
            docs = [p["texts"][k] for k in sorted(p["texts"].keys())]
            for other in stored_padding:
                op = bench[other]
                docs += [op["texts"][k] for k in sorted(op["texts"].keys())]
        except KeyError as e:
            print("ERROR missing bench entry for", patient, "key", e)
            continue
        # Check recomputed padding order matches stored order.
        patients_sorted = sorted(bench.keys())
        own = patients_sorted.index(patient)
        recomputed = []
        if pad_n:
            for j in range(1, pad_n + 1):
                recomputed.append(patients_sorted[(own + j) % len(patients_sorted)])
        if recomputed != stored_padding:
            padding_mismatch += 1
            print("PADDING_MISMATCH", patient, "stored", stored_padding, "recomputed", recomputed)
        joined = "\n\n--\n\n".join(docs)
        chunks = chunk_text(joined)
        if len(chunks) != row.get("n_chunks"):
            chunk_mismatch += 1
            print("CHUNK_MISMATCH", patient, "stored_n_chunks", row.get("n_chunks"), "recomputed", len(chunks))

        for si, step in enumerate(row.get("steps", [])):
            step_jobs = []
            for job in step.get("out", []):
                total_jobs += 1
                quote = job.get("quote")
                if quote is None or str(quote).strip() == "":
                    continue
                testable += 1
                chunk_idx = job.get("chunk_idx")
                if chunk_idx is None or chunk_idx < 0 or chunk_idx >= len(chunks):
                    print("BAD_CHUNK_IDX", patient, "step", si, "chunk_idx", chunk_idx, "n_chunks", len(chunks))
                    continue
                chunk = chunks[chunk_idx]
                chunk_ws = norm_ws(chunk)
                chunk_punct = norm_punct(chunk)
                q_ws = norm_ws(quote)
                q_punct = norm_punct(quote)
                if len(q_ws) < 20:
                    too_short += 1
                    print(patient, "step", si, "chunk", chunk_idx, "quote_len", len(str(quote)), "TOO_SHORT")
                    continue
                ex = str(quote) in chunk
                nw = q_ws in chunk_ws
                np_ = q_punct in chunk_punct
                piece_all, hits, nparts = piece_check(quote, chunk_ws)
                if ex:
                    exact_pass += 1
                if nw:
                    norm_ws_pass += 1
                if np_:
                    norm_punct_pass += 1
                if piece_all is not None:
                    piece_jobs += 1
                    if piece_all:
                        piece_all_pass += 1
                step_jobs.append((ex, nw, np_, piece_all))
                print(patient, "step", si, "chunk", chunk_idx, "quote_len", len(str(quote)),
                      "exact", ex, "norm_ws", nw, "norm_punct", np_,
                      "piece_all", piece_all, "piece_hits", hits, "piece_n", nparts)
            if step_jobs:
                # Step flagged if any relevant job fails that definition.
                for name, idx in (("exact", 0), ("norm_ws", 1), ("norm_punct", 2), ("piece_all", 3)):
                    vals = [j[idx] for j in step_jobs if j[idx] is not None]
                    if vals:
                        step_stats.append((patient, si, name, not all(vals), len(step_jobs)))
    print("SUMMARY rows", len(rows), "total_out_jobs", total_jobs, "testable", testable, "too_short", too_short)
    print("SUMMARY exact_pass", exact_pass, "norm_ws_pass", norm_ws_pass, "norm_punct_pass", norm_punct_pass,
          "piece_all_pass", piece_all_pass, "piece_jobs", piece_jobs)
    print("SUMMARY chunk_mismatch_questions", chunk_mismatch, "padding_mismatch_questions", padding_mismatch)
    # Step flag rates per definition, over steps with at least one testable job.
    seen = {}
    for patient, si, name, flagged, njobs in step_stats:
        key = name
        total, flagged_n = seen.get(key, (0, 0))
        seen[key] = (total + 1, flagged_n + (1 if flagged else 0))
    for name in ("exact", "norm_ws", "norm_punct", "piece_all"):
        total, flagged_n = seen.get(name, (0, 0))
        print("STEP_FLAG", name, "steps_with_testable_jobs", total, "flagged_any_fail", flagged_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
