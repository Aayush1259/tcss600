"""Closed-book baseline: the supervisor answers from the question and options alone, with no records.

Purpose: measure how much the cloud model can answer from its own knowledge. The LongHealth patients
are fictional, so a score near chance (5 options, 20 percent) means the answers depend on the records.
My advisor asked for this test first, on the 20 warmup questions, in the November 5 feedback.

I import the call code, grader, cost and logging from cloud_only.py, so the prompt is the only
difference from the cloud-only baseline: same supervisor, thinking level, JSON answer format and
grade_json. The prompt asks for the most likely option when unsure, because a refusal counts as
wrong under the frozen grading rule and would hide what the model knows.

Every prompt is checked for record text before it is sent. A prompt that holds any of the patient's
record stops the run before the call.

Each row records the model version the API reports for that answer. The first answer's version is the
run's version; if a later answer reports a different one, the run stops before that row is written,
so a provider update cannot mix two models in one log.

Labels on every row: protocol closed-book | worker none | supervisor gemini-3.8-flash, thinking low |
grading json + hand-check queue.

Usage, from a folder that also holds cloud_only.py, data_loader.py, grader.py and runlog.py:
    python3 closed_book.py --split warmup --dry-run
        # builds every prompt and checks it, no API call, no key needed
    python3 closed_book.py --split warmup --limit 1 --max-calls 3 --max-usd 0.05
        # one question
    python3 closed_book.py --split warmup --max-calls 30 --max-usd 0.10
        # all 20, resuming the log of the one-question run
Needs GEMINI_API_KEY, except with --dry-run.
"""
import argparse
import glob
import json
import os
import sys
import time

import cloud_only
import data_loader
import grader
import runlog

PROTOCOL = "closed-book"


def build_prompt(rec):
    opts = "\n".join(f"({L}) {rec['options'][L]}" for L in "ABCDE")
    return (
        # Same wording as the lab run of Nov 9 (11 of 20), so warmup and eval match.
        "You are answering a multiple-choice medical question. "
        "No medical records are provided. Answer from the question "
        "and the options only, then respond with JSON only, exactly in "
        "this shape, with X the single letter of the correct option:\n"
        '{"answer": "X"}\n\n'
        f"{rec['question']}\n{opts}"
    )


def record_text_in(prompt, rec):
    """True if the first 200 characters of any of the patient's documents appear in the prompt."""
    for doc in rec["documents"]:
        piece = doc.strip()[:200]
        if piece and piece in prompt:
            return True
    return False


class VersionRecorder:
    """Passes calls to the real client and keeps the last response, so the model version it reports can be logged."""

    def __init__(self, client):
        self._client = client
        self.last = None
        self.models = self

    def generate_content(self, **kwargs):
        self.last = self._client.models.generate_content(**kwargs)
        return self.last


def latest_log_for(split):
    cands = sorted(glob.glob(os.path.join(runlog.LOG_DIR, f"closed_book_{split}_json_*.jsonl")))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup", choices=["warmup", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="start a new log")
    ap.add_argument("--dry-run", action="store_true", help="build and check prompts only, no API call")
    args = ap.parse_args()

    recs = data_loader.load_split(args.split, pad_n=0)
    if args.limit:
        recs = recs[:args.limit]

    if args.dry_run:
        leaks = 0
        for i, rec in enumerate(recs):
            prompt = build_prompt(rec)
            leak = record_text_in(prompt, rec)
            leaks += leak
            if i == 0:
                print("Prompt for the first question, exactly as it would be sent:\n")
                print(prompt)
                print(f"\nIts patient has {len(rec['documents'])} documents, "
                      f"{sum(len(d) for d in rec['documents'])} characters, none of it in the prompt: {not leak}\n")
            print(f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: prompt {len(prompt)} characters, "
                  f"record text in prompt: {leak}, gold {rec['gold']}")
        print(f"\n{len(recs)} prompts built, {leaks} with record text. No API call was made.")
        sys.exit(1 if leaks else 0)

    cloud_only.MAX_CALLS = args.max_calls
    client = VersionRecorder(cloud_only.genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
    run_version = None
    log_path = None if args.fresh else latest_log_for(args.split)
    done = set() if args.fresh else cloud_only.done_ids(log_path)
    if log_path and done:
        print(f"resuming {log_path} ({len(done)} already done)")
        logged = set()
        for line in open(log_path, encoding="utf-8"):
            try:
                logged.add(json.loads(line).get("supervisor_model_version"))
            except ValueError:
                pass
        if len(logged) > 1:
            sys.exit(f"the log already holds more than one model version: {sorted(map(str, logged))}")
        run_version = next(iter(logged)) if logged else None
    else:
        log_path = runlog.new_log(f"closed_book_{args.split}_json")
        print(f"logging to {log_path}")

    spent = 0.0
    for i, rec in enumerate(recs):
        if (rec["patient"], rec["q_idx"]) in done:
            print(f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: already done, skipping", flush=True)
            continue
        if args.max_usd is not None and spent >= args.max_usd:
            print(f"budget cap reached: {spent:.4f} dollars; stopping", flush=True)
            break
        prompt = build_prompt(rec)
        if record_text_in(prompt, rec):
            sys.exit(f"record text found in the prompt for {rec['patient']} q{rec['q_idx']}; nothing sent")
        try:
            out = cloud_only.ask_supervisor(client, prompt)
        except cloud_only.BudgetStop as e:
            print(f"budget cap reached: {e}; stopping", flush=True)
            break
        version = getattr(client.last, "model_version", None)
        if run_version is None:
            run_version = version
        elif version != run_version:
            sys.exit(f"model version changed from {run_version} to {version} at {rec['patient']} q{rec['q_idx']}; "
                     f"that answer was not logged")
        g = grader.grade_json(out["text"], rec["gold"], rec["options"])
        cost = runlog.supervisor_cost_usd(out["prompt_tokens"], out["completion_tokens"], out["thinking_tokens"])
        spent += cost
        runlog.log_row(log_path, {
            "split": args.split,
            "patient": rec["patient"],
            "q_idx": rec["q_idx"],
            "q_no": rec["q_no"],
            "protocol": PROTOCOL,
            "worker": None,
            "supervisor": f"{cloud_only.SUPERVISOR_MODEL}, thinking {cloud_only.THINKING_LEVEL}",
            "supervisor_thinking_level": cloud_only.THINKING_LEVEL,
            "supervisor_model_version": version,
            "grading": f"json ({g['method']}) + hand-check queue",
            "output_format": "json",
            "records_in_prompt": False,
            "predicted": g["letter"],
            "gold": rec["gold"],
            "match": g["match"],
            "needs_handcheck": g["needs_handcheck"],
            "wall_time_s": out["wall_time_s"],
            "supervisor_prompt_tokens": out["prompt_tokens"],
            "supervisor_completion_tokens": out["completion_tokens"],
            "supervisor_thinking_tokens": out["thinking_tokens"],
            "est_cost_usd": round(cost, 6),
            "prompt": prompt,
            "answer": out["text"],
        })
        print(f"[{i + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}: pred={g['letter']} gold={rec['gold']} "
              f"match={g['match']} tok={out['prompt_tokens']}/{out['completion_tokens']}"
              f"(+{out['thinking_tokens']} think) ${cost:.4f} (run total ${spent:.4f}, "
              f"{cloud_only.ATTEMPTS} calls) version {version}", flush=True)
        time.sleep(2)

    runlog.summarize(log_path)


if __name__ == "__main__":
    main()
