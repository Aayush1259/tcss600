"""27B alone, Option A: one call over the whole padded record, with qwen3.8:27b (November 23).

Prof. Cai's decision of November 23: Option B (local_alone_27b.py) first, then Option A, "with the
automatic stop on any truncated prompt" and "the largest prompt size" reported. The note of
November 23 (note-2026-11-23-27b-alone.md, section 3) fixes the design:

- the prompt is cloud alone's (cloud_only.build_prompt, unchanged and hash checked) with the same
  padded record at N = 2, in one call at a context of 65,536, thinking off, temperature 0, 300 reply
  tokens, "truncate": false (guarded_backend.guarded_chat, unchanged); the reply is read by cloud
  alone's rule (grader.grade_json); a reply with no letter is wrong;
- the reading rule of Amendment 3 section 4 (a): the prompt tokens Ollama reports must equal the
  prompt's length in the model's own vocabulary (the tokenizer in the model's GGUF file, read with
  llama-cpp-python, vocabulary only) plus the fixed chat template overhead measured on a short
  prompt; any difference halts the run before the reply is kept;
- the probe, before any question: the template overhead; the largest prompt of the 200 sent twice
  (both counts equal, and equal to the count plus the overhead); one deliberately oversized prompt
  refused by Ollama. If any of these fails, Option A does not run;
- the call timeout for the run is given on the command line: the probe's longest call times three,
  at least 600 seconds; it is logged on every row;
- the GPU residency check after every call, the GPU lock shared with the other runs, resume, failed
  and halt rows, counts only on the screen; the check reports the largest prompt size.

Changed November 25, after the warmup wr27_warm failed its reply cut check and before any evaluation
call (section 6 of note-2026-11-25-option-a-rules.md): with no reply schema the model wrote its
reasoning first and the JSON answer last, 8 of 20 replies used all 300 tokens, and the strict reading
found a letter in none of the 20. Every call now sends a reply schema, a JSON object with a
"reasoning" field first and an "answer" field last that holds one letter A to E, and the reply budget
is 2,048 tokens. The prompt, the model, thinking off, temperature 0, the strict reading and every
check are unchanged. The warmup check also requires that the strict rule reads every reply.

    ollama stop qwen3.8:27b
    BASELINE_LOG_DIR=~/c27_logs python whole_record_27b.py probe --run-id wr27_probe --gguf BLOB
    BASELINE_LOG_DIR=~/c27_logs python whole_record_27b.py run --split warmup --limit 1 --run-id wr27_trace \
        --gguf BLOB --template-offset K --timeout T
    BASELINE_LOG_DIR=~/c27_logs python whole_record_27b.py check --split warmup --run-id wr27_trace \
        --ollama-journal journal.txt
BLOB is the model file named on the FROM line of `ollama show --modelfile qwen3.8:27b`.
"""
import argparse
import collections
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud_only  # noqa: E402
import grader  # noqa: E402
import guarded_backend as gb  # noqa: E402
import local_alone_27b as la27  # noqa: E402
import ollama_guard  # noqa: E402
import run_shared_c2 as rs  # noqa: E402
import runlog  # noqa: E402

WORKER_MODEL, NUM_CTX, NUM_PREDICT = "qwen3.8:27b", 65536, 2048
# the reply schema (November 25): the reasoning first, then one letter; cloud alone's strict reader
# (grader.grade_json) reads the "answer" field of a JSON object, so it reads this reply unchanged
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                   "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}},
    "required": ["reasoning", "answer"],
}
REPLY_FORMAT = "schema: reasoning, then answer A to E"
LOG_PREFIX = "whole_record_27b_"
PROTOCOL = "27B alone, whole record, one call (Option A)"
MIN_TIMEOUT, PROBE_TIMEOUT = 600, 1800
MAX_OFFSET = 64                 # a chat template adds a few tokens, never more than this
OVERSIZE_MARGIN = 2000          # the oversized prompt is at least this many tokens over the context
SHORT = 'You are answering a test. Respond with JSON only, exactly in this shape: {"answer": "A"}'
MAX_FAILURES_IN_ROW, MAX_FAILURES_TOTAL = 3, 10


class TokenCountHalt(gb.Halt):
    """The prompt tokens Ollama reports differ from the model's own count plus the template overhead."""


class WholeWorker(gb.GuardedOllamaWorker):
    """The guarded worker with the Option A reply schema (REPLY_SCHEMA, since November 25), no
    log-probs and the timeout given for Option A. The residency check is the same: after every call,
    before the reply is used."""

    def call(self, prompt, logprobs=False):
        out = gb.guarded_chat(self.model, prompt, self.num_ctx, self.num_predict, fmt=REPLY_SCHEMA, logprobs=False,
                              post=self.post, retries=self.attempts, sleep=self.sleep, timeout=self.timeout)
        size, vram = self.residency()
        if not vram:
            raise gb.GpuHalt(f"the model is not in GPU memory after this call (size {size}, size_vram {vram})")
        if self.reference is None:
            self.reference = {"model_size": size, "model_size_vram": vram}
        elif (size, vram) != (self.reference["model_size"], self.reference["model_size_vram"]):
            raise gb.GpuHalt(f"GPU residency changed: size {size}, size_vram {vram}; reference "
                             f"size {self.reference['model_size']}, size_vram {self.reference['model_size_vram']}")
        return {**out, "guard": gb.GUARD_TAG, **self.env, "model_size": size, "model_size_vram": vram}


class GgufCounter:
    """A text's length in the model's own vocabulary: the tokenizer stored in the model's GGUF file,
    read by llama-cpp-python with vocab_only (no weights are loaded). No chat template is applied;
    its fixed overhead is measured by the probe.

    The text is counted as the model receives it. Ollama's renderer for this model drops whitespace at
    the end of a message (checked on November 25, before the probe: "alpha beta gamma" with 0, 1 and 2
    final newlines all gave 15 prompt tokens, against an own count of 3, 4 and 4), and every Option A
    prompt ends with a newline, so trailing whitespace is removed before counting. On the 4,350 Option B
    prompts, which have none, Ollama's count was the own count plus 12 on every prompt."""

    def __init__(self, path):
        import llama_cpp
        self.path = path
        self.llm = llama_cpp.Llama(model_path=path, vocab_only=True, verbose=False)
        self.tag = f"gguf vocab {os.path.basename(path)[:19]}, llama-cpp-python {llama_cpp.__version__}"

    def __call__(self, text):
        return len(self.llm.tokenize(text.rstrip().encode("utf-8"), add_bos=False, special=False))


def default_counter(args):
    if not args.gguf or not os.path.isfile(args.gguf):
        raise SystemExit(f"refused: --gguf must name the model file ({args.gguf})")
    try:
        return GgufCounter(args.gguf)
    except ImportError:
        raise SystemExit("refused: llama-cpp-python is needed to count tokens with the model's own vocabulary")


def preflight(split_records):
    problems = rs.frozen_problems(rs.module_hashes())
    if problems:
        return "a frozen module differs from the pushed one: " + "; ".join(problems)
    bad = [(r["patient"], r["q_idx"], rs.record_problems(r)) for r in split_records if rs.record_problems(r)]
    if bad:
        return f"padding check failed before any call: {bad[:5]}"
    return None


def read(path):
    return runlog.read_rows(path) if os.path.exists(path) else []


def call_fields(r):
    return {k: r.get(k) for k in ("prompt_tokens", "completion_tokens", "done_reason", "reply_cut", "attempts",
                                  "chars_per_token", "wall_time_s", "model_size", "model_size_vram")}


# ---------------------------------------------------------------- probe

def cmd_probe(args, make_worker=None, make_counter=None):
    recs = rs.load_records("eval")
    problem = preflight(recs)
    if problem:
        print("refused: " + problem)
        return 5
    lock = rs.take_lock(os.path.join(rs.HERE, ".gpu.lock"))
    if lock:
        print("refused: " + lock)
        return 5
    counter = (make_counter or default_counter)(args)
    worker = (make_worker or (lambda t: WholeWorker(WORKER_MODEL, NUM_CTX, NUM_PREDICT, timeout=t)))(PROBE_TIMEOUT)
    path = os.path.join(runlog.LOG_DIR, f"{LOG_PREFIX}probe_{args.run_id}.jsonl")
    base = {"run_id": args.run_id, "ollama": worker.env, "counter": getattr(counter, "tag", "test counter"),
            "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think off, guarded {gb.GUARD_TAG}", "timeout_s": PROBE_TIMEOUT,
            "reply_format": REPLY_FORMAT, "reply_budget": NUM_PREDICT}
    fails = []

    def check(cond, text):
        print(("PASS  " if cond else "FAIL  ") + text, flush=True)
        if not cond:
            fails.append(text)
        return cond

    def step(name, prompt, **extra):
        try:
            r = worker.call(prompt)
        except (gb.Halt, ollama_guard.GuardError, ollama_guard.Rejected) as e:
            runlog.log_row(path, {**base, **extra, "step": name, "status": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            return None, e
        runlog.log_row(path, {**base, **extra, "step": name, "status": "ok", **call_fields(r)})
        return r, None

    # 1. the chat template overhead, on a short prompt
    r, err = step("template overhead", SHORT, own_tokens=counter(SHORT))
    if not check(r is not None, f"short prompt answered ({err!r})" if err else "short prompt answered"):
        return 1
    offset = r["prompt_tokens"] - counter(SHORT)
    if not check(0 < offset <= MAX_OFFSET, f"template overhead {offset} tokens (Ollama {r['prompt_tokens']}, "
                                           f"own count {counter(SHORT)}; expected 1 to {MAX_OFFSET})"):
        return 1

    # 2. the largest prompt of the 200, sent twice
    sizes = [(counter(cloud_only.build_prompt(rec)), i) for i, rec in enumerate(recs)]
    own, i = max(sizes)
    big = recs[i]
    prompt = cloud_only.build_prompt(big)
    print(f"      largest prompt of the {len(recs)}: {own + offset:,} tokens with the template "
          f"({big['patient']} q{big['q_idx']}); context {NUM_CTX:,}, reply budget {NUM_PREDICT}", flush=True)
    if not check(own + offset + NUM_PREDICT <= NUM_CTX, f"the largest prompt plus {NUM_PREDICT} fits {NUM_CTX:,}"):
        return 1
    extra = {"patient": big["patient"], "q_idx": big["q_idx"], "own_tokens": own, "expected_tokens": own + offset,
             **rs.input_hashes(big)}
    r1, e1 = step("largest prompt, first", prompt, **extra)
    r2, e2 = step("largest prompt, second", prompt, **extra)
    ok = check(r1 is not None and r2 is not None, "the largest prompt answered twice"
               + (f" ({e1 or e2!r})" if (e1 or e2) else ""))
    if ok:
        check(r1["prompt_tokens"] == r2["prompt_tokens"] == own + offset,
              f"Ollama's count {r1['prompt_tokens']} and {r2['prompt_tokens']}, own count plus overhead {own + offset}")

    # 3. one deliberately oversized prompt must be refused
    docs, n = list(big["documents"]), 1
    while counter(cloud_only.build_prompt({**big, "documents": docs})) + offset <= NUM_CTX + OVERSIZE_MARGIN:
        docs, n = docs + list(big["documents"]), n + 1
    over = cloud_only.build_prompt({**big, "documents": docs})
    r3, e3 = step("oversized prompt", over, own_tokens=counter(over), copies_of_record=n)
    check(isinstance(e3, ollama_guard.Rejected),
          f"an oversized prompt of {counter(over) + offset:,} tokens refused by Ollama "
          f"({'refused, HTTP ' + str(e3.status) if isinstance(e3, ollama_guard.Rejected) else 'answered or other: ' + repr(e3 or r3 and call_fields(r3))})")

    if ok:
        longest = max(r1["wall_time_s"], r2["wall_time_s"])
        timeout = max(MIN_TIMEOUT, math.ceil(3 * longest))
        print(f"      longest probe call {longest:.1f} s; run timeout {timeout} s (3 times, at least {MIN_TIMEOUT})")
        print(f"      for the run: --template-offset {offset} --timeout {timeout}; GPU reference "
              f"--expect-size {r2['model_size']} --expect-size-vram {r2['model_size_vram']}")
        runlog.log_row(path, {**base, "step": "result", "status": "pass" if not fails else "fail",
                              "template_offset": offset, "largest_prompt_tokens": own + offset,
                              "longest_call_s": longest, "run_timeout_s": timeout, "failed": fails})
    print(f"\n{'PROBE PASSED' if not fails else 'PROBE FAILED: Option A does not run as designed'}")
    return 0 if not fails else 1


# ---------------------------------------------------------------- run

def run_question(rec, worker, counter, offset):
    prompt = cloud_only.build_prompt(rec)
    own = counter(prompt)
    if own + offset + NUM_PREDICT > NUM_CTX:
        raise TokenCountHalt(f"prompt of {own + offset} tokens plus {NUM_PREDICT} does not fit {NUM_CTX}")
    r = worker.call(prompt)
    if r["prompt_tokens"] != own + offset:
        raise TokenCountHalt(f"Ollama counted {r['prompt_tokens']} prompt tokens, the model's own count plus the "
                             f"template overhead is {own + offset}: the prompt may have been changed or cut")
    g = grader.grade_json(r["text"], rec["gold"], rec["options"])
    return {"patient": rec["patient"], "q_idx": rec["q_idx"], "q_no": rec["q_no"], "protocol": PROTOCOL,
            "own_tokens": own, "expected_tokens": own + offset, **call_fields(r), "answer": r["text"],
            "predicted": g["letter"], "gold": rec["gold"], "match": g["match"],
            "needs_handcheck": g["needs_handcheck"], "grading": f"json ({g['method']})"}


def cmd_run(args, make_worker=None, make_counter=None):
    recs = rs.load_records(args.split)
    if args.limit:
        recs = recs[:args.limit]
    problem = preflight(recs)
    if problem:
        print("refused: " + problem)
        return 5
    path = os.path.join(runlog.LOG_DIR, f"{LOG_PREFIX}{args.split}_{args.run_id}.jsonl")
    lock = rs.take_lock(os.path.join(rs.HERE, ".gpu.lock"))
    if lock:
        print("refused: " + lock)
        return 5
    counter = (make_counter or default_counter)(args)
    worker = (make_worker or (lambda t: WholeWorker(WORKER_MODEL, NUM_CTX, NUM_PREDICT, timeout=t)))(args.timeout)
    if args.expect_size_vram:
        worker.reference = {"model_size": args.expect_size, "model_size_vram": args.expect_size_vram}
    rows = read(path)
    done, failures, final, in_row, total = rs.log_state(rows)
    launch = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    print(f"{'resuming' if rows else 'logging to'} {path}: {len(done)} done, {len(final)} final failures; "
          f"timeout {args.timeout} s, template overhead {args.template_offset}; "
          f"GPU reference {worker.reference or 'from the first call'}", flush=True)
    for q_index, rec in enumerate(recs):
        key = (rec["patient"], rec["q_idx"])
        tag = f"[{q_index + 1}/{len(recs)}] {rec['patient']} q{rec['q_idx']}"
        if key in done or key in final:
            continue
        base = {"split": args.split, "run_id": args.run_id, "launch": launch, "patient": rec["patient"],
                "q_idx": rec["q_idx"], "attempt": failures.get(key, 0) + 1, "ollama": worker.env,
                "worker": f"{WORKER_MODEL}, ctx {NUM_CTX}, think off, guarded {gb.GUARD_TAG}",
                "counter": getattr(counter, "tag", "test counter"), "template_offset": args.template_offset,
                "timeout_s": args.timeout, "reply_format": REPLY_FORMAT, "reply_budget": NUM_PREDICT,
                "padding_n": rec["padding_n"],
                "padding_patients": rec["padding_patients"], **rs.input_hashes(rec)}
        try:
            row = run_question(rec, worker, counter, args.template_offset)
        except (gb.Halt, ollama_guard.GuardError, ollama_guard.Rejected) as e:
            runlog.log_row(path, {**base, "status": "halt", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: HALT ({type(e).__name__}: {rs.scrub(str(e))}). Find the cause before relaunching.")
            return 3
        except (KeyError, TypeError, AttributeError, IndexError, NameError, ZeroDivisionError, ValueError) as e:
            runlog.log_row(path, {**base, "status": "halt", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: HALT (code error {type(e).__name__}). Fix the code first.")
            return 3
        except Exception as e:  # noqa: BLE001
            failures[key] = failures.get(key, 0) + 1
            in_row += 1
            total += 1
            runlog.log_row(path, {**base, "status": "failed", "error_type": type(e).__name__,
                                  "error_message": rs.scrub(str(e))})
            print(f"{tag}: FAILED ({type(e).__name__}); failure {failures[key]} of 2", flush=True)
            if in_row >= MAX_FAILURES_IN_ROW or total > MAX_FAILURES_TOTAL:
                print(f"stopping: {in_row} failures in a row, {total} in this run")
                return 4
            continue
        in_row = 0
        runlog.log_row(path, {**base, **row, "status": "ok"})
        print(f"{tag}: prompt {row['prompt_tokens']:,} tokens (equals the own count plus overhead), reply "
              f"{row['completion_tokens']} tokens, {row['wall_time_s']:.0f} s, letter read "
              f"{'yes' if row['predicted'] else 'no'}", flush=True)
    print("done. Counts only; the log holds the predictions for the analysis.")
    return 0


# ---------------------------------------------------------------- check

def cmd_check(args):
    rows = read(os.path.join(runlog.LOG_DIR, f"{LOG_PREFIX}{args.split}_{args.run_id}.jsonl"))
    ok = [r for r in rows if r.get("status") == "ok"]
    fails = []

    def check(cond, text):
        print(("PASS  " if cond else "FAIL  ") + text)
        if not cond:
            fails.append(text)

    keys = collections.Counter((r["patient"], r["q_idx"]) for r in ok)
    print(f"run {args.run_id}: {len(rows)} rows {dict(collections.Counter(r.get('status') for r in rows))}")
    check(all(n == 1 for n in keys.values()), "no question answered twice")
    if args.expect_questions:
        failed = collections.Counter((r["patient"], r["q_idx"]) for r in rows if r.get("status") == "failed")
        final = {k for k, n in failed.items() if n >= 2 and k not in keys}
        check(len(keys) + len(final) == args.expect_questions,
              f"every question answered or final ({len(keys)} answered, {len(final)} final, of {args.expect_questions})")
    check(all(r["prompt_tokens"] == r["expected_tokens"] for r in ok),
          "every prompt count equals the model's own count plus the template overhead")
    check(len({r.get("template_offset") for r in ok}) <= 1, "one template overhead for the whole run")
    check(all(r.get("timeout_s", 0) >= MIN_TIMEOUT for r in ok), f"every call timeout at least {MIN_TIMEOUT} s")
    cut = [r for r in ok if r["done_reason"] == "length" and r.get("reply_cut") and r.get("completion_tokens") == NUM_PREDICT]
    check(all(r["done_reason"] == "stop" or r in cut for r in ok), "every call a normal stop or a reply cut")
    rate = len(cut) / len(ok) if ok else 0.0
    check(rate <= 0.01, f"reply cuts at {NUM_PREDICT} tokens: {len(cut)} of {len(ok)} calls "
                        f"({100 * rate:.2f} percent, limit 1; amendment 2 section 3)")
    res = collections.Counter((r["model_size"], r["model_size_vram"]) for r in ok)
    check(len(res) == 1 and all(v for _, v in res), f"one GPU residency, never zero {dict(res)}")
    if args.expect_size_vram:
        check(set(res) == {(args.expect_size, args.expect_size_vram)}, "GPU residency equals the reference")
    check(all(r["prompt_tokens"] + NUM_PREDICT <= NUM_CTX for r in ok), f"every prompt plus {NUM_PREDICT} fits {NUM_CTX:,}")
    check(all(r.get("reply_format") == REPLY_FORMAT and r.get("reply_budget") == NUM_PREDICT for r in ok),
          f"every call sent the reply schema with a budget of {NUM_PREDICT} (section 6 of the note of November 25)")
    readable = sum(1 for r in ok if r["predicted"])
    if args.split == "warmup":
        # the warmup gate of section 6: the strict rule reads every reply (counts only, no grade)
        check(readable == len(ok), f"the strict rule reads a letter in every reply: {readable} of {len(ok)}")
    if ok:
        top = max(ok, key=lambda r: r["prompt_tokens"])
        print(f"      largest prompt: {top['prompt_tokens']:,} tokens as Ollama reports it ({top['patient']} "
              f"q{top['q_idx']}); slowest call {max(r['wall_time_s'] for r in ok):.0f} s; median "
              f"{sorted(r['wall_time_s'] for r in ok)[len(ok) // 2]:.0f} s")
    print(f"      replies with no letter (counted wrong) {sum(1 for r in ok if not r['predicted'])}, "
          f"reply cuts {sum(bool(r.get('reply_cut')) for r in ok)}, retried calls {sum((r.get('attempts') or 1) > 1 for r in ok)}")
    if args.ollama_journal:
        calls = [{"attempts": r.get("attempts")} for r in ok]
        la27.journal_checks(args.ollama_journal, [dict(r, n_chunks=1) for r in rows], calls, check)
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
    return 0 if not fails else 1


def main(argv=None, make_worker=None, make_counter=None):
    ap = argparse.ArgumentParser(description="27B alone, whole record in one call (Option A), guarded")
    sub = ap.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--run-id", required=True)
    probe.add_argument("--gguf")
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["warmup", "eval"], required=True)
        p.add_argument("--run-id", required=True)
        p.add_argument("--expect-size", type=int)
        p.add_argument("--expect-size-vram", type=int)
    run = sub.choices["run"]
    run.add_argument("--limit", type=int)
    run.add_argument("--gguf")
    run.add_argument("--template-offset", type=int, required=True, help="from the probe")
    run.add_argument("--timeout", type=int, required=True, help="from the probe: 3 times its longest call, at least 600")
    sub.choices["check"].add_argument("--expect-questions", type=int)
    sub.choices["check"].add_argument("--ollama-journal", help="journalctl -u ollama for the run's window")
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        return cmd_probe(args, make_worker, make_counter)
    if bool(args.expect_size) != bool(args.expect_size_vram):
        ap.error("--expect-size and --expect-size-vram go together")
    if args.cmd == "run":
        if args.timeout < MIN_TIMEOUT:
            ap.error(f"--timeout must be at least {MIN_TIMEOUT} seconds")
        if not 0 < args.template_offset <= MAX_OFFSET:
            ap.error(f"--template-offset must be 1 to {MAX_OFFSET}, as the probe measured it")
        if args.split == "eval" and not args.expect_size_vram:
            ap.error("the eval needs the GPU residency reference: --expect-size and --expect-size-vram")
        return cmd_run(args, make_worker, make_counter)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
