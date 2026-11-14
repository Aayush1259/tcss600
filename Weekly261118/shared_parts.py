"""Shared parts, computed once per question and cached (proposal v4, sec. 2).

    plan = SUPERVISOR.plan(question)            # K <= 6 subtasks, 1 cloud call
    for s in plan:                              # one step = one subtask
        out[s]      = WORKER.run(s, chunks)     # one local job per chunk
        note[s]     = SUPERVISOR.review(s, out[s])   # sees answers, not chunks
        reviewed[s] = WORKER.run(note[s], chunks) if note[s] is REDO else out[s]

Everything every protocol might need is computed up front, so protocols
are judged on IDENTICAL outputs (protocols.py only chooses between
out[s] and reviewed[s]). Both signals are precomputed per step too:
the log-prob score from out[s], and the agreement flag from two extra
paraphrase passes. Each protocol is charged only for what it uses.

run_shared() returns a JSON-safe dict (written to the shared-parts log):

    {"patient","q_idx","gold","question","options","n_chunks",
     "plan": {"steps":[str], "fallback":bool, "usage":{...}},
     "steps": [{"instruction", "out":[job], "review":{"verdict","instruction",
                "ok","usage"}, "reviewed":[job]|None, "logprob_score",
                "agreement_flag", "agreement_runs":[[job],[job]]}]}

A job: {"chunk_idx","answer"|None,"quote","parse_ok","mean_logprob",
        "prompt_tokens","completion_tokens","wall_time_s"}.
`reviewed` is None when the verdict was accept (identical to `out`).
"""
import pipeline_prompts as P
import signals

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "thinking_tokens",
              "wall_time_s")


def usage_of(resp):
    return {k: resp.get(k) or 0 for k in USAGE_KEYS}


def run_job(worker, instruction, question, chunk, chunk_idx):
    resp = worker.generate(P.job_prompt(instruction, question, chunk),
                           component="worker", logprobs=True)
    answer, quote, ok = P.parse_job(resp["text"])
    return {"chunk_idx": chunk_idx, "answer": answer, "quote": quote,
            "parse_ok": ok, "mean_logprob": resp.get("mean_logprob"),
            "prompt_tokens": resp.get("prompt_tokens") or 0,
            "completion_tokens": resp.get("completion_tokens") or 0,
            "wall_time_s": resp.get("wall_time_s") or 0.0}


def run_step_jobs(worker, instruction, question, chunks):
    return [run_job(worker, instruction, question, c, i)
            for i, c in enumerate(chunks)]


def run_shared(rec, worker, supervisor, chunk_fn, paraphrases=P.PARAPHRASES,
               with_agreement=True):
    """worker/supervisor are CachedBackends; chunk_fn splits the joined record."""
    question, options = rec["question"], rec["options"]
    chunks = chunk_fn("\n\n---\n\n".join(rec["documents"]))

    presp = supervisor.generate(P.plan_prompt(question, options),
                                component="plan", json_mode=True)
    steps_text, fallback = P.parse_plan(presp["text"], question)

    steps = []
    for instr in steps_text:
        out = run_step_jobs(worker, instr, question, chunks)
        rresp = supervisor.generate(P.review_prompt(question, instr, out),
                                    component="review", json_mode=True)
        verdict, new_instr, ok = P.parse_review(rresp["text"])
        reviewed = (run_step_jobs(worker, new_instr, question, chunks)
                    if verdict == "redo" else None)
        runs = []
        if with_agreement:
            runs = [run_step_jobs(worker, t.format(instr=instr), question, chunks)
                    for t in paraphrases]
        steps.append({
            "instruction": instr,
            "out": out,
            "review": {"verdict": verdict, "instruction": new_instr, "ok": ok,
                       "usage": usage_of(rresp)},
            "reviewed": reviewed,
            "logprob_score": signals.logprob_score(out),
            "agreement_flag": (signals.agreement_flag(out, runs)
                               if with_agreement else None),
            "agreement_runs": runs,
        })
    return {"patient": rec["patient"], "q_idx": rec["q_idx"],
            "gold": rec["gold"], "question": question, "options": options,
            "n_chunks": len(chunks),
            "padding_n": rec.get("padding_n", 0),
            "padding_patients": rec.get("padding_patients", []),
            "plan": {"steps": steps_text, "fallback": fallback,
                     "usage": usage_of(presp)},
            "steps": steps}
