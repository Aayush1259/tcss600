"""Protocols over the cached shared parts (proposal v4, sec. 2).

    plan_once:           out[s] for every step                    # 0 reviews
    check_every_step:    reviewed[s] for every step               # K reviews
    triggered_logprob:   reviewed[s] if the log-prob signal flags s else out[s]
    triggered_agreement: reviewed[s] if the agreement signal flags s else out[s]

The ONLY difference between protocols is which steps use the reviewed
answer. Then each protocol makes its own aggregation call over what it
chose (cached, so protocols that choose identically share one call).

Cost accounting, as the proposal states it: a protocol is charged only for
what it actually uses: the plan, the reviews (and reruns) it chose, any
extra local passes for its signal, and its aggregation call. Latency is
the sum of the measured times of those parts, as if run one after another.
The review calls and reruns exist in the cache for every step regardless;
a protocol that does not review step s does not pay for it.
"""
import grader
import pipeline_prompts as P
import runlog
import signals
from shared_parts import usage_of

PROTOCOLS = ("plan_once", "check_every_step", "triggered_logprob",
             "triggered_agreement")


def reviewed_steps(shared, protocol, threshold=None):
    """Indices of the steps this protocol sends to the supervisor."""
    steps = shared["steps"]
    k = len(steps)
    if protocol == "plan_once":
        return ()
    if protocol == "check_every_step":
        return tuple(range(k))
    if protocol == "triggered_logprob":
        if threshold is None:
            raise ValueError("triggered_logprob needs a warmup-calibrated threshold")
        return tuple(i for i, s in enumerate(steps)
                     if signals.logprob_flag(s["logprob_score"], threshold))
    if protocol == "triggered_agreement":
        if any(s["agreement_flag"] is None for s in steps):
            raise ValueError("shared parts were built without agreement passes")
        return tuple(i for i, s in enumerate(steps) if s["agreement_flag"])
    raise ValueError(f"unknown protocol {protocol!r}")


def chosen_steps(shared, reviewed):
    """[(instruction, jobs)]: reviewed jobs for reviewed steps that were
    actually redone, original jobs otherwise (accept == unchanged)."""
    out = []
    for i, s in enumerate(shared["steps"]):
        jobs = s["reviewed"] if (i in reviewed and s["reviewed"] is not None) else s["out"]
        out.append((s["instruction"], jobs))
    return out


def _local(jobs):
    return (len(jobs),
            sum((j["prompt_tokens"] or 0) + (j["completion_tokens"] or 0) for j in jobs),
            sum(j["wall_time_s"] or 0.0 for j in jobs))


def cost_of(shared, reviewed, agg_usage, protocol):
    plan_u = shared["plan"]["usage"]
    review_us = [shared["steps"][i]["review"]["usage"] for i in reviewed]
    cloud = [plan_u, *review_us, agg_usage]
    tin = sum(u["prompt_tokens"] for u in cloud)
    tout = sum(u["completion_tokens"] for u in cloud)
    tthink = sum(u["thinking_tokens"] for u in cloud)

    local_jobs = [j for s in shared["steps"] for j in s["out"]]       # base pass
    local_jobs += [j for i in reviewed if shared["steps"][i]["reviewed"]
                   for j in shared["steps"][i]["reviewed"]]            # reruns
    if protocol == "triggered_agreement":                              # signal cost
        local_jobs += [j for s in shared["steps"]
                       for run in s["agreement_runs"] for j in run]
    passes, ltokens, lsecs = _local(local_jobs)
    cloud_s = sum(u["wall_time_s"] for u in cloud)
    return {
        "cloud_calls": len(cloud),
        "cloud_tokens_in": tin, "cloud_tokens_out": tout,
        "cloud_tokens_think": tthink,
        "cloud_usd": runlog.supervisor_cost_usd(tin, tout, tthink),
        "agg_usd": runlog.supervisor_cost_usd(
            agg_usage["prompt_tokens"], agg_usage["completion_tokens"],
            agg_usage["thinking_tokens"]),
        "local_passes": passes, "local_tokens": ltokens,
        "local_gpu_s": round(lsecs, 3),
        "latency_s": round(cloud_s + lsecs, 3),
    }


def run_protocol(rec, shared, supervisor, protocol, threshold=None,
                 reviewed=None, component="aggregate"):
    """One protocol on one question -> a result row. `reviewed` overrides the
    protocol's own choice (used by the oracle search)."""
    if reviewed is None:
        reviewed = reviewed_steps(shared, protocol, threshold)
    reviewed = tuple(reviewed)
    steps = chosen_steps(shared, reviewed)
    resp = supervisor.generate(
        P.aggregate_prompt(rec["question"], rec["options"], steps),
        component=component, json_mode=True)
    letter, how = P.parse_letter(resp["text"])
    row = {
        "protocol": protocol, "patient": shared["patient"],
        "q_idx": shared["q_idx"], "gold": shared["gold"],
        "predicted": letter, "agg_parse": how,
        "correct": grader.matches(letter, shared["gold"], rec["options"]),
        "needs_handcheck": letter is None,
        "reviewed_steps": list(reviewed), "n_reviews": len(reviewed),
        "n_steps": len(shared["steps"]),
    }
    row.update(cost_of(shared, reviewed, usage_of(resp), protocol))
    return row
