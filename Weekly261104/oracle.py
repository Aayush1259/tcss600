"""Oracle upper bound (proposal v4, sec. 2).

LongHealth labels only the final answer, so the oracle is defined after
the fact: try sets of reviewed steps in order of cloud cost (size 0, then
1, then 2, then all K, cheapest first within a size) and keep the first
set whose aggregation is correct. If none is, the question counts as
wrong. For K = 6 that is at most 1 + 6 + 15 + 1 = 23 aggregation calls
per question.

    for R in sets of steps (size 0, then 1, then 2, then all K), cheapest first:
        answers = reviewed[s] if s in R else out[s], for every step s
        if SUPERVISOR.aggregate(question, answers) is correct:
            return R
    return none

Two practical notes, both pre-registration items:
- A set whose steps were all ACCEPTED (no rerun happened) chooses exactly
  the same answers as a smaller set already tried, so it is skipped: it
  cannot change the outcome and would only repeat an aggregation call.
- This bounds the review sets it searches, not every possible protocol
  (max_size < K): see the proposal's Oracle paragraph.

search_agg_usd is what the search itself cost in aggregation calls (the
bound's own cost, bound_row, is the winning set run once). Reviews and
reruns are already in the shared parts.
"""
import itertools

import protocols
import runlog


def candidate_sets(shared, max_size=2):
    """Review sets in the order the search tries them."""
    steps = shared["steps"]
    k = len(steps)
    review_usd = [
        runlog.supervisor_cost_usd(
            s["review"]["usage"]["prompt_tokens"],
            s["review"]["usage"]["completion_tokens"],
            s["review"]["usage"]["thinking_tokens"])
        for s in steps]
    sets = [()]
    for size in range(1, min(max_size, k) + 1):
        combos = itertools.combinations(range(k), size)
        sets += sorted(combos, key=lambda c: (sum(review_usd[i] for i in c), c))
    if k > max_size:
        sets.append(tuple(range(k)))
    return sets


def _signature(shared, R):
    """The steps in R that actually changed the answers (were redone)."""
    return tuple(i for i in R if shared["steps"][i]["reviewed"] is not None)


def oracle_search(rec, shared, supervisor, max_size=2):
    seen, tried, search_usd = set(), 0, 0.0
    for R in candidate_sets(shared, max_size):
        sig = _signature(shared, R)
        if sig in seen:
            continue
        seen.add(sig)
        row = protocols.run_protocol(rec, shared, supervisor, "oracle",
                                     reviewed=R, component="oracle_aggregate")
        tried += 1
        search_usd += row["agg_usd"]
        if row["correct"]:
            return {"found": True, "set": list(R), "n_aggregations": tried,
                    "search_agg_usd": search_usd, "bound_row": row,
                    "patient": shared["patient"], "q_idx": shared["q_idx"]}
    return {"found": False, "set": None, "n_aggregations": tried,
            "search_agg_usd": search_usd, "bound_row": None,
            "patient": shared["patient"], "q_idx": shared["q_idx"]}
