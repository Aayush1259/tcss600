"""The two training-free uncertainty signals (proposal v4, sec. 2).

Log-probability: among the chunks where the worker did NOT answer "none",
take each chunk's mean token log-probability; the step's score is the
LOWEST of those. A step is flagged if the score falls below a threshold
set on the warmup questions so that about half of the steps are flagged.
It needs no extra local pass (Ollama returns log-probabilities with each
answer).

Agreement: the worker reruns the step with two paraphrased instructions;
the step is flagged if any chunk's answer differs across the three runs
(a simple form of the idea behind semantic entropy, Farquhar et al.
2024). It costs two extra local passes per chunk; that cost is counted in
latency and local compute.

Design points the proposal leaves open, implemented literally and listed
here so they can be pinned in the Week 5 pre-registration:
  1. A step where EVERY chunk answered "none" has no eligible log-prob, so
     the log-prob signal never flags it (flag_if_no_answer=False). The
     mock world's q_hidden shows the consequence: a missed-evidence step
     is invisible to this signal.
  2. The mean log-probability averages over all generated tokens, JSON
     scaffolding included, which dilutes it toward 0. The threshold is
     calibrated on the same quantity, so the ranking is unaffected, but
     absolute values are not comparable across prompt formats.
  3. "Differs" is exact equality after normalization (case, punctuation,
     whitespace). Looser similarity would flag less.
"""
import re


def logprob_score(jobs):
    """Lowest mean log-prob over chunks with an answer; None if no such chunk."""
    vals = [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]
    return min(vals) if vals else None


def logprob_flag(score, threshold, flag_if_no_answer=False):
    if score is None:
        return flag_if_no_answer
    return score < threshold


def quantile(values, q):
    """Linear-interpolated quantile of a non-empty list (q in [0, 1])."""
    xs = sorted(values)
    if not xs:
        raise ValueError("quantile of empty list")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def calibrate_threshold(scores, flag_rate=0.5):
    """Threshold such that about `flag_rate` of the scored steps are flagged.

    `scores` are step scores from the warmup questions; steps with no score
    (all chunks 'none') are ignored here and are never flagged later.
    """
    vals = [s for s in scores if s is not None]
    if not vals:
        raise ValueError("no warmup step had a log-prob score; cannot calibrate")
    return quantile(vals, flag_rate)


def _norm(answer):
    if answer is None:
        return "none"
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s=]", "", answer.lower())).strip()


def agreement_flag(out_jobs, extra_runs):
    """True if any chunk's answer differs across the original run and the
    paraphrase runs. `extra_runs` is a list of job lists, same chunk order."""
    runs = [out_jobs] + list(extra_runs)
    for i in range(len(out_jobs)):
        answers = {_norm(run[i].get("answer")) for run in runs}
        if len(answers) > 1:
            return True
    return False
