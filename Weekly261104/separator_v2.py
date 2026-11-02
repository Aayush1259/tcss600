"""Separator test v2: log-prob and agreement, both labels, cluster bootstrap.

Updated for Cai feedback Oct 29, 2026. Supersedes separator_4b_prelim.py
for any result that will be cited in a report or in pre-registration.
Do NOT run the prelim script as a final result: it covers log-prob only,
redo label only, and no confidence intervals.

Input: one shared-parts log (run_pipeline.py shared output), one row
per warmup question. No new API calls, no key needed, stdlib only.

Per step, from the step's OUT jobs (eligible chunks: answer present
and mean_logprob present):
  mean_agg = mean of the eligible chunk mean_logprobs
  min_agg  = min  of the eligible chunk mean_logprobs
  stored logprob_score should equal min_agg; mismatches are reported.

Signals:
  log-prob, mean and min, lower score means more uncertain.
  agreement, from the shared log: agreement_flag is True when any chunk
  answer differs across the original run and the two paraphrase runs.
  The script recomputes the flag from out plus agreement_runs with the
  same normalization as signals.py, and reports mismatches.

Labels, reported separately and never mixed:
  redo label: review verdict is "redo" (supervisor asked for a redo).
    This is the original separator label. A redo request does not mean
    the step was actually wrong, because the supervisor can be wrong.
  changed label: for redo steps, the reviewed jobs differ from the out
    jobs on at least one chunk (normalized answers compared per chunk).
    For accept steps, reviewed is None, so changed is False by definition.
    This is a step-level before/after label, computed directly from the
    cached jobs, with no new API calls.
  Final-answer wrong-to-right label: NOT computed here, because the
    shared log holds no aggregation letters. A final answer comes from
    a supervisor aggregation over chosen step jobs, so a true
    wrong-to-right per step needs aggregation calls (plan_once plus
    single-step flips, via evaluate/oracle). That step is separate,
    needs a key, and must be discussed before it runs. Do not present
    the changed label as a final-answer wrong-to-right label.

AUCs:
  log-prob AUC = P(redo score < accept score) + 0.5 * P(tie).
  agreement AUC = P(flagged redo > flagged accept) + 0.5 * P(tie),
    which for a binary flag equals (sensitivity + specificity) / 2.
  Both AUCs are also computed against the changed label, for comparison.

Cluster bootstrap:
  Questions are resampled with replacement, same count as the input,
  1000 replicates, fixed seed 20261030. All steps from a resampled
  question move together, because steps in one question are not
  independent. Reported interval is the 2.5th to 97.5th percentile of
  the bootstrap AUCs. Replicates where one label side is empty are
  skipped and counted. With only 20 warmup questions, intervals will
  be wide; that width is the finding, not a defect to hide. State the
  n=20 instability in any report that cites these numbers.

Thresholds, warmup only:
  log-prob mean and min thresholds are the median (quantile 0.5, linear
  interpolation, same rule as signals.calibrate_threshold), so about
  half of scored steps are flagged. Actual flag rate at that threshold
  is reported, because ties and interpolation can shift it.
  Agreement has no threshold: flagged means agreement_flag is True.

Winner rule, mean vs min only, against the redo label:
  If one AUC is higher, name it. If they tie, use mean, per the Oct 28
  Decisions tie rule. If both AUCs are below 0.5, report a negative
  finding: neither score separates redo steps from accepted steps on
  this log. Do not treat mean as a separator in that case, even if
  mean is higher than min. Agreement AUC is reported descriptively
  and is not forced into the mean vs min winner rule, because it has
  a different cost (two extra local passes per chunk).

Freeze note:
  A result on one worker alone is interim, not a freeze. The freeze
  decision for section 2 needs the same script run on both completed
  warmup shared logs (27B and 4B), then a combined read. Do not freeze
  on a single-worker result.

Usage:
  python3 separator_v2.py /home/localai/baseline_logs/shared_27b_warmup_20261028_015729.jsonl
"""

import json
import random
import re
import sys


def _norm(answer):
    if answer is None:
        return "none"
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s=]", "", answer.lower())).strip()


def _eligible(jobs):
    return [j["mean_logprob"] for j in jobs
            if j.get("answer") and j.get("mean_logprob") is not None]


def _agreement_recomputed(out_jobs, extra_runs):
    runs = [out_jobs] + list(extra_runs or [])
    if len(runs) < 2:
        return None
    for i in range(len(out_jobs)):
        answers = {_norm(run[i].get("answer")) if i < len(run) else "none"
                   for run in runs}
        if len(answers) > 1:
            return True
    return False


def _changed(out_jobs, reviewed_jobs):
    if reviewed_jobs is None:
        return False
    for i, job in enumerate(out_jobs):
        other = reviewed_jobs[i] if i < len(reviewed_jobs) else None
        other_ans = other.get("answer") if other else None
        if _norm(job.get("answer")) != _norm(other_ans):
            return True
    return False


def auc_lower(pos_scores, neg_scores):
    """P(pos < neg) + 0.5 * P(tie); None if either side is empty."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p < n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def auc_higher(pos_scores, neg_scores):
    """P(pos > neg) + 0.5 * P(tie); None if either side is empty."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def quantile(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("quantile of empty list")
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _fmt(value):
    return "None" if value is None else f"{value:.4f}"


def main():
    if len(sys.argv) != 2:
        print("usage: python3 separator_v2.py <shared log path>")
        sys.exit(2)
    path = sys.argv[1]
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Per-question step records, grouped for the cluster bootstrap.
    # Each record: (mean, min, agreement 0/1/None, redo 0/1, changed 0/1)
    by_question = []
    n_skipped = 0
    n_agreement_missing = 0
    n_logprob_mismatch = 0
    n_agreement_mismatch = 0
    n_redo_no_reviewed = 0
    for r in rows:
        worker = str(r.get("worker", ""))
        if worker.startswith("MOCK"):
            continue
        qid = f"{r.get('patient')} q{r.get('q_idx')}"
        q_steps = []
        for s in r.get("steps", []):
            vals = _eligible(s.get("out", []))
            verdict = s.get("review", {}).get("verdict", "?")
            redo = 1 if verdict == "redo" else 0
            if not vals:
                n_skipped += 1
                q_steps.append((qid, None, None, None, redo, 0))
                continue
            mean_agg = sum(vals) / len(vals)
            min_agg = min(vals)
            stored = s.get("logprob_score")
            if stored is not None and abs(stored - min_agg) > 1e-9:
                n_logprob_mismatch += 1
            flag = s.get("agreement_flag")
            if flag is None:
                n_agreement_missing += 1
                agree = None
            else:
                agree = 1 if flag else 0
                recomputed = _agreement_recomputed(s.get("out", []),
                                                   s.get("agreement_runs", []))
                if recomputed is not None and bool(recomputed) != bool(flag):
                    n_agreement_mismatch += 1
            reviewed = s.get("reviewed")
            if redo and reviewed is None:
                n_redo_no_reviewed += 1
            changed = 1 if _changed(s.get("out", []), reviewed) else 0
            q_steps.append((qid, mean_agg, min_agg, agree, redo, changed))
        by_question.append(q_steps)

    n_questions = len(by_question)
    all_steps = [st for q in by_question for st in q]
    scored = [st for st in all_steps if st[1] is not None]
    n_redo = sum(st[4] for st in scored)
    n_accept = len(scored) - n_redo
    n_changed = sum(st[5] for st in scored)
    n_agree_flagged = sum(1 for st in scored if st[3] == 1)
    n_agree_scored = sum(1 for st in scored if st[3] is not None)

    def aucs(steps):
        mean_pos = [st[1] for st in steps if st[1] is not None and st[4] == 1]
        mean_neg = [st[1] for st in steps if st[1] is not None and st[4] == 0]
        min_pos = [st[2] for st in steps if st[2] is not None and st[4] == 1]
        min_neg = [st[2] for st in steps if st[2] is not None and st[4] == 0]
        agr_pos = [st[3] for st in steps if st[3] is not None and st[4] == 1]
        agr_neg = [st[3] for st in steps if st[3] is not None and st[4] == 0]
        ch_mean_pos = [st[1] for st in steps if st[1] is not None and st[5] == 1]
        ch_mean_neg = [st[1] for st in steps if st[1] is not None and st[5] == 0]
        ch_min_pos = [st[2] for st in steps if st[2] is not None and st[5] == 1]
        ch_min_neg = [st[2] for st in steps if st[2] is not None and st[5] == 0]
        ch_agr_pos = [st[3] for st in steps if st[3] is not None and st[5] == 1]
        ch_agr_neg = [st[3] for st in steps if st[3] is not None and st[5] == 0]
        return {
            "mean_redo": auc_lower(mean_pos, mean_neg),
            "min_redo": auc_lower(min_pos, min_neg),
            "agr_redo": auc_higher(agr_pos, agr_neg),
            "mean_changed": auc_lower(ch_mean_pos, ch_mean_neg),
            "min_changed": auc_lower(ch_min_pos, ch_min_neg),
            "agr_changed": auc_higher(ch_agr_pos, ch_agr_neg),
        }

    base = aucs(scored)

    # Cluster bootstrap by question, fixed seed for reproducibility.
    rng = random.Random(20261030)
    boot = {k: [] for k in base}
    boot_skipped = {k: 0 for k in base}
    for _ in range(1000):
        sample = []
        for _ in range(n_questions):
            sample.extend(rng.choice(by_question))
        vals = aucs(sample)
        for k, v in vals.items():
            if v is None:
                boot_skipped[k] += 1
            else:
                boot[k].append(v)

    def ci(values):
        if not values:
            return None
        xs = sorted(values)
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        return (lo, hi)

    print(f"Separator test v2: {n_questions} questions, {len(all_steps)} steps, "
          f"{len(scored)} scored, {n_skipped} skipped (no eligible chunk)")
    print(f"Redo steps: {n_redo}, accepted steps: {n_accept}, "
          f"changed steps: {n_changed}")
    print(f"Agreement flagged: {n_agree_flagged} of {n_agree_scored} scored "
          f"steps with an agreement flag, {n_agreement_missing} missing")
    print(f"Sanity checks: logprob_score vs recomputed min mismatches: "
          f"{n_logprob_mismatch}, agreement_flag vs recomputed mismatches: "
          f"{n_agreement_mismatch}, redo steps with reviewed None: "
          f"{n_redo_no_reviewed}")
    print("Redo steps by question (clustering check):")
    for q in by_question:
        rc = sum(st[4] for st in q if st[1] is not None)
        if rc:
            print(f"  {q[0][0] if q else '?'}: {rc} redo of "
                  f"{sum(1 for st in q if st[1] is not None)} scored steps")
    print()
    print("AUCs against redo label (supervisor asked for a redo):")
    for key in ("mean_redo", "min_redo", "agr_redo"):
        interval = ci(boot[key])
        lo = "None" if interval is None else f"{interval[0]:.4f}"
        hi = "None" if interval is None else f"{interval[1]:.4f}"
        print(f"  {key}: {_fmt(base[key])}  cluster-bootstrap 95% CI "
              f"[{lo}, {hi}]  skipped replicates: {boot_skipped[key]}")
    print("AUCs against changed label (reviewed jobs differ from out jobs):")
    for key in ("mean_changed", "min_changed", "agr_changed"):
        interval = ci(boot[key])
        lo = "None" if interval is None else f"{interval[0]:.4f}"
        hi = "None" if interval is None else f"{interval[1]:.4f}"
        print(f"  {key}: {_fmt(base[key])}  cluster-bootstrap 95% CI "
              f"[{lo}, {hi}]  skipped replicates: {boot_skipped[key]}")
    print()
    # Thresholds at flag rate 0.5, warmup only.
    mean_vals = [st[1] for st in scored]
    min_vals = [st[2] for st in scored]
    if mean_vals:
        thr = quantile(mean_vals, 0.5)
        flagged = sum(1 for v in mean_vals if v < thr)
        redo_flagged = sum(1 for st in scored if st[1] < thr and st[4] == 1)
        print(f"Mean threshold at median: {thr:.4f}, flagged "
              f"{flagged}/{len(mean_vals)} scored steps "
              f"({flagged / len(mean_vals):.1%}), redo steps flagged "
              f"{redo_flagged}/{n_redo}")
    if min_vals:
        thr = quantile(min_vals, 0.5)
        flagged = sum(1 for v in min_vals if v < thr)
        redo_flagged = sum(1 for st in scored if st[2] < thr and st[4] == 1)
        print(f"Min threshold at median: {thr:.4f}, flagged "
              f"{flagged}/{len(min_vals)} scored steps "
              f"({flagged / len(min_vals):.1%}), redo steps flagged "
              f"{redo_flagged}/{n_redo}")
    if n_agree_scored:
        agr_redo_flagged = sum(1 for st in scored if st[3] == 1 and st[4] == 1)
        print(f"Agreement flag rate: {n_agree_flagged}/{n_agree_scored} "
              f"({n_agree_flagged / n_agree_scored:.1%}), redo steps flagged "
              f"{agr_redo_flagged}/{n_redo}")
    print()
    mean_auc = base["mean_redo"]
    min_auc = base["min_redo"]
    if mean_auc is None or min_auc is None:
        print("Winner (mean vs min, redo label): undecided (one side empty)")
    elif mean_auc < 0.5 and min_auc < 0.5:
        print("Winner (mean vs min, redo label): negative finding, both "
              "AUCs below 0.5, neither score separates redo steps here")
    elif mean_auc > min_auc:
        print("Winner (mean vs min, redo label): mean-of-chunk-means")
    elif min_auc > mean_auc:
        print("Winner (mean vs min, redo label): min-of-chunk-means")
    else:
        print("Winner (mean vs min, redo label): tie, use "
              "mean-of-chunk-means per the Oct 28 Decisions tie rule")
    print("Note: this result is interim if it covers one worker only. "
          "Freeze section 2 only after the same script runs on both "
          "completed warmup shared logs, and the n=20 instability is "
          "stated in any report that cites these numbers.")
    print()
    print("Per-step table (question, step, mean, min, agreement, redo, changed):")
    for q in by_question:
        for idx, st in enumerate(q):
            m = "None" if st[1] is None else f"{st[1]:.4f}"
            n = "None" if st[2] is None else f"{st[2]:.4f}"
            a = "None" if st[3] is None else str(st[3])
            print(f"{st[0]} step {idx + 1}: mean={m} min={n} "
                  f"agreement={a} redo={st[4]} changed={st[5]}")


if __name__ == "__main__":
    main()
