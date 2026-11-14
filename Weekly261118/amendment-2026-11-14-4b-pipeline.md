# Amendment 2 (November 14, 2026): the 4B pipeline, its checks, the threshold, scoring and cost

Pushed with the code it names, before the first evaluation batch of the 4B pipeline.

This amendment changes my pre-registration of November 3 and adds to my amendments of November 11 (record-dependent questions) and November 12 (cloud alone and how "close" is decided). No 4B pipeline evaluation result exists yet. I had seen cloud alone's evaluation cost and accuracy (November 12) before writing it.

## 1. Scope and priority

- As you asked on November 13, the result due on November 20 is plan once and check every step against cloud alone and against 4B alone on the 91 record-dependent questions, with cost; everything else comes after it. The 200 are secondary.
- Question B is exploratory, with the log-prob signal only. Paused for this quarter: the agreement signal and its passes, the random controls, the 8 Holm tests, the 200-question oracle and the threshold sweep. The 27B worker is a supplement; no new 27B run, so the 27B local only evaluation is not run.
- This amendment changes no rule of the November 12 amendment except one, stated in section 3 (an empty reply). The margin, the interval and its seed, the 91 questions, the failure rule and the reading of "close", "worse" and "inconclusive" stay as written. A comparison the interval does not decide is reported as inconclusive, and no criterion is loosened after a result is seen.

## 2. The arms

| arm | role |
|---|---|
| cloud alone | reference (done November 12) |
| plan once, 4B | main |
| check every step, 4B | main |
| triggered log-prob, 4B | exploratory (Question B), appendix |
| 4B alone | reference for the second comparison |
| MinionS, 4B | deferred: after November 20, only if time allows before December 1 |
| closed-book | defines the 91 |

MinionS: my November 4 correction named `use_async=True` as the fix; that was wrong (the library sends all jobs of a round as one conversation). The fix is a per-job client. Its runner, trace and cap go into a later dated amendment before any MinionS call.

4B alone (Section 11 had said it would not run): the per-chunk vote you confirmed on November 12, from `local_alone_vote.py` unchanged, through the guarded calls and the GPU check below; traced on one evaluation question before its run. It is not the same setting as 27B alone (one call over the whole record).

## 3. Checks that decide whether a reply or a question counts

- Worker calls: qwen3:4b, context 3,072, temperature 0, think off, 300-token reply budget, `"truncate": false`. The run halts, and the reply is not stored, on any refused request, abnormal stop, prompt without room for the reply, missing log-probabilities, a cached reply without the run's marks, or a cloud reply with another model version (`guarded_backend.py`).
- A reply that used exactly its 300 tokens is kept and counted. If such replies pass 1 percent of a batch's worker calls, the next batch does not start; the results stay in the analysis with the count reported; the reply budget is never changed on the evaluation set; and I write to you.
- GPU check: after every worker call the run reads how much of the model is in GPU memory and halts if it is zero or differs from the reference: size 3,059,925,389 bytes, 2,316,912,557 in GPU memory (75.7 percent, 27 of 37 layers), Ollama 0.40.3, NVIDIA driver 621.24, fixed from the guarded trace of November 13. If the GPU is lost during the evaluation, rows from the failing call on are set aside, and the run resumes only after a one-question trace matches the reference.
- A crash of Ollama's model process (a CUDA error) that Ollama recovers from by reloading the model is accepted only with proof: in Ollama's log the reload comes before the next answered request, every model load has the reference split, every call matched the GPU reference, and the request that got the error was retried. Each such crash is counted and reported. GPU discovery timeouts in Ollama's log are reported; the GPU check on every call decides. Anything else fails the check. I set this rule after the warmup, which had one such crash (section 5), and before any evaluation run.
- Failures: a failed question is rerun once, then counted wrong. Only connection or API failures make a failed row; a code error halts the run. Every runner stops after 3 failures in a row or more than 10.
- Departure from section 3 of my November 12 amendment: in the pipeline an empty supervisor reply is a failure, rerun once, because the backend never stores it; cloud alone graded empty answers wrong (0 on the evaluation set).

## 4. Batches and caps

One run id for all 200 evaluation questions, in patient order: batches of 50 (`--limit` 50, 100, 150, 200), each checked before the next starts. The caps count every fresh cloud call and dollar of the evaluation over all run ids: shared parts 2.00 dollars and 2,000 calls (the traces show about 0.006 dollars and 7 calls per question); scoring 1.50 dollars and 900 calls. A cap is never raised after the evaluation starts.

## 5. The guarded warmup and Gate 1

`warmup4` ran the 20 warmup questions with the final runner, held to the reference, on November 13 and 14: 20 of 20 answered, no failed or halted question; 1,983 worker calls (1,815 new, 168 replayed from the second trace), all at the reference residency, every prompt plus the reply under 3,072 (largest prompt 2,199 tokens), 1 reply at the 300-token limit (0.05 percent, read as no finding); 88 cloud calls, all gemini-3.8-flash, 0.0793 dollars; 3.4 seconds per worker job (median). Ollama's log: two model loads, both 27 of 37 layers on the GPU; one model process crash (a CUDA error at 00:12) followed by a reload at the same split 21 seconds later, and the one request that got the error was retried; two GPU discovery timeouts with no effect on the residency; no cut prompt. It passes every check of `check_shared.py` under the rule of section 3. Excluded and kept as records: warmup1 (an invalid key), warmup2 (the first reply cut, before the rule above), warmup3 (the driver update; its cache is set aside). Gate 1 passes when this amendment and the code are pushed and a fresh clone matches the laptop's file hashes, the traces of November 13 passed, warmup4 passes every check, 4B alone passed its warmup checks (November 11), and cloud alone's warmup check, which you accepted on November 13.

## 6. The 4B log-prob threshold

Rule unchanged (Sections 3 and 10): a step's score is the mean of the mean chunk log-probabilities over the chunks of its first pass that gave an answer; the threshold is the median of the scored warmup steps; a step is flagged only if its score is strictly below it. From warmup4 (`threshold_c2.py`, which reproduces both earlier values), computed on the laptop with Python 3.11: THRESHOLD_4B = -0.8578214314085831 (rounded -0.8578), 35 of 68 steps scored, 17 flagged; one step's score equals the median and is not flagged. Scoring uses the same Python on the laptop, because a newer Python adds floats differently in the last digit. It is a constant in `score_c3.py`; a live scoring run refuses any other value, and the analysis refuses score rows with more than one. It replaces -0.5289 (cut prompts, voided) and the -0.8749 of my November 11 report (an unguarded run).

## 7. Scoring and analysis

- `score_c3.py` scores plan once, check every step and triggered log-prob through the frozen `protocols.run_protocol` unchanged. Before any evaluation scoring it reproduced the November 2 warmup scores from the cache with no call (a check of the code only; those scores stay voided).
- `analyze_a.py` applies the November 12 rule to plan once and check every step against cloud alone, and reports the same interval against 4B alone with no verdict (no margin was fixed for it). It refuses to run while any arm has open questions. If an arm still has open questions at 12:00 Pacific on November 19, only patients complete in cloud alone, plan once, check every step and 4B alone are analyzed, the same subset for every arm.
- Cloud alone is graded on its JSON answer only; the pipeline arms also accept the rule-based reading of an answer. I report how many answers were read only that way, with the comparison redone counting them wrong.
- Cost: the headline is the protocol accounting ratio (the arm's cloud dollars over cloud alone's on the same questions), with no cutoff, written after I saw cloud alone's cost; the fresh dollars paid for the whole evaluation are one total in the appendix. Time is wall-clock laptop seconds per question, because the model is split between the GPU and the processor. The chart: cloud dollars per question on a linear axis (4B alone at 0) against accuracy on the 91, with each arm's interval and a line at cloud alone minus 10 points. Who does the work: the correct answers whose steps had no worker finding.
- The drift check of my November 12 amendment (cloud alone on 20 of the 91) runs after November 20, as planned.

## 8. Limitations added

- Each pipeline arm runs once; the cloud model's plan for the same question can differ between runs (5 and 6 steps in my two traces), so a result near the margin could change on a rerun.
- The NVIDIA driver changed on November 13 (621.14 to 621.24); every 4B run from the second trace on uses the new one.

## 9. Files

In `Weekly261118/`: `guarded_backend.py`, `run_shared_c2.py`, `check_shared.py`, `threshold_c2.py`, `score_c3.py`, `local_alone_eval.py`, `analyze_a.py` and this file, with copies of the frozen helper modules they import. Their fake-backend tests are kept with my records, not pushed.
