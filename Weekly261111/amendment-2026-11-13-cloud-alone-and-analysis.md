# Amendment to my pre-registration of November 3, 2026: cloud alone on the evaluation set, failed questions, and how "close" is decided

Dated Friday, November 13, 2026. I wrote it on November 12 and pushed it before any cloud alone call on the evaluation set. When it is pushed, no cloud alone answer on the evaluation set exists, and no protocol has been scored on the evaluation set.

## Why

My amendment of November 11 made the record-dependent questions the primary analysis, and cloud alone is the reference of that comparison. Before its evaluation run, three things were not yet fixed in writing: how the model version is recorded when the cloud arms run on different days, what happens to a question that fails, and how "close to cloud alone" is decided. On November 12 I proposed four rules to Prof. Cai by email. He agreed to all four the same day and asked that some comparisons be reported as inconclusive if the intervals do not settle them, and that the criteria not be loosened after seeing the results. Sections 3 to 5 are those rules.

## 1. Cloud alone code

- Cloud alone on the evaluation set runs with `cloud_alone.py` (SHA-256 prefix 81fe1466). It imports the frozen `cloud_only.py` (prefix 1e5cdd18, unchanged) for the prompt, the call code with its retries, the supervisor (gemini-3.8-flash) and the thinking level (low), so every request is the one the frozen script sends. Its 17 fake-client tests are in `test_cloud_alone.py`.
- Padding is fixed at 2, the frozen setting, with no flag. Before any call, every record is checked: padding 2, two distinct padding patients, never the question's own patient.
- The model version the API reports is logged on every row. A run stops before writing the row if the version is missing, is not gemini-3.8-flash, or changes within the run. A log is resumed only if it holds one version, padding 2 on every row, and rows written by the same code.
- Two seconds between calls instead of fifteen. The fifteen seconds were for the free tier; the account is prepay now, and a rate-limit reply is still retried as frozen. The pause does not change any answer.
- Every row records the SHA-256 of the script and of the four frozen modules, the google-genai and Python versions, and the machine.
- Caps: the frozen cap for cloud alone stays (Section 11: 6 dollars and 300 calls). I run with at most 6 dollars and 240 calls, counted over the whole run including any resume. If a cap is reached, the run stops and I report it; a higher cap needs a dated change before the run goes on, as Section 11 says.

Checks before this push, none of them an evaluation call:

- The 20 warmup prompts the new script builds are identical, byte for byte, to the prompts of the frozen run of October 24 on the lab machine.
- One warmup question with the new script on the laptop (November 12): the same prompt and the same 36,187 prompt tokens as on October 24, and the same answer. The first launch of that trace failed with an invalid API key (not billed); the script wrote a failure row, and the next launch asked the question once more, as Section 3 below says. Before the evaluation run I check the key with a call that costs nothing.
- A dry run of the 200 evaluation prompts: every padding check passes; projected cost 5.24 dollars at list price.

## 2. Cloud alone on warmup

The warmup result stays 19 of 20, from the frozen script on October 24, on the lab machine. The model version was not recorded then. I do not rerun it; the warmup is not part of the primary comparison.

## 3. Failed questions (agreed November 12)

For every arm on the evaluation set, this replaces the part of Section 11 that reruns a question until it completes:

- A question whose call ends in an error gets a failure row. After I find the cause, it is rerun once. If it fails again, it counts as failed and wrong for that arm.
- An empty or unreadable answer is not a failure. It is graded as frozen: wrong, and listed for hand-check.
- A run stops after 3 failures in a row, or after more than 10 failure rows, so that I find the cause before going on.
- An arm with more than 5 percent of its questions failed is reported as partial (more than 10 of the 200, or more than 4 of the 91).
- The fallback of Section 11 stays: if an arm cannot finish, I analyze complete patients only, in patient order, with the same subset for every arm.

## 4. How "close to cloud alone" is decided (agreed November 12)

- Primary comparisons: plan once against cloud alone, and check every step against cloud alone, with the 4B worker, on the 91 record-dependent questions (the set formed on November 12, SHA-256 prefix d288c671). Each is judged on its own. The results are exploratory, so there is no correction for two comparisons.
- What is compared: the arm's accuracy minus cloud alone's accuracy on the same questions, paired by question. A failed question counts as wrong. Only patients complete in both arms are used.
- The interval is the one of Section 8, unchanged: patient cluster bootstrap, resampling the 20 patients with replacement and keeping each sampled patient's questions in the set together, paired within question, 10,000 replicates, seed 20261105, two-sided 95 percent percentile interval. The question paired bootstrap stays a sensitivity analysis, as in Section 8.
- Close: the lower end of the interval is above minus 10 points. Worse: the upper end is below minus 10 points. Otherwise the result is inconclusive, and I report it as inconclusive.
- With each comparison I also report the number of questions only the arm got right and the number only cloud alone got right.
- Expected precision, stated before any result: with 91 questions in 20 patients (1 to 8 per patient), if the two arms disagree on about 30 percent of the questions, the interval is about 11 points wide on each side before clustering, and wider with it. So inconclusive is a likely result. I will not change the margin, the interval, the seed or the set after seeing any evaluation result.
- Secondary: the same comparisons on all 200 questions, and on the 109 questions outside the set, labeled secondary.
- The other arms (local alone, closed-book, MinionS, triggered log-prob) give context. Closed-book is not shown as a competing arm on the 91, because it is wrong on them by construction.

## 5. MinionS dates (agreed November 12)

| date | if | then |
|---|---|---|
| November 18 | fewer than 120 evaluation questions have the 4B shared parts done | MinionS runs on 5 seeded patients only |
| November 21 | the 4B shared parts are not 200 of 200 | MinionS becomes a limitation paragraph |
| November 22 | the MinionS trace is clean, its projected cost fits the money left after the other arms and the reserve (read from the console on November 21), and it ends by November 30 | I run the largest seeded patient subset that fits all three; otherwise MinionS becomes a limitation paragraph |

## 6. Cost and reserve

Cost means dollars, from the logs, checked against the billing console the day after each cloud run; GPU hours and latency are reported next to it. I keep 1.50 dollars for halted reruns and retraces only; MinionS does not use it.

## 7. Limitations stated now

- The version the API reports is the model name, gemini-3.8-flash. Logging it catches a change of name, not an update behind the same name. As a check, I run cloud alone once more on the first 20 of the 91 in benchmark order near the end of the runs (about November 30) and report the agreement, descriptive only.
- Cloud alone runs once on the evaluation set, so its run-to-run variation is not measured. The five closed-book runs varied from 107 to 115 correct of 200.
- The set's tilt: the 91 were chosen by the cloud model's own closed-book mistakes, and the cloud model gives the final answer in every arm except local alone.
- The warmup cloud alone run used the lab machine and the evaluation run uses the laptop, with different Python and google-genai versions. The prompts are identical, and the traced warmup question had the same prompt token count and the same answer on both.

## 8. Not covered here

The other changes since November 5 (the paused items, the corrected MinionS fix and its per-job runner, the 4B threshold at a context of 3,072, the 4B alone design, the guarded worker in the pipeline and its checks, running without agreement passes, and the patient-order batches) are in my next dated amendment, pushed before the first evaluation run of the 4B pipeline.
