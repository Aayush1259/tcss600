# Amendment to my pre-registration of November 3, 2026: record-dependent questions

Dated Wednesday, November 11, 2026. I wrote and pushed this before any closed-book call on the evaluation questions. No closed-book answer on the evaluation set exists when this is pushed.

## Why

On November 9 I ran the closed-book test on the 20 warmup questions: the cloud supervisor (gemini-3.8-flash, thinking low) saw only the question and the five options, with no record. It answered 11 of 20 correctly, 55 percent, exact 95 percent interval 32 to 77 percent, against 20 percent for guessing. With the record, cloud alone scored 19 of 20 at the same setting. So the cloud can answer about half of these questions from its own knowledge, and on those questions no protocol can show the effect of supervision. Prof. Cai agreed on November 11 to keep the benchmark and asked me to make the questions that need the record the primary analysis, with the rule fixed in advance.

## 1. Benchmark

Unchanged: LongHealth, padding N = 2, the frozen evaluation set (q-indices 3 to 12 of all 20 patients, 200 questions, SHA-256 c1acdc648af20a52b7328e786bdba95017b74fec963436601a1d242c75a8bf65), scored against the benchmark answer. Disputed answers follow my rule of November 9: scored as the benchmark says, and reported separately with the record's evidence.

## 2. Closed-book runs on the evaluation set

- Five runs over all 200 evaluation questions, with `closed_book.py` (SHA-256 prefix 044bfd86).
- The prompt is the one of the warmup run on November 9, word for word: the question and the five options only, no record text, with a JSON answer of one letter. The script checks every prompt for record text.
- Supervisor gemini-3.8-flash, thinking low, the same call settings as cloud alone. No temperature is set, so each run samples its answers independently. Each run is a separate log.
- The model version reported by the API is recorded on every row. A run stops if the version changes.
- A failed call is retried (up to six attempts, as in cloud alone) and is not counted as an answer. An answer that cannot be read counts as wrong, as frozen.
- Every run must cover all 200 questions. If a run cannot finish, I report that and do not form the set until five complete runs exist.

## 3. The rule for "record-dependent"

A question is record-dependent if its closed-book answer is wrong in at least 3 of the 5 runs.

Reason: if the cloud does not know an answer and guesses among five options, it is right at least once in five runs with probability 1 minus 0.8 to the fifth power, about 0.67, so a rule of wrong in all five runs would drop most such questions. It is right in 3 or more of 5 runs with probability about 0.058, so this rule keeps them. A question the cloud answers correctly 9 times in 10 enters the set with probability about 0.009.

The set is fixed once the five runs are complete and before any protocol is scored on the evaluation set. I report how many questions it holds, how many per patient, and the closed-book accuracy on it.

## 4. Analysis

- Primary: on the record-dependent questions, the accuracy of each protocol, compared with cloud alone and with local alone, as paired differences on the same questions.
- Secondary: the same on all 200 questions.
- Intervals: the patient cluster bootstrap of Section 8 (10,000 replicates, seed 20261105), resampling patients and keeping each sampled patient's questions in the set together.
- These results are exploratory. The eight Holm tests stay paused, as since November 5.
- I do not use the "share of the gap closed" measure I proposed on November 9. On record-dependent questions closed-book accuracy is close to zero by construction, so each protocol's accuracy is reported directly.

## 5. Limitation

Some answer options give the answer away. This is a known issue with multiple-choice benchmarks, and I report it as a limitation. A benchmark whose answers cannot be guessed, for example one with exact numeric answers, is left for later work.

## 6. Not covered here

The other changes since November 5 (the paused items, the corrected MinionS fix, the 4B threshold at a context of 3,072, the 4B alone design and the worker call checks) are in my next dated amendment, before the warmup gate on November 13.
