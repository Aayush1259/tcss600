# Amendment 3 (November 21, 2026): the final scope, the split of the 34 lost questions, the 27B items and the drift check

Pushed before I open any worker finding of the 34 lost questions and before any 27B score or drift check exists.

This amendment adds to my pre-registration of November 3 and my amendments of November 11, November 12 and November 14. It follows Prof. Cai's feedback of November 21 on Weekly Report 7. When I wrote it I knew the November 16 results (Weekly Report 7) and these facts from the logs: 16 of the 34 lost questions had no worker finding and 18 had one; the 27B run of November 6 had a largest worker prompt of 2,270 tokens at a context of 16,384; and the 27B local only warmup of November 4 at a context of 32,768 was cut to 16,386 tokens on 16 of 20 prompts. I had not looked at any finding's text for the 34, at any 27B answer, or at any new cloud answer.

Before pushing, I ran the part of the preparation that reads no finding: a script that lists the 34 and draws the 14 comparison questions (section 2), and checks LongHealth's answer positions against the pipeline's chunks, which it rebuilds and matches to the run's chunk counts and padding on all 91. No answer span of the 34 or of the 14 is split by a chunk boundary (1 question of the 91 is). The answer spans are short (206 spans on the 91, median 12 characters, 124 shorter than 20), so I replaced a planned quote code that needed 20 shared characters, which could not fire on 41 of the 91, with the position code in section 2.

## 1. Scope until December 11

- Runs allowed, and only these: (1) scoring the three protocols from the existing 27B run of November 6; (2) 27B alone as its own local alone baseline, only after Prof. Cai decides how (section 4); (3) the drift check (section 5). The split of the 34 lost questions (section 2) is an analysis of existing logs, not a run.
- MinionS is dropped, under Prof. Cai's permission of November 13. The oracle, the agreement signal, the random controls and the 8 Holm tests are paused, under his feedback of November 5. Each is written as a limitation in the final report in place of the matching parts of my proposal.
- Every 27B result is exploratory. Nothing in this amendment changes a rule or a result of the 4B evaluation.
- Extra analyses of existing logs are capped at two, at most 2 hours together, done only after the final report draft exists: the characters of record text sent to the cloud per question, and a post hoc rule that sends a question to cloud alone when plan once had no worker finding (accuracy and cost). Anything else goes to my thesis list.

## 2. The split of the 34 lost questions (post hoc, within 4 hours)

- **The 34.** The questions of the 91 that cloud alone answered correctly and plan once (4B) answered wrong, taken from the scoring logs of November 16 by script before any finding is opened.
- **The question it answers (Prof. Cai's wording).** For each of the 34: did the worker's findings hold the correct fact?
- **Rule.** A question counts as "held the correct fact" if at least one finding among plan once's chosen steps (first pass) comes from a chunk of the target patient's record and states the fact that tells the gold option apart from the other four. Matching option text alone, or a quote overlapping the answer location alone, is not enough; I use those only to sort the questions before reading. A finding from a padding patient never counts, and I report how many such findings stated the fact.
- **Material for each question.** The question, the five options, the gold option, the text at LongHealth's answer location (each span marked, with 200 characters on either side), and every finding of plan once's chosen steps with its chunk, patient and quote. The cloud's final letter is not shown while I label.
- **Mechanical codes, by script, no judgment.** Whether the question had any finding; whether it had a finding from the target patient; whether a 6,000-character chunk boundary splits the answer location (at least one answer span, and every answer span); whether any quote, found word for word in its chunk, overlaps the characters of an answer span. Quotes not found word for word are counted and reported. A finding counts as the target patient's when its chunk lies wholly in the target record, or, for a chunk that holds both the end of the target record and the start of the padding, when its quote is found in the target part; a quote there that is not found word for word is decided by reading and reported.
- **Labeling.** One rater (me), stated as such. If time remains inside the 4 hours, I add a comparison group: 14 of the 48 questions both arms answered correctly, drawn with seed 20261105, shuffled with the 34, with the outcome hidden, labeled by the same rule. Without time for it, the counts are reported without a comparison group, and I say so.
- **Wording fixed now.** Let H be the number of the 34 whose findings held the correct fact. If H is 8 or fewer (26 or more of the 34 lacked the fact), I write that the loss lies mainly in reading. If H is 9 or more, I write that both reading and combining lose answers, and I give both counts. In both cases I write that this is reading as this pipeline poses it: the worker never saw the options, the plan was written before the record was read, and the chunks do not overlap.
- **Reporting.** Counts only, with no interval and no test, marked post hoc, one run. If the 4 hours end before all 34 are labeled, I report the labeled count and the number left.

## 3. Scoring the 27B run of November 6

- **The run.** 200 evaluation questions, padding N = 2, qwen3.8:27b at a context of 16,384 with thinking off, 200 plans, 713 reviews, 48,592 worker calls (November 2, 05:12 to November 6, 02:23). It ran before the current guards. Its largest worker prompt was 2,270 tokens, so no prompt was cut. I state all of this beside every 27B result.
- **Code.** `score_c3.py` gains a worker option that selects the 27B run's cache and the pre-registered 27B threshold of November 2, -0.11813126318156719, flagging a step only if its score is strictly below it. The frozen protocol code is unchanged. The code is pushed before the first scoring call.
- **Trace first.** One question end to end with the printed prompts. Before scoring the rest, the counts must reconcile: 200 shared rows, 200 plans, 713 reviews and 48,592 worker calls, between the shared log and the calls log. If a count does not reconcile, or a worker prompt reached the context, the 27B run is reported as not scored, with the reason, and it is not rerun.
- **Plans.** The 27B run's plans were written in its own run, so they differ from the 4B run's plans. A 27B against 4B comparison therefore changes the worker and the plan together, and I say so.
- **Caps.** 0.60 dollars and 400 fresh cloud calls. The run stops at either.
- **What is reported.** On the 91 and the 200: each 27B arm's accuracy, its difference from cloud alone with the patient cluster interval of November 12 and the margin line (exploratory), rescued and broken against cloud alone, the cost ratio, the redo rate of check every step, and the worker seconds per question on the lab machine. The 27B against 4B comparison is paired by question and descriptive only, because the date, the guards, the context (16,384 against 3,072) and the plans differ.
- **Wording fixed now, for 27B plan once on the 91.** 62 or fewer correct: no gain over 4B plan once (48) that 91 questions can resolve. 63 to 73: the gap to cloud alone partly closes with a stronger worker. 74 or more (within 8 of cloud alone's 82): the gap largely closes.

## 4. 27B alone

- **The situation, to be reported to Prof. Cai before anything runs.** The padded record at N = 2 is about 142,000 characters (36,187 tokens by Gemini's count). On November 4, 27B local only at a context of 32,768 was still cut to 16,386 tokens on 16 of 20 warmup prompts, so the whole record does not fit 32,768 in Qwen's count. At N = 0, with nothing cut, 27B alone scored 17 of 20 on warmup.
- **What I check first, read only, with no inference.** On the lab machine: the model's maximum context and the memory available.
- **The two options I put to him.** (a) One call over the whole record at a context of at least 40,960, if the model and memory allow it, with a check that the prompt tokens Ollama reports equal the full tokenized length, so a cut halts the run. (b) A per-chunk vote as for 4B, with the same chunks, prompt rule and tie rule. Nothing runs before his answer.
- **Then.** One question traced, then the 20 warmup questions, then the 200. Truncation is never accepted. If it is not run and checked by December 4, I report the situation above as a limitation.

## 5. The drift check

As my November 13 amendment fixed it: cloud alone once more on the first 20 of the 91 in benchmark order, the same prompt and settings as November 12, descriptive only. I report the number of the 20 letters that match November 12 and the accuracy. It runs on the same day as the 27B scoring, so the 27B final answers and the check share a date. It is not rerun. Cap: 0.60 dollars.

## 6. Files

This amendment. The 27B scoring change and the split's scripts follow with their results; benchmark text and run logs never go to the repository.
