# Warmup hand-check results (Oct 20, 2026)

Method: every answer the rule-based grader could not parse (needs_handcheck)
was read in full and judged against the gold label: correct in substance
(the answer's stated choice is the gold option), wrong (it chose a different
option), never concluded (truncated or empty, no choice stated).

The totals in the three tables below are the totals after this hand-check
and before the matching correction described in the last section. The final
scores are in the correction table.

## 27B local-only, think off (15 flagged; grader had scored 3/20)

| Patient | Gold | Verdict | Note |
|---|---|---|---|
| patient_01 q2 | Multiple sclerosis | Correct | Concludes "(A) Multiple sclerosis" |
| patient_02 q2 | brain, lungs, liver, heart, lymph nodes, muscles, bones | Never concluded | Hit the 1200-token cap mid-deliberation, leaning to the gold option (A) |
| patient_03 q2 | Below 5% | Correct | "the correct option is: (C)", evidence quoted matches gold |
| patient_04 q2 | Endoscopic ultrasound-guided FNA | Correct | Evaluates all options, ends with bare "B" |
| patient_05 q2 | CDKN2A/B deletion | Correct | "Option (C) is correct", ends with bare "C" |
| patient_06 q2 | Hypoglossal nerve | Never concluded | Hit the 1200-token cap mid-deliberation |
| patient_08 q2 | Acetaminophen | Correct | Ends "(B)" |
| patient_10 q2 | 01/16/2019 | Correct | "(A) 01/16/2019" |
| patient_11 q2 | Mycobacterium tuberculosis | Correct | Ends with bare "B" |
| patient_12 q2 | No evidence of malignancy | Correct | "(C) No evidence of malignancy" |
| patient_15 q2 | Vomiting | Correct | Ends with bare "D" |
| patient_17 q2 | 4-6 weeks | Correct | Ends "(B)" |
| patient_18 q2 | 2.5 mg once daily | Chose a different option | Chose "(C)" = 5 mg once daily; see the patient_18 note below |
| patient_19 q2 | 5060 g | Correct | "(C)", with the other options eliminated by weight |
| patient_20 q2 | Hizentra 20% | Correct | Ends "(A)" |

Score after hand-check: 3 (grader) + 12 (hand-check) = 15/20.

## 27B local-only, think on (5 flagged; grader had scored 12/20)

| Patient | Gold | Verdict | Note |
|---|---|---|---|
| patient_06 q2 | Hypoglossal nerve | No answer | Answer empty; thinking consumed the full 1200-token budget |
| patient_14 q2 | 2 mitotic figures | Correct | "The answer is **(C)**" |
| patient_15 q2 | Vomiting | No answer | Answer empty; thinking consumed the full 1200-token budget |
| patient_16 q2 | -4 | Correct | Whole answer is "(B)"; options from the saved prompt: (B) = -4 = gold |
| patient_19 q2 | 5060 g | Correct | Whole answer is "(C)"; option C = 5060 g per the same question's think-off log |

Score after hand-check: 12 + 3 = 15/20.

## Cloud-only, gemini-3.8-flash (2 flagged; grader had scored 12/20)

| Patient | Gold | Verdict | Note |
|---|---|---|---|
| patient_10 q2 | 01/16/2019 | Correct | "**Correct Answer:** **(A)**" |
| patient_13 q2 | Initiate Abiraterone + Prednisolone therapy | Correct | "**Correct Answer:** **(C)**" |

Score after hand-check: 12 + 2 = 14/20.

## Matching correction (Oct 20)

The hand-check above covers only answers the grader could not parse. After
it, I regraded the saved logs with a grader that reads more answer formats,
and the regraded totals came out below the hand-checked totals. The cause was
the matching function (matches() in grader.py), not the answer reader: for
any gold answer of 8 characters or fewer it tested whether the predicted
letter appears inside the gold text. The golds "15%", "-4", and "Vomiting"
contain no letter A to E, so a correct answer was always scored wrong, and
"Below 5%" tested the letter B instead of the option. Correct answers on five
warmup questions (patient_03 "Below 5%", patient_07 "15%", patient_15
"Vomiting", patient_16 "-4", patient_19 "5060 g") were scored wrong whenever
the grader could read the answer. Those rows were never flagged, so the
hand-check never saw them.

The fix compares the predicted option's text with the gold text (grader.py,
matches(), v2.1). Final scores, from the saved run logs:

| Run | Grader alone, first run | After the hand-check above | Final, with the matching correction |
|---|---|---|---|
| 27B think off | 3/20 | 15/20 | 17/20 (adds patient_07, patient_16) |
| 27B think on | 12/20 | 15/20 | 17/20 (adds patient_03, patient_07) |
| Cloud-only | 12/20 | 14/20 | 19/20 (adds patient_03, 07, 15, 16, 19) |
| 4B chunked | 10/20, 0 flagged | 10/20 | 11/20 (adds patient_03) |

Check: all 22 verdicts in the tables above agree with the corrected grader
(correct, wrong, or never concluded, row by row). Think off and think on
finish level at 17/20, so turning thinking off does not reduce quality on the
warmup set.

From now on I read every answer that is scored wrong by hand as well as
every answer the grader cannot parse, because this defect sat in the first
group.

## Patient_18 q2

Three runs read this record (27B think off, 27B think on, cloud-only) and all
three chose "(C) 5 mg once daily". The gold answer is "(A) 2.5 mg once daily".
The question asks for the Ramipril dose prescribed at discharge in May 2023.
The only May 2023 document (the 05/20/2023 to 05/21/2023 admission) lists
Ramipril 5 mg, 1-0-0-0 under "Medication upon Discharge"; the 2.5 mg entries
are in the 2016 letters. The answer is scored against the gold label and
counts as wrong, but the label may be wrong, so the question is treated as
disputed, like patient_08 in the pilot. All three runs agreed on it, so an
agreement-based uncertainty signal would not flag it.

## What the hand-checks show

- Think off and think on both finish at 17/20 after the correction: level.
  Turning thinking off does not reduce quality on the warmup set. The
  grader's first 3/20 for think off was an artifact: its answers usually
  state their choice as a bare trailing letter or a bare "(C)" line.
- The two conditions fail differently at a 1200-token budget. Think off:
  2 answers reasoned in the open and hit the cap before choosing. Think on:
  2 answers came back empty because the hidden thinking used the whole budget.
- Grader formats added in v2 after these checks: bare trailing letters, bare
  "(C)" lines, "The answer is", and "Correct Answer:". The matching defect
  was fixed in v2.1.
