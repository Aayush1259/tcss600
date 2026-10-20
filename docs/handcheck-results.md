# Warmup hand-check results (Oct 20, 2026)

Method: every answer the rule-based grader could not parse (needs_handcheck)
was read in full and judged against the gold label: correct in substance
(the answer's stated choice is the gold option), wrong (it chose a different
option), never concluded (truncated or empty, no choice stated).

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
| patient_18 q2 | 2.5 mg once daily | Wrong | Chose "(C)" = 5 mg once daily; misread the dose |
| patient_19 q2 | 5060 g | Correct | "(C)", with the other options eliminated by weight |
| patient_20 q2 | Hizentra 20% | Correct | Ends "(A)" |

Final think-off score: 3 (grader) + 12 (hand-check) = **15/20**.

## 27B local-only, think on (5 flagged; grader had scored 12/20)

| Patient | Gold | Verdict | Note |
|---|---|---|---|
| patient_06 q2 | Hypoglossal nerve | No answer | Answer empty; thinking consumed the full 1200-token budget |
| patient_14 q2 | 2 mitotic figures | Correct | "The answer is **(C)**" |
| patient_15 q2 | Vomiting | No answer | Answer empty; thinking consumed the full 1200-token budget |
| patient_16 q2 | -4 | Correct | Whole answer is "(B)"; options from the saved prompt: (B) = -4 = gold |
| patient_19 q2 | 5060 g | Correct | Whole answer is "(C)"; option C = 5060 g per the same question's think-off log |

Final think-on score: 12 + 3 = **15/20**.

## Cloud-only, gemini-3.8-flash (2 flagged; grader had scored 12/20)

| Patient | Gold | Verdict | Note |
|---|---|---|---|
| patient_10 q2 | 01/16/2019 | Correct | "**Correct Answer:** **(A)**" |
| patient_13 q2 | Initiate Abiraterone + Prednisolone therapy | Correct | "**Correct Answer:** **(C)**" |

Final cloud-only score: 12 + 2 = **14/20**.

## What the hand-checks show

- Think-off final 15/20 vs think-on final 15/20: level. Turning thinking
  off does not reduce quality on the warmup set. The grader's 3/20 was an
  artifact: think-off answers usually state their choice as a bare trailing
  letter or a bare "(C)" line, which the rule-based patterns do not parse.
- The two conditions fail differently at a 1200-token budget. Think-off:
  2 answers reasoned in the open and hit the cap before choosing. Think-on:
  2 answers came back empty because the hidden thinking used the whole budget.
- Grader gap to note: bare trailing letters, bare "(C)" lines, "The answer
  is", and "Correct Answer:" formats are not in the pattern pool. Flagged
  rows are hand-checked by design, so scores stay honest either way.
