# Correction to the pre-registration, November 4, 2026

This file adds to `pre-registration-2026-11-03.md`, which stays as frozen at commit 06fd446. I found the problems below while the eval shared runs were in progress. I checked them by counting prompts, replies, and whether a job answered. I compared no answer from these eval runs with gold.

## 1. Prompts cut by Ollama

When a prompt reaches the context limit, Ollama raises no error. It keeps the first 4 tokens and the end of the prompt, down to half the limit plus 2 tokens, and the model answers as usual. My run logs record only the reduced count. Ollama's service log shows the cut: `truncating input prompt limit=1026 prompt=2073 keep=4 new=1026`.

| Run | Limit | Prompts cut |
|---|---|---|
| 27B pipeline, warmup and eval so far | 16,384 | None. Largest prompt 2,270 tokens |
| 4B pipeline, warmup | 2,048 | 402 of 5,293 jobs, in 18 of 20 questions |
| 4B pipeline, eval, 154 questions | 2,048 | 2,724 of 44,579 jobs, in 125 questions |
| 27B local only, N = 2, warmup | 32,768 | 16 of 20 questions |

A 4B job prompt starts with the subtask and the question, so a cut job never sees them. Cut jobs answer 99.3 percent of the time on warmup, and whole jobs 4.2 percent, so 66 percent of 4B warmup answers and 61 percent of 4B eval answers so far come from cut prompts. The 4B warmup results, the frozen 4B thresholds, and the 4B eval log are affected. I do not know by how much, and this does not show that the signals or reviews would work with whole prompts.

A 27B local only prompt puts the question, options, and instruction first, then the patient's own record, then the distractor records, so a cut prompt keeps only distractor text. On warmup at N = 2, cut prompts score 3 of 16 and whole prompts 3 of 4. In the earlier prompt-JSON run, the 16 malformed answers are exactly the 16 cut prompts. So "27B alone 6 of 20" does not measure what 27B can do at N = 2, and the capability gap behind choosing N = 2, the cascade numbers, and the MinionS regression count are affected. The cascade numbers in Section 10 and my Week 4 report rely on this run, and I withdraw the `--num-ctx 32768` flag for 27B local only in Section 11.

## 2. What I did about the cut

- I stopped the 4B eval shared run on November 4 at about 22:29, after 154 complete questions. The log is kept, and the run can resume from it.
- The 27B eval shared run continues, because its prompts are whole.
- I will not run 27B local only on eval at a limit of 32,768. I first rerun it on warmup with a limit that fits the largest prompt, and I return to the choice of N = 2 with my advisor.
- I proposed to my advisor to raise the 4B limit to 3,072, redo 4B warmup and thresholds on warmup only, and restart the 4B eval. On 4B eval so far, Ollama logged cut prompts of 2,048 to 2,224 tokens, so 3,072 fits the largest prompt and the 300-token reply. If the restart goes ahead, its settings and the new thresholds are recorded in a further dated entry before that run starts.
- From now on I check Ollama's service log for cuts before I use a run. A cut prompt has exactly L / 2 + 2 tokens at a limit L, but a whole prompt can have that count too, so the count alone does not show a cut.

## 3. MinionS worker jobs

My MinionS runners in `Weekly261028` build the worker client without `use_async=True`. In that mode the library sends all the chats prepared for a round to Ollama as one conversation. Ollama returns one reply, and the library records it as the result for the first job. Every saved round on both workers got exactly one reply, against a median of 15 chats prepared per round on 4B and 25 on 27B. Ollama's log agrees on 4B: 24 requests for the 24 rounds of the 20 warmup questions.

So MinionS 11 of 20 (27B) and 10 of 20 (4B) on warmup do not measure MinionS, and the four regressions and the fix test rest on the same runs. Before any MinionS eval run, I set `use_async=True`, rerun MinionS on warmup for both workers, and measure its time and cost. The eval estimates in Section 11 came from the old runs and no longer hold.

## 4. Earlier use of the eval set

On October 22, before this freeze, I ran cloud only at the old settings (padding N = 0, free-text answers) on 11 eval questions, patient 01 q3 to q12 and patient 02 q3: 9 scored correct, 0.1346 dollars. No decision used those answers. I report them as a limitation. The frozen file leaves this out.

## 5. Cost caps

The caps in Section 11 total 20 dollars, and my credit was 18.24 dollars on November 2 and 16.86 on November 4. My MinionS logs also leave out thinking tokens, which bill as output, so real MinionS cost has been about 1.5 times the log value. These caps replace those in Section 11 for every run that has not started. The 27B run in progress keeps its flags, and I watch it against the limit below. All are in dollars.

| Run | Cap | Worst case billed |
|---|---|---|
| 27B eval shared parts, running | I stop it if the logged total passes 1.20 | 1.20 |
| 4B eval shared parts, first run, stopped | Spent 0.72 | 0.72 |
| 4B warmup rerun, all warmup steps | 0.50 | 0.50 |
| 4B eval shared parts, restart | 1.25 | 1.25 |
| Protocols and oracle on eval | 4B 2.00, 27B 0.75 | 2.75 |
| MinionS warmup rerun | 0.75 per worker, log value | 2.25 |
| MinionS eval | 2.00 per worker, log value | 6.00 |
| Cloud only eval | The smaller of 6.00 and the credit left divided by 1.10, minus 0.50 | What the credit leaves |

My credit fell 1.38 dollars from November 2 to November 4 while the spend meter rose 1.26, about 10 percent more, which may be sales tax. Until I confirm this, I treat the credit as buying 10 percent less than list price. Cloud only runs last, and I check the bill before it starts. If the credit left cannot cover 200 questions, I stop and report. If the MinionS warmup rerun shows that its eval does not fit, a further dated entry comes before that run.

Two facts about the caps in the code. The cap is checked after each question, so a run can pass it by at most one question's calls. `--max-usd` and `--max-calls` count per process, so a resume resets the count, and I check the running total from the call logs before I resume a run.

## 6. Stops

The 27B eval shared run stopped on November 4 at 05:06 during question 106, because the lab machine briefly could not resolve the Gemini server's name and my code retries only 503 and 429 errors. No completed question was lost. I resumed it at 10:28 with the same flags, log, and cache, under the failure rule in Section 11 and with no code change, inside a shell loop that restarts it only after a crash.

The pipeline code on both machines still matched commit 06fd446 at that time. The 4B stop in Section 2 of this file was my choice, not a failure.

## 7. Additions to the repository

- `paired_tests.py` runs the primary tests in Section 6, with the frozen p-value formulas and no smoothing, and applies the failure rule and the complete-patients fallback in Section 11. McNemar stays a sensitivity figure, in its exact form. On the warmup logs it reproduces every score, review count, flip, oracle, and MinionS figure in the frozen key table. Its 4B warmup table will change if 4B warmup is rerun.
- `README.md` has a line for `Weekly261104/`.
- `minions_run_27b_warmup_fix.py` has a corrected description at the top. Its code is the same.

Section 12 of the frozen file mentions a folder named `outgoing/`. That is my local working folder and it is not in the repository.
