# TCSS 600 Independent Study: Cloud-Supervised Local LLM Workers

Aayush Modi, University of Washington Tacoma. Advisor: Prof. Wei Cai. Autumn 2026. This README describes the study as of November 26, 2026.

## The study

A small language model runs locally as a worker and reads long patient records from the LongHealth benchmark (five-option questions about fictional patients). A cloud model, `gemini-3.8-flash`, acts as supervisor: it writes a plan of steps, may review the worker's findings, and writes the final answer. The study asks whether this comes close to the cloud model reading the whole record alone, and at what cost. Two workers: `qwen3:4b` on my laptop and `qwen3.8:27b` on a lab machine.

## Questions

- **Question A (primary; RQ2 of my proposal).** Do the supervised arms plan once and check every step, with the 4B worker, come within 10 points of cloud alone on the 91 record-dependent questions, and at what cost?
- **Question B (exploratory; RQ1).** Does the worker's log-prob signal pick the steps worth reviewing?
- **27B worker (exploratory; RQ3).** What changes with a stronger worker, including 27B alone in two forms: a per-chunk vote (Option B) and one call over the whole record (Option A).

Question A became primary on November 5, when the oracle, the agreement signal, the random controls and the Holm tests were paused; MinionS was dropped on November 13. The rule files below record each change.

## Key results

Evaluation set: 20 patients, 200 questions, of which the 91 record-dependent questions are the primary set. Each record is padded with two other patients' records (N = 2). Supervisor `gemini-3.8-flash`; one run per arm; graded against LongHealth's gold by the strict JSON rule. Each cell is a count of correct answers.

| arm | 4B, of 91 | 27B, of 91 | 4B, of 200 | 27B, of 200 |
|---|---|---|---|---|
| cloud alone (whole record, no worker) | 82 | 82 | 184 | 184 |
| plan once | 48 (worse) | 74 (inconclusive) | 138 | 175 |
| check every step | 49 (worse) | 75 (inconclusive) | 139 | 175 |
| triggered log-prob (exploratory) | 46 | 76 | 137 | 177 |
| local alone, per-chunk vote (no supervisor) | 35 | 56 | 107 | 139 |
| 27B alone, whole record in one call (Option A) | | to come | | to come |

- The words in parentheses are verdicts by the rule of my amendment of November 12: an arm is close if the whole 95 percent patient cluster interval of its difference from cloud alone lies above minus 10 points, worse if it lies below, and inconclusive otherwise.
- Plan once used about a tenth of cloud alone's cloud dollars with either worker. That saves about 2.4 cents per question for minutes of local worker time, which has little practical value.
- The 27B results are exploratory. The 27B supervised arms come from a run of November 6, before my current guards; no prompt in it was cut.
- Sources: the 4B results in `Weekly261118/results/` (from `Weekly261118/analyze_a.py`); 27B alone (Option B) from `Weekly261125/local_alone_27b.py`, analyzed with the same `analyze_a.py`. A per-question results file and a script that rebuilds every table follow in `Weekly261202/` by December 2.

## Rules fixed before the data

| rule file | what it fixes | in the repository |
|---|---|---|
| `Weekly261028/frozen-rules-2026-10-24.md` | JSON final answers, thinking levels, context windows, padding and the gap decision rule | [ae1af47](https://github.com/Aayush1259/tcss600/commit/ae1af47) |
| `Weekly261104/pre-registration-2026-11-03.md` | the protocols, the analysis, the tests | [06fd446](https://github.com/Aayush1259/tcss600/commit/06fd446), before the evaluation runs |
| `Weekly261104/correction-2026-11-04.md` | the cut prompts and the MinionS replies found on November 4 | [221db95](https://github.com/Aayush1259/tcss600/commit/221db95) |
| `Weekly261111/amendment-2026-11-11-record-dependent.md` | the 91 record-dependent questions as the primary set | [0fe7130](https://github.com/Aayush1259/tcss600/commit/0fe7130) |
| `Weekly261111/amendment-2026-11-13-cloud-alone-and-analysis.md` | cloud alone, failed questions, and how "close" is decided (agreed November 12) | [2fa69d0](https://github.com/Aayush1259/tcss600/commit/2fa69d0) |
| `Weekly261118/amendment-2026-11-14-4b-pipeline.md` (Amendment 2) | the guarded 4B pipeline, its checks, the threshold, scoring and cost | [7c6d7cb](https://github.com/Aayush1259/tcss600/commit/7c6d7cb), before the first 4B evaluation batch |
| `Weekly261125/amendment-2026-11-21-final-scope.md` (Amendment 3) | the final scope: the 27B scoring, the split of the 34 lost questions, 27B alone, the drift check | [209e846](https://github.com/Aayush1259/tcss600/commit/209e846) |
| `Weekly261125/note-2026-11-23-27b-alone.md` | the two 27B alone designs and how they are read | [412bbb3](https://github.com/Aayush1259/tcss600/commit/412bbb3); written before any 27B alone call and pushed after Option B ran, with its rules unchanged |
| `Weekly261125/note-2026-11-25-option-a-rules.md` | Option A's probe, its token count, its halt rules, a second reading and the reply schema | [8ae293c](https://github.com/Aayush1259/tcss600/commit/8ae293c), [733fdcd](https://github.com/Aayush1259/tcss600/commit/733fdcd), [1f69d3c](https://github.com/Aayush1259/tcss600/commit/1f69d3c), [359b0c8](https://github.com/Aayush1259/tcss600/commit/359b0c8), each before the step it governs |

## Corrections and limitations

- **Week 3 grader.** A matching fix for short gold answers changed the Week 3 scores; I corrected them on October 20 and 21 and pushed the corrections on November 2.
- **Cut prompts and MinionS (found November 4).** Ollama cut long prompts without an error, and the MinionS runner sent each round's jobs as one conversation, so it got one reply per round. The affected results were set aside. From November 13 on, every worker call is guarded: `"truncate": false`, a check of each prompt's token count, a check that the model stays in GPU memory, and a stop on any failed check. The MinionS fix I first named on November 4 was wrong; Amendment 2 records the right one, and MinionS was later dropped.
- **27B alone, Option A.** Its first warmup failed its reply check: with no reply schema the model wrote its reasoning before the JSON answer and many replies hit the 300-token limit. I fixed the reply format in section 6 of the November 25 note before a fresh warmup, which passed every check. No evaluation question had been sent.
- **Limits of the design.** One run per arm, 20 patients, and a 91-question primary set chosen by the cloud model's misses without the record. The supervised arms' final-answer prompt gives an example letter that cloud alone's prompt does not. The worker never sees the answer options, and the plan is written before the record is read; both are choices of this pipeline, not of supervision itself.

## Repository layout

Each folder is named after the weekly report it belongs to and runs on its own: modules used across weeks (`grader.py`, `data_loader.py`, `runlog.py`, `cloud_only.py`) are copied into each folder in the version used that week, and later scripts check the frozen ones by SHA-256. The weekly reports themselves are not in the repository.

- `Weekly261014/`: Weeks 1 and 2. Pilot code on the lab machine (`bosgame/`) and on the laptop (`laptop/`), and the lab machine setup log.
- `Weekly261021/`: Week 3. The baseline harness (data loader, rule-based grader, run logger), the 27B local-only, cloud-only and 4B chunked baselines, the regrade and hand-check tools, the hand-check results, the Week 3 status and the note on the 4B design.
- `Weekly261028/`: Week 4. JSON final answers, padding N = 2, the MinionS warmup runners for both workers, the frozen rules of October 24 and the lab machine log.
- `Weekly261104/`: Week 5. The pre-registration and its dated correction, the shared-parts pipeline, the evaluation protocols and the paired tests.
- `Weekly261111/`: Weeks 6 and 7. The amendments of November 11 and 13, the closed-book script and the script that forms the record-dependent set, cloud alone with its tests, the guarded 4B worker calls and the 4B alone vote.
- `Weekly261118/`: Week 7. Amendment 2, the guarded shared-parts runner and its count check, the threshold script, the scorer, 4B alone on the evaluation set, the analysis (`analyze_a.py`), the chart and `results/` (the Question A results, the chart and the list of the 91).
- `Weekly261125/`: Week 8. Amendment 3, the 27B scoring option and its count check, the drift check, the two 27B alone notes, the per-chunk vote (`local_alone_27b.py`) and the whole record (`whole_record_27b.py`) with their tests, and the comparison of 27B alone with 4B alone.
- `Weekly261202/`: Week 9. The environment (`environment.md`, `requirements.txt`) and the fixed question sets (`question_sets/`); by December 2 also the per-question results, the input log hashes and the script that rebuilds every table.

## Setup and running

1. Set up WSL Ubuntu (laptop) or Linux (lab machine), Python, Ollama 0.40.3 and the models in `Weekly261202/environment.md`, and install `Weekly261202/requirements.txt`.
2. Download LongHealth's `benchmark_v5.json` to `~/LongHealth/data/` and copy `Weekly261202/question_sets/warmup_ids.json` and `eval_ids.json` to your home folder; check them against `SHA256SUMS.txt`.
3. Put the Gemini key in a `.env` file as `GEMINI_API_KEY`; git ignores it.
4. Run each script from inside its week folder. Each script's opening comment gives its commands, and run logs go to the folder named by `BASELINE_LOG_DIR`. From Week 7 on, the scripts that call the cloud stop at the cost caps their amendment sets.

## Reporting rule

Every number in a weekly report states the protocol, the worker model, the supervisor model and the grading method. Run logs store full prompts and answers so every reported number can be checked against the code and the logs; they contain benchmark text and stay out of the repository.

## Paths before the reorganization of November 2

| Before | Now |
|---|---|
| `bosgame/`, `laptop/` | `Weekly261014/bosgame/`, `Weekly261014/laptop/` |
| `docs/lab-machine-setup-log.md` | `Weekly261014/` |
| Week 3 files in `harness/`, `docs/handcheck-results.md`, `docs/progress-status.md`, `docs/decision-4b-local-only-design.md` | `Weekly261021/` |
| Week 4 files in `harness/`, `docs/frozen-rules-2026-10-24.md`, `docs/lab-machine-log.md` | `Weekly261028/` |

Links in my earlier reports use the old paths. They still work at commit `6de8c96`: https://github.com/Aayush1259/tcss600/tree/6de8c96
