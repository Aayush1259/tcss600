# TCSS 600 Independent Study: Cloud-Supervised Local LLM Workers

Aayush Modi, University of Washington Tacoma. Advisor: Prof. Wei Cai. Autumn 2026.

## About this study

Small language models run locally as workers that read long medical records (the LongHealth benchmark) and answer multiple-choice questions. A cloud model (Gemini) acts as supervisor. The study measures when supervisor review is worth its cost.

Research questions:

- **RQ1 (primary):** Can an uncertainty signal trigger supervisor review only when it is needed?
- **RQ2:** Reference points: no review vs review at every step vs the MinionS protocol.
- **RQ3:** Worker strength: a 27B local worker vs a 4B local worker.

Workers: `qwen3.8:27b` on the lab machine (bosgame), `qwen3:4b` on a laptop. Supervisor: `gemini-3.8-flash`. Question sets are frozen: 20 warmup questions and a 200-question evaluation set, with no overlap with the pilot questions.

## Repository layout

Each folder is named after the weekly report it belongs to and holds the code and documents for that week. Each folder runs on its own, so modules used across weeks (`grader.py`, `data_loader.py`, `runlog.py`, `cloud_only.py`) appear in each week's folder in the version used that week. The weekly reports themselves are not in the repository.

- `Weekly261014/`: Weeks 1 and 2. Pilot code that ran on the lab machine (`bosgame/`) and on the laptop (`laptop/`), and the lab machine setup log.
- `Weekly261021/`: Week 3. The baseline harness: data loader, rule-based grader, run logger, the 27B local-only, cloud-only, and 4B chunked baselines, the regrade and hand-check tools, the hand-check results, the Week 3 status, and the note on the 4B design.
- `Weekly261028/`: Week 4. JSON final answers, padding N = 2, the MinionS warmup runners for both workers, the frozen rules of October 24, and the lab machine log.

The Week 3 folder includes the grader's short-gold matching fix, the 4B-aware regrade script, and the corrected hand-check results and status. The corrected final warmup scores are 17 out of 20 for the 27B worker with thinking off, 17 with thinking on, 19 for cloud-only, and 11 for the 4B chunked baseline. I made these corrections on October 20 and 21 and reported them in Weekly261021, but pushed them only now, together with this reorganization.

Paths before the reorganization:

| Before | Now |
|---|---|
| `bosgame/`, `laptop/` | `Weekly261014/bosgame/`, `Weekly261014/laptop/` |
| `docs/lab-machine-setup-log.md` | `Weekly261014/` |
| Week 3 files in `harness/`, `docs/handcheck-results.md`, `docs/progress-status.md`, `docs/decision-4b-local-only-design.md` | `Weekly261021/` |
| Week 4 files in `harness/`, `docs/frozen-rules-2026-10-24.md`, `docs/lab-machine-log.md` | `Weekly261028/` |

Links in my earlier reports use the old paths. They still work at commit `6de8c96`: https://github.com/Aayush1259/tcss600/tree/6de8c96

## Reporting rule

Every number in a weekly report states the protocol, the worker model, the supervisor model, and the grading method. Run logs store full prompts and answers so every reported number can be re-checked against the code and the logs.
