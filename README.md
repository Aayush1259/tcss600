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

- `bosgame/`: Week 1-2 pilot code that ran on the lab machine (Minion pilot, measurements, graders).
- `laptop/`: Week 1-2 pilot code that ran on the laptop.
- `harness/`: The Week 3 baseline harness. All scripts run from one folder.
  - `data_loader.py`: Loads the benchmark and the frozen question sets into standard records.
  - `grader.py`: Rule-based answer grader with a hand-check queue for unparseable answers.
  - `runlog.py`: JSONL run logger. Every row records protocol, worker, supervisor, grading method, timing, tokens, cost, and the full prompt and answer.
  - `local_only_27b.py`: 27B worker alone (bosgame).
  - `cloud_only.py`: Supervisor alone on the full record (bosgame).
  - `local_only_4b_chunked.py`: 4B worker alone over ~1,500-token record chunks with majority voting (laptop).
  - `regrade.py`: Regrades a finished run log with the current grader, without rerunning models.
  - `probe_38flash_quota.py`: Measures the gemini-3.8-flash free-tier daily request limit with 10 small calls.
- `docs/`: Study documents.
  - `lab-machine-setup-log.md`, `lab-machine-commands-explained.md`: Lab machine setup, step by step.
  - `week3-plan.md`: Week 3 plan (verification, harness, baselines, cost estimate, report).
  - `progress-status.md`: Current status: what is done, what is pending, open decisions.
  - `feedback-oct15-response.md`: Prof. Cai's feedback on Weekly Report 1 and my response to each point.
  - `decision-4b-local-only-design.md`: Why the 4B local-only baseline uses chunked majority voting.

## Reporting rule

Every number in a weekly report states the protocol, the worker model, the supervisor model, and the grading method. Run logs store full prompts and answers so every reported number can be re-checked against the code and the logs.
