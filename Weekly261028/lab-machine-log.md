# Lab Machine Log (bosgame)

Hours used against the proposal's Section 6 plan (core 42 hours, optional 8,
if-needed 10). Machine time is taken from the run logs' timestamps; attended
sessions are noted where they differ. The server is shared with other members
of Prof. Cai's lab, so heavy runs are coordinated before they start.

Plan by week: Week 2: 2 hours (setup, smoke test, throughput). Week 3: 6 hours
(27B local-only baseline). Week 4: 8 hours (MinionS baseline, 27B). Week 5:
4 hours (shared parts, thresholds). Week 6: 4 hours (first 27B shared-parts
runs). Week 7: 12 hours (all protocols, 27B; 4B on the lab machine).

## Week 2 (Oct 7 to 13)

- Oct 13: SSH and Ollama setup, qwen3.8:27b pull, first extraction test,
  throughput measurement, first Minion loop, 10-question pilot (two batches).
  Sessions ran into the night of Oct 13 to 14. Machine time: from the pilot
  scripts' run timestamps (to be totaled from the logs).

## Week 3 (Oct 14 to 20)

- Oct 17, evening: Ollama server started in tmux; harness smoke tests
  (27B local-only 1 question, cloud-only 1 question); first think-off 20q
  launch (later relabeled think-on after the think-parameter bug was found).
- Oct 18, after midnight: overnight tmux session ran the corrected think-off
  20q and think-on 20q back to back at the 1,200-token budget; cloud-only 20q
  launched and stopped at 12/20 on the free-tier daily quota.
- Oct 18, about 00:43: cloud-only resumed with quota-safe pacing; reached
  13/20 before the daily quota stopped it again.
- Oct 20, afternoon: cloud-only resumed on the fresh daily quota and
  completed 20/20. Short session.
- Machine time for the week: the overnight run is the bulk of it; exact
  totals come from the baseline logs' logged_at timestamps and per-row
  wall times.

## Week 4 (Oct 21 to 28)

- Oct 22: grader v2.1 regression on bosgame, all cases OK; regrades completed (think-off 3 to 17, think-on 12 to 17, cloud-only 12 to 19).
- Oct 24: 27B local-only warmup at the frozen setting (N=0 JSON 17 of 20, N=2 prompt JSON 3 of 20, N=2 constrained JSON 6 of 20), cloud-only N=2 19 of 20, MinionS 27B warmup 11 of 20 with 87 supervisor calls in about 28 minutes.
- Oct 28: 27B shared-parts pipeline smoke (1 question, 6 steps, 439 calls) and full 27B shared run started (question 2 in progress).
- Machine time for the week: lab machine 2 hours 20 minutes (all Week 4 local runs on bosgame, from run-log timestamps).
