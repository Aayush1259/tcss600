# Week 3 Cross-Check Audit (October 20, 2026)

Purpose: verify Weekly261021 against Prof. Cai's feedback on Report 1, against
Report 1 itself, and against the signed proposal. Every item lists the demand,
where it is answered, and its status. Fixes applied on Oct 20 are marked.

## A. Prof. Cai's two questions

**1. Thinking off (confirm quality on the 20 warmup questions).**
ADDRESSED. Weekly261021, Baseline results: think-off 15/20 and think-on 15/20
after hand-checks, verdict stated with the flagged-row breakdown (12 correct in
substance, 1 wrong, 2 never concluded). The check Cai asked for is the check
that was run.

**2. Cloud billing.**
- Free options, Gemini free tier with caching and spreading runs over days:
  ADDRESSED (460 days at about 20 requests/day; caching is in the design and
  does not change the arithmetic).
- Free options, other providers (Groq backup from the proposal): ADDRESSED
  (fixed Oct 20; the report now states Groq serves open models only, so it
  cannot be the gemini-3.8-flash supervisor, and its free limit is tight
  against 9,204 calls).
- Free options, student credit programs: ADDRESSED (fixed Oct 20; GitHub
  Student Pack, Azure for Students, and GCP education credits all sit outside
  the AI Studio billing path).
- Cover the cost himself: ADDRESSED (decision stated: self-pay, about $66
  expected, $85 worst case).
- Calls by component (plans, step reviews, aggregation per protocol,
  cloud-only, MinionS, oracle, both workers): ADDRESSED. Aggregation is given
  as 1,600 total across the 4 protocols (400 per protocol: 1 call x 200
  questions x 2 workers).
- Realistic tokens per call measured on gemini-3.8-flash: PARTIAL, disclosed.
  Cloud-only is measured (about 12,000 input, thinking 142 to 436, $0.00996
  per question). Plan, review, aggregation, and MinionS calls use stated
  assumptions because those protocols first run in Week 4; the report commits
  to replacing the assumptions with Week 4 measurements.
- Debugging allowance: ADDRESSED (30%).
- Free-tier days: ADDRESSED (3.8-flash about 460 days; flash-lite about 19
  days).
- Drop list if cost or quota breaks: ADDRESSED (oracle 50 to 20; eval 200 to
  100 for the expensive protocols; flash-lite bulk reviews with a validation
  subset; agreement passes; the 380-question expansion stays dropped).

**3. Subscription question (Claude or ChatGPT/Codex).**
ADDRESSED for Claude (paid subscription, stated, with the no-API-credits
caveat) and for Google AI Pro (1-year student offer, quota effect unverified).
One micro-item open: the report does not name a ChatGPT status. Awaiting the
fact; if there is no ChatGPT subscription, one clause closes it.

## B. The five code-versus-report clarification points

1. Protocol: pilot ran Minion, not MinionS. ADDRESSED (corrected pilot record
   in the report's feedback responses and in the run table).
2. Supervisor: the 9/10 came from 27B plus gemini-3.5-flash-lite. ADDRESSED
   (stated in the corrected record).
3. Hand-check all 10 pilot answers. ADDRESSED (8 confirmed, 1 ambiguous,
   1 inconclusive; Report 1's "9/10, matching my manual check" is superseded
   by this full hand-check).
4. No local-only or cloud-only reference points. ADDRESSED (all baselines run
   in Week 3; results table in the report).
5. 4B context and chunking; confirm before running pilot_4b.py. ADDRESSED
   (pilot_4b.py was not run as written; the 4B baseline uses the frozen
   chunked design with JSON-mode votes, described in the report and in
   docs/decision-4b-local-only-design.md).
Standing rule (every number states protocol, worker, supervisor, grading):
APPLIED (run table carries all four labels; cost figures carry model and
pricing labels).

## C. Prof. Cai's open Week 2 items

- Freeze supervisor and worker models; resolve qwen3.5:4b versus qwen3:4b:
  DONE (frozen qwen3:4b; proposal source corrected; report explains).
- Freeze warmup and evaluation question sets: DONE (20 and 200, zero overlap
  with the pilot, SHA-256 hashes recorded in the frozen ID files; the formal
  hash listing belongs to the Week 5 pre-registration deliverable).
- Padding decision: DONE (N = 0, rationale and limitation stated).
- Read Fang et al.: DONE Oct 15 with notes; now stated in the report's
  timeline walk (fixed Oct 20; it was missing).
- Bosgame memory: CORRECTED in the report (120 GB unified memory, with an
  apology for Report 1's 30 GB figure).
- Week 3 core tasks (baselines as top priority, data loader, grader, logging):
  DONE, except local-compute (GPU seconds) logging, which is disclosed as not
  done in the report's timeline table and is a Week 4 plan item.

## D. Report 1 claims that Report 2 corrects

- "First real MinionS loops" and "9 out of 10 correct": corrected to Minion
  and to the hand-checked 8 of 10 confirmed, 1 ambiguous, 1 inconclusive.
- "The full study needs about 2,500 supervisor calls": that figure covered
  only the MinionS loop. Report 2 gives the full-design estimate (9,204) and
  now says why the number grew (bridge sentence added Oct 20).
- "30 GB RAM": corrected to 120 GB unified memory.
- Report 1's next-week item 2 (run the 4B pilot on the same 10 questions):
  superseded by Prof. Cai's chunking directive; the 4B baseline ran chunked on
  the frozen warmup set instead. The design note documents the change.
- Report 1's next-week item 5 (test shorter output prompts): superseded by the
  fairness fix; the output budget was raised to 1,200 tokens for both think
  conditions because 500 tokens crippled both, and the report's issues section
  explains this.

## E. Signed proposal consistency

- RQ1, RQ2, RQ3 wordings, protocol definitions, the shared-parts caching
  design, the oracle bound (at most 23 aggregation calls at K = 6, fixed
  50-question sample), metrics, and the paired-statistics plan: CONSISTENT.
- Supervisor on the free tier (proposal) versus self-pay (Report 2): a real
  deviation, now named as a change in the report with the measured reason
  (free-tier cap makes the proposal's plan infeasible).
- Deliverables on watch: pre-registration at the end of Week 5 (frozen design,
  thresholds, primary comparisons, question-list hashes); interim memo in
  Week 6; lab machine log against the Section 6 hours plan; code repository
  with a table-rebuild script by Dec 11.

## F. Academic risks and what they change in the plan

1. **Grader scalability.** Think-off flagged 15 of 20 answers (75%). At eval
   scale that is roughly 150 hand-checks per run, which is not feasible. Plan
   change, before the Week 5 pre-registration: extend the rule-based grader
   with the four documented miss classes (bare trailing letter; bare "(C)"
   line; "The answer is"; "Correct Answer:"), regrade the warmup logs, and
   report both gradings in the Week 4 report so the change is disclosed and
   the warmup numbers stay comparable. docs/handcheck-results.md is the
   evidence base for the new patterns.
2. **Warmup noise.** At 20 questions, a 1 or 2 question gap is noise. The
   thinking-off verdict is a pilot check, exactly as Prof. Cai framed it, and
   the report claims no more than "does not drop". Real claims wait for the
   200-question eval set with paired tests.
3. **GPU-seconds logging is a hard gate.** The proposal's local-compute metric
   cannot be reported for the eval runs unless the logging lands first. It is
   Week 4 plan item 4; no eval run starts before it.
4. **Lab hours log.** A proposal deliverable. Start the dated log now; Week 3
   usage was the baseline runs of Oct 17 to 20.
5. **Empty worker answers.** Think-on lost 2 of 20 answers to hidden thinking
   consuming the budget. In the protocol runs, an empty worker output is a
   defined case in the logs and must stay one in the Week 4 pipeline design.
6. **Worker versus machine confound.** The 4B ran on the laptop and the 27B on
   bosgame, so latency is reported per machine and accuracy is the cross-worker
   comparison, per the proposal. The Week 7 item (4B on the lab machine,
   50-question sample) closes the confound; no latency claims across workers
   before then.

## Verdict

On track. After the Oct 20 fixes, every explicit item in Prof. Cai's feedback
is answered in Weekly261021, Report 1's errors are corrected on the record,
and the one deviation from the signed proposal (supervisor billing) is stated
as a change with its measured reason. The two partial items (Week 4 token
measurements, GPU-seconds logging) are disclosed in the report itself.
