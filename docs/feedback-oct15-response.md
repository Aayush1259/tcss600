# Prof. Cai's Feedback on Weekly Report 1 (Oct 15, 2026): Response Plan

## What he praised

Fast progress in one week. I caught the truncation bug and the extractor bug myself by checking token counts and hand grading, which he called a good research habit. Deleting the exposed API key was the right call.

## Corrections to my record

1. **Protocol:** the 10-question pilot ran **Minion** (the simpler back-and-forth protocol from the Minions codebase), not MinionS. My report described MinionS. I now label every run with its actual protocol.
2. **Supervisor:** the pilot used **gemini-3.5-flash-lite**, not gemini-3.8-flash. So 9/10 means "27B worker + flash-lite supervisor."
3. **Grading:** the saved files show 6/10 from the original extractor. The 9/10 comes from re-grading with the cloud-based hardened grader. I am hand-checking all 10 answers before the 9/10 counts as confirmed.
4. **Baselines:** no local-only or cloud-only results exist yet, so the 9/10 has nothing to compare against. Week 3 runs them.

## His orders and my response

**1. Labeling rule.** Every number in every report states protocol, worker model, supervisor model, and grading method. Adopted as a standing rule; the run table format is: run / protocol / worker / supervisor / grading / result.

**2. Thinking-off check.** Agreed: verify on 20 warmup questions that quality does not drop noticeably before relying on it. No need to cut the question count or switch workers yet. This runs in Week 3 Phase 3, pass bar within 10 percent of the think-on reference.

**3. No lab funding for API costs.** I must either complete the study on free options or cover the cost myself. Free options: Gemini free tier (~15 req/min, ~500 to 1,500 req/day, ~250K tokens/min) with every supervisor response cached to disk and runs spread across days; Groq free tier as a fallback for open-model calls only (it cannot serve Gemini); Google student credit programs as a billing-account fallback.

**4. Full cost estimate in the next report**, with his breakdown:
- Supervisor calls by component: plans, step reviews, aggregation per protocol, cloud-only baseline, MinionS baseline, oracle search, both workers. My budget: ~8,980 calls subtotal, ~11,700 with 30% debugging. (The old ~2,500 estimate covered only the MinionS loop.)
- Real tokens per call measured on gemini-3.8-flash with the planned thinking level, thinking tokens billed as output and recorded separately.
- Debugging and rerun allowance (30%).
- Days the full study takes on free tier only: CORRECTION (verified Oct 17): the gemini-3.8-flash free tier is ~20 requests/day (5 RPM cap), not 1,000/day. At 20/day the ~11,700-call study would take ~585 days, so free-on-3.8-flash is impossible. The free path is flash-lite at ~500 req/day, about 24 days at $0, but then the supervisor is flash-lite and the report must state that plainly.
- What gets dropped if cost or quota breaks: oracle 50 to 20 questions; eval 200 to 100 for the expensive protocols; bulk reviews on flash-lite with a 3.8-flash validation subset (stated openly); drop the agreement signal's extra local passes if laptop time binds; the 380-question extension stays dropped.
- Verified pricing: gemini-3.8-flash $0.75/$3.75 per 1M input/output through Dec 31, 2026 (doubles Jan 1, 2027; the study ends Dec 11). Base case ~$70, ~$91 with debugging. Sensitivity: ~$350 if thinking runs 10k tokens/call instead of ~1.5k. The Week 3 measurement decides.

**5. Subscription question.** He asked before and got no answer. I will answer yes or no this week with one line of reasoning. A subscription does not include API credits; it is for coding, debugging, and reading papers faster.

**6. 4B chunking.** The 4B worker's context is 2,048 tokens; the Minion protocol hands it the whole 11,000 to 13,000 token record at once. Same silent truncation as the 27B case. I will not run `pilot_4b.py` as written; the 4B runs only after MinionS chunking is in.

**7. Freeze decisions.** Proposal says `qwen3.5:4b`, code uses `qwen3:4b`: I will check `ollama list`, freeze the real tag, and sync the proposal. Same for the supervisor (3.8-flash if healthy, else flash-lite stated openly), the 20 warmup + 200 eval question sets (hashed, zero overlap with the pilot), and padding N.

**8. Read Fang et al.** Done (arXiv:2509.24050). Same escalation question as RQ1, solved with RL post-training; their "prompt surface features fail at routing" finding supports our logprobs-based signal; their policy-collapse warning applies to our threshold calibration; RL training itself is out of scope this quarter.

**9. Week 3 priority.** Local-only + cloud-only baselines with both workers, plus the data loader, grader, and cost/latency/compute logging. This was missing from my plan; it is now the top item. Week 4 runs the real MinionS baseline. The interim results memo is due Week 6 (Nov 4 to 10).

**10. Timeline checklist.** The next report walks the proposal timeline item by item: done or not done, with reasons.

## Status of open items

- Billing decision: after Week 3 token measurements.
- Subscription answer: due in Weekly261021.
- Hand verification of the 10 pilot answers: Phase 1.
- All freezes (models, question sets, padding): Phase 1.
- 4B chunking: before any 4B run.
