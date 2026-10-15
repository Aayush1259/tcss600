import os, sys, time, json, re
sys.path.insert(0, "/home/localai/is-env/lib/python3.14/site-packages")
sys.path.insert(0, "/home/localai/minions")

from minions.minion import Minion
from minions.clients.ollama import OllamaClient
from minions.clients.gemini import GeminiClient


class QuietOllamaClient(OllamaClient):
    def __init__(self, model_name="qwen3.8:27b", num_ctx=16384, **kwargs):
        super().__init__(model_name=model_name, num_ctx=num_ctx, **kwargs)

    def _ensure_model_available(self):
        return

    def chat(self, messages, **kwargs):
        kwargs["think"] = False
        return super().chat(messages, **kwargs)


class RetryGeminiClient(GeminiClient):
    def chat(self, messages, **kwargs):
        delays = [15, 30, 60, 120, 240]
        for attempt in range(6):
            try:
                return super().chat(messages, **kwargs)
            except Exception as e:
                if "503" in str(e) and attempt < 5:
                    print(f"  Gemini 503, retry in {delays[attempt]}s", flush=True)
                    time.sleep(delays[attempt])
                else:
                    raise


def extract_letter(final_answer):
    m = re.search(r'\(([A-E])\)', final_answer)
    if m:
        return m.group(1), "auto"
    m = re.search(r'[Aa]nswer:\s*([A-E])\b', final_answer)
    if m:
        return m.group(1), "auto"
    return None, "manual-review"


worker = QuietOllamaClient(model_name="qwen3.8:27b", num_ctx=16384)
worker.max_tokens = 500
supervisor = RetryGeminiClient(model_name="gemini-3.5-flash-lite",
                               api_key=os.environ["GEMINI_API_KEY"])
supervisor.max_tokens = 500
minion = Minion(local_client=worker, remote_client=supervisor)

data = json.load(open("/home/localai/LongHealth/data/benchmark_v5.json"))
pilot = [("patient_04", 1), ("patient_08", 1), ("patient_12", 1),
         ("patient_16", 1), ("patient_20", 1)]

results = []
for pid, qidx in pilot:
    p = data[pid]
    docs = list(p["texts"].values())
    q = p["questions"][qidx]
    task = (q["question"] + "\n"
            + "\n".join(f"({k[-1].upper()}) {q[k]}" for k in
                        ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"])
            + "\nAnswer with the letter of the correct option.")
    print(f"=== {pid} q{q['No']} ({len(docs)} docs, {sum(len(x) for x in docs)} chars) ===", flush=True)
    t0 = time.time()
    result = minion(task=task, context=docs, max_rounds=2,
                    doc_metadata=f"pilot {pid} q{q['No']}")
    dt = time.time() - t0
    pred, how = extract_letter(result["final_answer"])
    correct = (pred == q["correct"])
    print(f"  predicted={pred} ({how}) correct={q['correct']} match={correct} time={dt:.1f}s", flush=True)
    results.append({
        "patient": pid, "q_no": q["No"], "correct": q["correct"],
        "predicted": pred, "grade_how": how, "match": correct,
        "wall_time_s": round(dt, 1),
        "local_tokens": result["local_usage"].prompt_tokens + result["local_usage"].completion_tokens,
        "remote_tokens": result["remote_usage"].prompt_tokens + result["remote_usage"].completion_tokens,
        "final_answer": result["final_answer"][:500],
    })
    time.sleep(5)

json.dump(results, open("/home/localai/pilot_results_27b_batch2.json", "w"), indent=1)
n_correct = sum(1 for r in results if r["match"])
print(f"\nPILOT DONE: {n_correct}/{len(results)} correct", flush=True)
for r in results:
    print(f"  {r['patient']} q{r['q_no']}: pred={r['predicted']} actual={r['correct']} match={r['match']} {r['wall_time_s']}s", flush=True)
