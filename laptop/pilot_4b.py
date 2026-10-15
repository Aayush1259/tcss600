import sys, json, time, re
sys.path.insert(0, "/home/aayush/is-env/lib/python3.11/site-packages")
sys.path.insert(0, "/home/aayush/minions")
from minions.minions import Minions
from minions.clients.ollama import OllamaClient
from minions.clients.gemini import GeminiClient
from minions.minions import JobOutput

class FixedOllamaClient(OllamaClient):
    def __init__(self, **kwargs):
        kwargs["structured_output_schema"] = JobOutput
        super().__init__(**kwargs)
    def _ensure_model_available(self):
        return
    def chat(self, messages, **kwargs):
        kwargs["think"] = False
        return super().chat(messages, **kwargs)

class RetryGeminiClient(GeminiClient):
    def chat(self, messages, **kwargs):
        d = 15
        for a in range(6):
            try:
                return super().chat(messages, **kwargs)
            except Exception as e:
                if "503" not in str(e) or a == 5:
                    raise
                print(f"  503, retry in {d}s", flush=True)
                time.sleep(d); d *= 2

worker = FixedOllamaClient(model_name="qwen3:4b", num_ctx=2048)
worker.max_tokens = 500
supervisor = RetryGeminiClient(model_name="gemini-3.5-flash-lite")
supervisor.max_tokens = 500
minion = Minions(local_client=worker, remote_client=supervisor)

data = json.load(open("/home/aayush/LongHealth/data/benchmark_v5.json"))
# Same 10 questions as the bosgame 27B pilot for direct RQ3 comparison
pilot = [("patient_03",0),("patient_07",0),("patient_11",0),("patient_15",0),("patient_19",0),
         ("patient_04",1),("patient_08",1),("patient_12",1),("patient_16",1),("patient_20",1)]

results = []
for pid, qidx in pilot:
    p = data[pid]
    docs = list(p["texts"].values())
    q = p["questions"][qidx]
    task = q["question"] + "\n" + "\n".join(
        f"({k[-1].upper()}) {q[k]}" for k in
        ["answer_a","answer_b","answer_c","answer_d","answer_e"])
    print(f"=== {pid} q{q['No']} ===", flush=True)
    t0 = time.time()
    try:
        result = minion(task=task, doc_metadata=f"LongHealth {pid}", context=docs, max_rounds=2)
        dt = time.time() - t0
        final = result["final_answer"]
        m = re.search(r'correct choice is \(([A-E])\)', final)
        pred = m.group(1) if m else re.findall(r'\(([A-E])\)', final)[-1]
        print(f"  pred={pred} time={dt:.1f}s", flush=True)
        results.append({"patient": pid, "q_no": q["No"], "predicted": pred,
                        "time_s": round(dt,1), "final_answer": final[:400]})
    except Exception as e:
        print(f"  FAILED: {str(e)[:100]}", flush=True)
        results.append({"patient": pid, "q_no": q["No"], "predicted": None,
                        "error": str(e)[:100]})
    time.sleep(3)

json.dump(results, open("/home/aayush/pilot_results_4b.json", "w"), indent=1)
print(f"\n4B PILOT DONE: {len(results)} questions", flush=True)
