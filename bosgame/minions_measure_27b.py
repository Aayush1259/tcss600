import os, sys, time, json
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
                    print(f"Gemini 503, retrying in {delays[attempt]}s (attempt {attempt + 2}/6)", flush=True)
                    time.sleep(delays[attempt])
                else:
                    raise


worker = QuietOllamaClient(model_name="qwen3.8:27b", num_ctx=16384)
worker.max_tokens = 500

supervisor = RetryGeminiClient(
    model_name="gemini-3.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
)
supervisor.max_tokens = 500

minion = Minion(local_client=worker, remote_client=supervisor)

data = json.load(open("/home/localai/LongHealth/data/benchmark_v5.json"))
p = data["patient_02"]
docs = list(p["texts"].values())
q = p["questions"][0]
task = (
    q["question"] + "\n"
    + "\n".join(f"({k[-1].upper()}) {q[k]}" for k in ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"])
    + "\nAnswer with the letter of the correct option."
)
print("patient_02, question No.", q["No"], flush=True)
print("context chars:", sum(len(x) for x in docs), flush=True)
result = minion(task=task, context=docs, max_rounds=2,
                doc_metadata="LongHealth patient_02 measurement run")
print("===== FINAL ANSWER =====", flush=True)
print(result["final_answer"], flush=True)
print("correct answer:", q["correct"], flush=True)
print("===== TIMING =====", flush=True)
print(result["timing"], flush=True)
print("===== USAGE =====", flush=True)
print("local:", result["local_usage"], flush=True)
print("remote:", result["remote_usage"], flush=True)
