import os, time
import sys
sys.path.insert(0, "/home/localai/minions")

from minions.minion import Minion
from minions.clients.ollama import OllamaClient
from minions.clients.gemini import GeminiClient


class QuietOllamaClient(OllamaClient):
    """Ollama worker: capped context, thinking forced off on every call."""
    def __init__(self, model_name="qwen3.8:27b", num_ctx=8192, **kwargs):
        super().__init__(model_name=model_name, num_ctx=num_ctx, **kwargs)

    def _ensure_model_available(self):
        # Skip the built-in probe: it sends a system-only message that
        # Ollama 0.40.3 rejects with 500. Model presence was verified
        # separately with `ollama list`.
        return

    def chat(self, messages, **kwargs):
        kwargs["think"] = False
        return super().chat(messages, **kwargs)


class RetryGeminiClient(GeminiClient):
    """Gemini supervisor: retry on 503s with backoff."""
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


worker = QuietOllamaClient(model_name="qwen3.8:27b", num_ctx=8192)
worker.max_tokens = 500

supervisor = RetryGeminiClient(
    model_name="gemini-3.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
)
supervisor.max_tokens = 500

minion = Minion(local_client=worker, remote_client=supervisor)

doc = (
    "On 2024-03-15, hemoglobin was 13.2 g/dL. "
    "On 2024-06-20, hemoglobin was 11.8 g/dL. "
    "The patient is a 58-year-old male with type 2 diabetes and hypertension. "
    "He takes metformin 1000 mg twice daily and lisinopril 10 mg daily."
)
task = "What was the patient's hemoglobin value on 2024-03-15?"

print("========= MINIONS 27B TASK STARTED =========", flush=True)
result = minion(task=task, context=[doc], max_rounds=2,
                doc_metadata="Short clinical note used for a setup test.")
print("===== FINAL ANSWER =====", flush=True)
print(result["final_answer"], flush=True)
print("===== TIMING =====", flush=True)
print(result["timing"], flush=True)
print("===== USAGE =====", flush=True)
print("local:", result["local_usage"], flush=True)
print("remote:", result["remote_usage"], flush=True)
