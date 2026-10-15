import sys
sys.path.insert(0, "/home/aayush/minions")  # make `import minions` find the real package

import time
from minions.clients.ollama import OllamaClient
from minions.clients.gemini import GeminiClient
from minions.minions import Minions, JobOutput

class QuietOllamaClient(OllamaClient):
    def chat(self, messages, **kwargs):
        kwargs.setdefault("think", False)
        return super().chat(messages, **kwargs)

class RetryGeminiClient(GeminiClient):
    """Retry transient 503s with exponential backoff instead of dying."""
    def chat(self, messages, **kwargs):
        delay = 15
        for attempt in range(6):
            try:
                return super().chat(messages, **kwargs)
            except Exception as e:
                if "503" not in str(e) or attempt == 5:
                    raise
                print(f"Gemini 503, retrying in {delay}s (attempt {attempt+2}/6)...", flush=True)
                time.sleep(delay)
                delay *= 2

local_client = QuietOllamaClient(model_name="qwen3:4b", num_ctx=2048, thinking=False,
                                       structured_output_schema=JobOutput)
local_client.max_tokens = 500

remote_client = RetryGeminiClient(model_name="gemini-3.5-flash-lite")

protocol = Minions(local_client, remote_client, max_rounds=2)

context = [
    "On 2024-03-15, hemoglobin was 13.2 g/dL. On 2024-06-20, hemoglobin was 11.8 g/dL. "
    "The patient is a 58-year-old male with type 2 diabetes and hypertension. "
    "He takes metformin 1000 mg twice daily and lisinopril 10 mg daily."
]

result = protocol(
    task="What was the patient's hemoglobin value on 2024-03-15?",
    doc_metadata="Short clinical note used for a setup test.",
    context=context,
    max_rounds=2,
)

print("\n===== RESULT KEYS =====")
print(list(result.keys()))
print("\n===== FINAL ANSWER =====")
print(result.get("generated_final_answer", result))
