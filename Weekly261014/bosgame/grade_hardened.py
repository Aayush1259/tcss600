import os, sys, json, re
sys.path.insert(0, "/home/localai/is-env/lib/python3.14/site-packages")
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def grade_letter(options_text, final_answer_text, model="gemini-3.5-flash-lite"):
    prompt = (
        "You are grading a multiple-choice answer. Read the OPTIONS and the MODEL ANSWER below.\n"
        "Output ONLY a JSON object with the single letter (A, B, C, D, or E) that the model chose.\n"
        "If the model chose none of the options, output {\"letter\": null}.\n"
        "No other text.\n\n"
        f"OPTIONS:\n{options_text}\n\nMODEL ANSWER:\n{final_answer_text[:1000]}"
    )
    for attempt in range(3):
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            m = re.search(r'\{\s*"letter"\s*:\s*(null|"([A-E])")\s*\}', r.text)
            if m:
                return m.group(2)
        except Exception as e:
            if "503" in str(e):
                import time; time.sleep(15); continue
            raise
    return None

if __name__ == "__main__":
    data = json.load(open('/home/localai/LongHealth/data/benchmark_v5.json'))
    total, n = 0, 0
    for fname in ['pilot_results_27b.json', 'pilot_results_27b_batch2.json']:
        for r in json.load(open(f'/home/localai/{fname}')):
            p = data[r["patient"]]; q = p["questions"][r["q_no"]]
            opts = "\n".join(f"({k[-1].upper()}) {q[k]}" for k in
                             ["answer_a","answer_b","answer_c","answer_d","answer_e"])
            pred = grade_letter(opts, r["final_answer"])
            correct = q["correct"]
            ok = (pred in correct) if len(str(correct).strip()) <= 8 else \
                 (q.get("answer_"+pred.lower(),"").strip().lower() == str(correct).strip().lower() if pred else False)
            total += 1; n += bool(ok)
            print(f"{r['patient']} q{r['q_no']}: hardened={pred} match={bool(ok)}", flush=True)
    print(f"\nHARDENED GRADER: {n}/{total}")
