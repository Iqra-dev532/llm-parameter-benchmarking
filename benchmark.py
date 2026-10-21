
import ollama
import pandas as pd
import time
import json

MODEL = "qwen2.5:3b"

# Load prompts from JSON file
with open("prompt.json", "r", encoding="utf-8") as file:
    prompt_data = json.load(file)

# Extract prompt text
prompts = [item["prompt"] for item in prompt_data]

temperatures = [0.0, 0.5, 1.0]
top_ks = [3, 7]
top_ps = [0.5, 1.0]

results = []

for prompt in prompts:
    for temperature in temperatures:
        for top_k in top_ks:
            for top_p in top_ps:
                for run in range(1, 3):
                    try:
                        start = time.time()

                        response = ollama.chat(
                            model=MODEL,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            options={
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p
                            }
                        )

                        answer = response["message"]["content"]
                        elapsed = round(time.time() - start, 2)

                        results.append({
                            "model": MODEL,
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                            "run": run,
                            "response": answer,
                            "response_time_seconds": elapsed,
                            "status": "success"
                        })

                        print(
                            f"Completed: T={temperature}, "
                            f"Top-K={top_k}, Top-P={top_p}, "
                            f"Run={run}"
                        )

                    except Exception as error:
                        print("Error:", error)

                        results.append({
                            "model": MODEL,
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                            "run": run,
                            "response": str(error),
                            "response_time_seconds": None,
                            "status": "failed"
                        })

# Save all results to CSV
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False, encoding="utf-8-sig")

print("\nExperiment finished.")
print("Results saved to results.csv")
print("Total records:", len(df))
print("Successful runs:", (df["status"] == "success").sum())
print("Failed runs:", (df["status"] == "failed").sum())
