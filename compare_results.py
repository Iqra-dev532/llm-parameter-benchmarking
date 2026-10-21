
from pathlib import Path
import re
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. FILE PATHS AND OUTPUT FOLDER
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

LLAMA_FILE = BASE_DIR / "lama_results.csv"
QWEN_FILE = BASE_DIR / "qwen_results.csv"

OUTPUT_DIR = BASE_DIR / "comparison_report"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

def load_results(file_path, model_name):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Please check the CSV filename."
        )

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "temperature",
        "top_k",
        "top_p",
        "response",
        "response_time_seconds",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Standardize the model name for comparison
    df["model"] = model_name

    # Convert numeric columns safely
    for col in [
        "temperature",
        "top_k",
        "top_p",
        "response_time_seconds",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" not in df.columns:
        df["status"] = "unknown"

    df["status"] = df["status"].fillna("unknown").astype(str).str.lower()

    # Use the recorded status where available
    df["success"] = df["status"].eq("success")

    # Normalize response text
    df["response"] = df["response"].fillna("").astype(str)

    # Failed runs should not be treated as valid model answers
    df["response_length_words"] = (
        df["response"].str.split().str.len()
    )

    # Use prompt category if available
    if "category" not in df.columns:
        df["category"] = "unknown"

    return df


llama = load_results(LLAMA_FILE, "Llama 3.2 (3B)")
qwen = load_results(QWEN_FILE, "Qwen 2.5 (3B)")

# Combine both datasets
data = pd.concat([llama, qwen], ignore_index=True)

data.to_csv(
    OUTPUT_DIR / "combined_results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== DATASET OVERVIEW ==========")
print("Total records:", len(data))
print("Models:", data["model"].unique().tolist())
print("\nRecords by model:")
print(data.groupby("model").size())


# ==========================================
# 3. NUMERICAL SUMMARY
# ==========================================

def success_rate(series):
    known = series.isin(["success", "failed"])
    if known.sum() == 0:
        return float("nan")
    return 100 * series[known].eq("success").mean()


summary_rows = []

for model_name, group in data.groupby("model", sort=False):
    successful = group[group["success"]]

    summary_rows.append({
        "model": model_name,
        "total_runs": len(group),
        "successful_runs": int(group["success"].sum()),
        "failed_runs": int(group["status"].eq("failed").sum()),
        "success_rate_percent": success_rate(group["status"]),
        "mean_response_time_seconds":
            successful["response_time_seconds"].mean(),
        "median_response_time_seconds":
            successful["response_time_seconds"].median(),
        "mean_response_words":
            successful["response_length_words"].mean(),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== MODEL COMPARISON ==========")
print(summary.round(3).to_string(index=False))


# ==========================================
# 4. PARAMETER-WISE SUMMARY
# ==========================================

successful_data = data[data["success"]].copy()

parameter_summary = (
    successful_data.groupby(
        ["model", "temperature", "top_k", "top_p"],
        dropna=False,
    )
    .agg(
        successful_runs=("response", "size"),
        mean_response_time=("response_time_seconds", "mean"),
        mean_response_words=("response_length_words", "mean"),
    )
    .reset_index()
)

parameter_summary.to_csv(
    OUTPUT_DIR / "parameter_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== PARAMETER SUMMARY ==========")
print(parameter_summary.round(3).to_string(index=False))


# ==========================================
# 5. GRAPH 1: AVERAGE RESPONSE TIME
# ==========================================

if summary["mean_response_time_seconds"].notna().any():
    plot_data = summary.dropna(
        subset=["mean_response_time_seconds"]
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_data["model"],
        plot_data["mean_response_time_seconds"],
    )
    plt.title("Average Response Time by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Response Time (seconds)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_average_response_time.png", dpi=300)
    plt.close()


# ==========================================
# 6. GRAPH 2: SUCCESS RATE
# ==========================================

plot_data = summary.dropna(subset=["success_rate_percent"])

if not plot_data.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_data["model"],
        plot_data["success_rate_percent"],
    )
    plt.ylim(0, 100)
    plt.title("Successful Run Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Success Rate (%)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_success_rate.png", dpi=300)
    plt.close()


# ==========================================
# 7. GRAPH 3: RESPONSE LENGTH
# ==========================================

if not successful_data.empty:
    length_data = successful_data.dropna(
        subset=["response_length_words"]
    )

    if not length_data.empty:
        plt.figure(figsize=(8, 5))
        length_data.boxplot(
            column="response_length_words",
            by="model",
        )
        plt.suptitle("")
        plt.title("Distribution of Response Length")
        plt.xlabel("Model")
        plt.ylabel("Words per Response")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "03_response_length.png", dpi=300)
        plt.close()


# ==========================================
# 8. GRAPH 4: TEMPERATURE EFFECT
# ==========================================

if not successful_data.empty:
    temp_summary = (
        successful_data.groupby(["model", "temperature"])
        ["response_time_seconds"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for model_name, group in temp_summary.groupby("model"):
        plt.plot(
            group["temperature"],
            group["response_time_seconds"],
            marker="o",
            label=model_name,
        )

    plt.title("Temperature vs Average Response Time")
    plt.xlabel("Temperature")
    plt.ylabel("Average Response Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_temperature_effect.png", dpi=300)
    plt.close()


# ==========================================
# 9. GRAPH 5: TOP-K EFFECT
# ==========================================

if not successful_data.empty:
    topk_summary = (
        successful_data.groupby(["model", "top_k"])
        ["response_time_seconds"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for model_name, group in topk_summary.groupby("model"):
        plt.plot(
            group["top_k"],
            group["response_time_seconds"],
            marker="o",
            label=model_name,
        )

    plt.title("Top-K vs Average Response Time")
    plt.xlabel("Top-K")
    plt.ylabel("Average Response Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_top_k_effect.png", dpi=300)
    plt.close()


# ==========================================
# 10. GRAPH 6: TOP-P EFFECT
# ==========================================

if not successful_data.empty:
    topp_summary = (
        successful_data.groupby(["model", "top_p"])
        ["response_time_seconds"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for model_name, group in topp_summary.groupby("model"):
        plt.plot(
            group["top_p"],
            group["response_time_seconds"],
            marker="o",
            label=model_name,
        )

    plt.title("Top-P vs Average Response Time")
    plt.xlabel("Top-P")
    plt.ylabel("Average Response Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_top_p_effect.png", dpi=300)
    plt.close()


# ==========================================
# 11. GRAPH 7: RESPONSE TIME BY CATEGORY
# ==========================================

if not successful_data.empty:
    category_data = (
        successful_data.groupby(["category", "model"])
        ["response_time_seconds"]
        .mean()
        .unstack("model")
    )

    if not category_data.empty:
        category_data.plot(kind="bar", figsize=(10, 6))
        plt.title("Average Response Time by Prompt Category")
        plt.xlabel("Prompt Category")
        plt.ylabel("Average Response Time (seconds)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / "07_category_response_time.png",
            dpi=300,
        )
        plt.close()


# ==========================================
# 12. GENERATE AUTOMATIC SUMMARY TEXT
# ==========================================

report_lines = [
    "LLM BENCHMARKING: NUMERICAL SUMMARY",
    "=" * 45,
    f"Total combined records: {len(data)}",
    "",
]

for _, row in summary.iterrows():
    report_lines.extend([
        f"Model: {row['model']}",
        f"Total runs: {row['total_runs']}",
        f"Successful runs: {row['successful_runs']}",
        f"Failed runs: {row['failed_runs']}",
        f"Success rate (%): {row['success_rate_percent']:.2f}"
        if pd.notna(row["success_rate_percent"])
        else "Success rate (%): unavailable",
        f"Mean response time (s): "
        f"{row['mean_response_time_seconds']:.3f}"
        if pd.notna(row["mean_response_time_seconds"])
        else "Mean response time: unavailable",
        f"Mean response length (words): "
        f"{row['mean_response_words']:.2f}"
        if pd.notna(row["mean_response_words"])
        else "Mean response length: unavailable",
        "",
    ])

report_lines.extend([
    "IMPORTANT LIMITATIONS",
    "- Response length is not a measure of answer quality.",
    "- Response time can depend on hardware and system load.",
    "- Hallucination requires factual checking of the answers.",
    "- Consistency requires comparing repeated answers.",
    "- Formatting compliance requires task-specific validation.",
])

(OUTPUT_DIR / "numerical_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\n========== FINISHED ==========")
print("Numerical summary saved to:", OUTPUT_DIR / "model_summary.csv")
print("Parameter summary saved to:", OUTPUT_DIR / "parameter_summary.csv")
print("Text report saved to:", OUTPUT_DIR / "numerical_report.txt")
print("Graphs saved in:", PLOT_DIR)
