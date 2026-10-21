# LLM Benchmarking: Qwen 2.5 vs Llama 3.2

## Project Overview

This project investigates the effect of generation parameters on the performance of two locally running Large Language Models (LLMs):

- **Qwen 2.5:3B**
- **Llama 3.2:3B**

The experiments examine how Temperature, Top-K, and Top-P settings influence response generation. Python is used to run benchmark experiments, store model responses, analyze results, generate comparison reports, and visualize performance metrics.

The project uses a collection of 10 prompts covering factual questions, reasoning, structured output, uncertainty, creative writing, and instruction following.

## Research Objectives

- Compare the response generation performance of Qwen 2.5:3B and Llama 3.2:3B.
- Investigate the effects of Temperature, Top-K, and Top-P settings.
- Measure response time and response length.
- Compare successful and failed model runs.
- Generate numerical summaries, CSV reports, and visualizations.
- Examine differences in model behavior across prompt categories.

## Experimental Configuration

| Parameter | Values |
|---|---|
| Qwen model | `qwen2.5:3b` |
| Llama model | `llama3.2:3b` |
| Temperature | 0.0, 0.5, 1.0 |
| Top-K | 3, 7 |
| Top-P | 0.5, 1.0 |
| Number of prompts | 10 |
| Repetitions per configuration | 2 |
| Runs per model | 240 |
| Combined recorded runs | 480 |

Each model has 3 × 2 × 2 = 12 parameter combinations. With 10 prompts and 2 repetitions, this gives 240 runs per model and 480 runs in total.

## Project Structure

```text
llm-parameter-benchmarking/
│
├── benchmark.py
├── compare_results.py
├── prompt.json
├── qwen_results.csv
├── lama_results.csv
├── LLM_Comparison_Report.docx
│
└── comparison_report/
    ├── combined_results.csv
    ├── model_summary.csv
    ├── numerical_report.txt
    ├── parameter_summary.csv
    │
    └── plots/
        ├── 01_average_response_time.png
        ├── 02_success_rate.png
        ├── 03_response_length.png
        ├── 04_temperature_effect.png
        ├── 05_top_k_effect.png
        ├── 06_top_p_effect.png
        └── 07_category_response_time.png
```

## File Descriptions

### Python Scripts

- **`benchmark.py`** — Runs the Qwen 2.5:3B benchmark using the prompts and configured generation parameters. Saves newly generated results to `results.csv`.
- **`compare_results.py`** — Loads the Qwen and Llama CSV datasets, combines the results, calculates summary statistics, and generates comparison tables, a numerical report, and seven graphs.

### Prompts and Experimental Data

- **`prompt.json`** — Contains the 10 benchmark prompts and their categories.
- **`qwen_results.csv`** — Contains the recorded Qwen benchmark results.
- **`lama_results.csv`** — Contains the recorded Llama benchmark results.

### Reports and Visualizations

- **`LLM_Comparison_Report.docx`** — Written project report.
- **`combined_results.csv`** — Combined experimental dataset.
- **`model_summary.csv`** — Model-level performance summary.
- **`parameter_summary.csv`** — Results grouped by model and generation parameters.
- **`numerical_report.txt`** — Automatically generated numerical summary and limitations.
- **`plots/`** — Contains seven performance comparison graphs.

## Technologies Used

- Python 3
- Ollama
- Qwen 2.5:3B
- Llama 3.2:3B
- pandas
- matplotlib
- JSON and CSV data formats

## Requirements

Install Python 3 and [Ollama](https://ollama.com/).

Install the Python dependencies:

```bash
pip install ollama pandas matplotlib
```

Download the models using Ollama:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

Make sure the Ollama service is running before executing the benchmark.

## How to Run the Project

### 1. Prepare the Files

Place `benchmark.py`, `compare_results.py`, `prompt.json`, and the required CSV files in the project directory.

Ensure the prompt file is named exactly `prompt.json`.

### 2. Run the Qwen Benchmark

```bash
python benchmark.py
```

This executes the Qwen benchmark and creates or overwrites `results.csv`.

If you want to preserve your existing results, back up the CSV before running the script.

### 3. Compare Both Models

Make sure `qwen_results.csv` and `lama_results.csv` exist in the same directory as `compare_results.py`.

Then run:

```bash
python compare_results.py
```

The script generates the `comparison_report` directory containing the combined dataset, summary CSV files, numerical report, and graphs.

## Recorded Results

The included numerical report records the following results:

| Metric | Llama 3.2 (3B) | Qwen 2.5 (3B) |
|---|---:|---:|
| Total runs | 240 | 240 |
| Successful runs | 240 | 240 |
| Failed runs | 0 | 0 |
| Success rate | 100% | 100% |
| Mean response time | 16.415 seconds | 1.119 seconds |
| Mean response length | 61.56 words | 35.80 words |

In the recorded experiment, Qwen had a lower mean response time than Llama. However, shorter response length or faster generation does not necessarily mean better answer quality.

These figures describe the included dataset and may vary with hardware, system load, and experimental conditions.

## Limitations

- Response length alone does not measure answer quality.
- Response time depends on hardware and system load.
- Hallucination requires factual verification of generated answers.
- Consistency requires comparing repeated answers.
- Instruction-following and formatting compliance need task-specific evaluation.
- The recorded success rate measures successful execution, not factual correctness.

## Future Improvements

- Add automatic factual accuracy evaluation.
- Measure consistency across repeated responses.
- Evaluate JSON validity and instruction-following accuracy.
- Add more prompts and repeated experiments.
- Record hardware specifications and software versions.
- Compare additional local language models.

## Author

**Iqra Aslam**

LLM Benchmarking and Comparative Analysis Project
