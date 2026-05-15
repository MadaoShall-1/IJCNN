# EXACT 2026 Physics LLM Baseline Runner

This repo currently focuses on one task: run a local open-source LLM baseline on the physics training data and benchmark its answers.

Important competition constraint: every LLM used in the system must be open-source and have 8B parameters or fewer. GPT, Claude, Gemini, and other closed-source models are not allowed.

## Current Scope

Implemented:

- Load `Physics_Problems_Text_Only.xlsx`
- Filter examples by ID prefix, start offset, and limit
- Send each question to a local LLM server
- Require JSON output with `answer`, `unit`, and `explanation`
- Compare predicted answer/unit with gold answer/unit
- Save `results.csv`, `results.jsonl`, and `summary.json`
- Write partial results during long runs

Not implemented yet:

- Full symbolic pipeline from `physics_solver_design_v2.docx`
- Formula retrieval, VSO, per-step verification, and repair
- Final competition API server
- Dataset Type 1 logic-query handling

## Install

```bash
pip install -r requirements.txt
```

Python requirements cover only the benchmark runner:

- `pandas` and `openpyxl`: read the Excel dataset
- `requests`: call the local LLM HTTP API
- `sympy`: parse simple symbolic/numeric answer forms such as `9\sqrt{3} × 10^-27`
- `tqdm`: progress bar

LM Studio, Ollama, vLLM, and downloaded model files are runtime dependencies, not Python packages.

## Local LLM Server

The runner talks to a local OpenAI-compatible HTTP API. This is only an interface format used by local model servers; it does not mean OpenAI models are used.

The code rejects non-local hosts and obvious closed-source model names. Use one of:

- `http://localhost:11434/v1` for Ollama
- `http://localhost:1234/v1` for LM Studio
- `http://localhost:8000/v1` for local vLLM

### LM Studio Setup

The current tested setup uses LM Studio's headless CLI (`lms`) on Windows.

```powershell
# One-time official LM Studio daemon/CLI install:
powershell -ExecutionPolicy Bypass -Command "irm https://lmstudio.ai/install.ps1 | iex"

# Current shell only; restart PowerShell later to get this path automatically.
$env:PATH="C:\Users\xuzhi\.lmstudio\bin;$env:PATH"
```

Start the local daemon and OpenAI-compatible API server:

```powershell
lms daemon up
lms server start
```

The local API should now be available at:

```powershell
$env:LLM_BASE_URL="http://localhost:1234/v1"
$env:LLM_API_KEY="lm-studio"
```

Health check:

```powershell
lms server status
Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -Method Get | ConvertTo-Json -Depth 6
```

### Download Models

Downloaded/tested local models:

- `qwen3-8b`: LM Studio identifier for `qwen/qwen3-8b`
- `llama3.1-8b`: LM Studio identifier for `meta-llama-3.1-8b-instruct`

Qwen baseline:

```powershell
# Qwen3-8B is marketed as 8B, but its model card also lists 8.2B total params.
lms get qwen/qwen3-8b --gguf --yes
lms load qwen/qwen3-8b --identifier qwen3-8b --context-length 4096 -y
```

Llama baseline:

```powershell
# Meta's official repo is gated, so this uses LM Studio's GGUF community quantization.
lms get "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF" --gguf --yes
lms load meta-llama-3.1-8b-instruct --identifier llama3.1-8b --context-length 4096 -y
```

Only one 8B model needs to be loaded for a run. To switch models:

```powershell
lms ps
lms unload qwen3-8b
lms load meta-llama-3.1-8b-instruct --identifier llama3.1-8b --context-length 4096 -y

lms unload llama3.1-8b
lms load qwen/qwen3-8b --identifier qwen3-8b --context-length 4096 -y
```

Set the runner model name to the loaded identifier:

```powershell
$env:LLM_MODEL="qwen3-8b"
# or
$env:LLM_MODEL="llama3.1-8b"
```

### After Restart

```powershell
$env:PATH="C:\Users\xuzhi\.lmstudio\bin;$env:PATH"
lms daemon up
lms server start
lms load qwen/qwen3-8b --identifier qwen3-8b --context-length 4096 -y

$env:LLM_BASE_URL="http://localhost:1234/v1"
$env:LLM_API_KEY="lm-studio"
$env:LLM_MODEL="qwen3-8b"
```

### Optional Ollama Setup

```powershell
ollama pull qwen2.5:7b-instruct
ollama serve

$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_MODEL="qwen2.5:7b-instruct"
```

## Run Baseline

Quick smoke run:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --limit 5
```

Fast Qwen3 smoke run:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --limit 5 --disable-thinking --output-dir outputs/qwen3_fast_td5
```

Run with local self-checking:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --limit 5 --self-check --output-dir outputs/qwen3_selfcheck_td5
```

Compare Qwen and Llama:

```powershell
$env:LLM_MODEL="qwen3-8b"
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --limit 3 --output-dir outputs/qwen3_8b_td3

$env:LLM_MODEL="llama3.1-8b"
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --limit 3 --output-dir outputs/llama31_8b_td3
```

Larger run:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix LD --limit 50 --output-dir outputs/ld_50
```

Resume-style run:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --prefix TD --start 100 --limit 50 --output-dir outputs/td_100_50
```

Direct CLI config instead of environment variables:

```powershell
python run_benchmark.py --data Physics_Problems_Text_Only.xlsx --limit 20 --base-url http://localhost:1234/v1 --model qwen3-8b --api-key lm-studio
```

## Script Options

`run_benchmark.py` supports:

- `--data`: required Excel file path
- `--prefix`: evaluate only IDs with this prefix, e.g. `TD`, `LD`
- `--start`: skip the first N filtered examples
- `--limit`: evaluate at most N examples after filtering/skipping
- `--output-dir`: output directory, default `outputs`
- `--base-url`: local model server URL, overrides `LLM_BASE_URL`
- `--model`: local open-source model name, overrides `LLM_MODEL`
- `--api-key`: optional local server token, overrides `LLM_API_KEY`
- `--timeout`: per-example HTTP timeout in seconds
- `--max-retries`: retries after request or JSON parsing failures
- `--max-tokens`: maximum generated tokens per example
- `--disable-thinking`: add `/no_think` for Qwen3-style models; faster, but may reduce accuracy
- `--self-check`: ask the same local model to audit arithmetic and unit conversions before scoring
- `--save-every`: write partial outputs every N examples

## Outputs

Each run writes:

- `results.csv`: one row per evaluated example
- `results.jsonl`: same records in JSONL format
- `summary.json`: total accuracy, parse errors, numeric/unit matches, and prefix accuracy

`results.csv` also includes diagnostic fields:

- `error_type`: `correct`, `parse_error`, `numeric_mismatch`, `unit_mismatch`, or `numeric_and_unit_mismatch`
- `numeric_ratio`: converted predicted value divided by converted gold value
- `converted_pred_answer_num` / `converted_gold_answer_num`: values after unit-prefix scaling
- `normalized_pred_unit` / `normalized_gold_unit`: canonicalized units

Analyze a run:

```powershell
python analyze_results.py --results outputs/qwen3_8b_td3/results.csv
```

## Verification

Basic local checks:

```powershell
python -m py_compile run_benchmark.py llm_client.py evaluator.py analyze_results.py
```

The benchmark runner requires a running local model server for real evaluation. Without one, it fails fast instead of silently producing parse errors.
