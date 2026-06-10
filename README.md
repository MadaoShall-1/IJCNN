# Education System — Physics & Logic Solver

An AI-powered educational solver that handles two types of queries:
- **Type 1**: Logic-based reasoning (premises + yes/no, MCQ, or open-ended questions)
- **Type 2**: Deterministic-first physics calculation (parse -> formula retrieval -> deterministic symbolic solve -> step verification -> optional diagnosis / guarded repair -> final-answer validation -> response)

## Project Structure

```
├── api.py                # FastAPI server, main entry point
├── router.py             # Auto-detect Type 1 vs Type 2
├── config.py             # SolverConfig (runtime parameters)
├── requirements.txt
│
├── parser/               # Stage 0 — Problem Parser (~96% accuracy)
│   ├── main.py           # parse_problem() entry
│   ├── llm_fallback.py   # LLM recovery for hard cases
│   ├── template_fallback.py
│   └── ...
│
├── type1/                # Type 1 — Logic Pipeline
│   ├── pipeline.py       # run(payload, config, solver)
│   ├── z3_solver.py      # Z3 formal logic solver
│   ├── dspy_modules.py   # LLM-based reasoning
│   └── schemas.py
│
├── type2/                # Type 2 — Physics Pipeline (6 stages)
│   ├── stage1.py         # Formula retrieval (beam search)
│   ├── stage2.py         # Symbolic solve (SymPy)
│   ├── stage4.py         # Trace diagnosis
│   ├── stage5.py         # Repair failed traces
│   ├── stage6.py         # Response assembly
│   └── formula_library.json
│
├── scripts/              # Dataset processing & evaluation
│   ├── run_stage0_on_dataset.py
│   ├── eval_stage1_retrieval.py
│   └── ...
│
├── Dataset/              # Input datasets
└── outputs/              # Cached parse results
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM

The system uses [DSPy](https://github.com/stanfordnlp/dspy) for optional local LLM fallback calls. The final vLLM default is `qwen3-8b-awq` on host port `8002`. Set environment variables before starting:

```powershell
$env:DSPY_MODEL="openai/qwen3-8b-awq"
$env:DSPY_API_BASE="http://localhost:8002/v1"
$env:DSPY_API_KEY="EMPTY"
```

```bash
export DSPY_MODEL="openai/qwen3-8b-awq"
export DSPY_API_BASE="http://localhost:8002/v1"
export DSPY_API_KEY="EMPTY"
```

Or edit `config.py` directly via `SolverConfig`.

### 3. (Optional) Local model

If using a local model, start the model server first, then point `DSPY_API_BASE` to it.
The included vLLM Docker setup serves `Qwen/Qwen3-8B-AWQ` as `qwen3-8b-awq`:

```bash
docker compose -f docker-compose.vllm.yml up --build -d
```

The final default host port is `8002`. If that port is already in use, choose another host port before starting the container and update `DSPY_API_BASE` to match:

```powershell
$env:VLLM_HOST_PORT="8002"
docker compose -f docker-compose.vllm.yml up --build -d
$env:DSPY_API_BASE="http://localhost:8002/v1"
```

Then verify:

```bash
python scripts/verify_vllm_smoke.py
```

See `docker/vllm/README.md` for model override examples.

## Run

### Start the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

### Health check

```bash
curl http://localhost:8080/health
```

### Type 2 — Physics problem

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "T2_0001",
    "type": "type2",
    "query": "A resistor of 10 ohm is connected to a 5V battery. Find the current.",
    "premises": [],
    "options": []
  }'
```

`/predict` returns a JSON list, even for one query: `[{"query_id": "...", "answer": "...", "unit": "...", "explanation": "...", "premises_used": [], "reasoning": ...}]`.

### Type 1 — Logic problem

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "T1_0001",
    "type": "type1",
    "query": "Can a penguin fly?",
    "premises": ["All birds can fly", "A penguin is a bird"],
    "options": ["Yes", "No", "Uncertain"]
  }'
```

### Batch processing (Stage 0 parser)

Split the dataset into 80% training and 20% testing:

```bash
python scripts/split_dataset.py \
  --dataset Dataset/Physics_Problems_Text_Only.csv \
  --train-ratio 0.8 \
  --seed 42
```

Then run the full API pipeline on the training split:

```bash
python scripts/run_api_on_dataset.py \
  --dataset Dataset/Physics_Problems_Text_Only_train.csv \
  --output-dir outputs/api_train \
  --query-type type2 \
  --type2-solver-mode deterministic
```

Run the same full API pipeline on the held-out test split:

```bash
python scripts/run_api_on_dataset.py \
  --dataset Dataset/Physics_Problems_Text_Only_test.csv \
  --output-dir outputs/api_test \
  --query-type type2 \
  --type2-solver-mode deterministic
```

Analyze full pipeline errors after a run:

```bash
python scripts/analyze_api_results.py \
  --results outputs/api_test/api_results.jsonl \
  --output-dir outputs/api_test/error_analysis
```

## Final Type2 Results

Confirmed final evaluation:

| Metric | Result |
|--------|--------|
| Dataset | `Dataset\Physics_Problems_Text_Only.csv` |
| Output directory | `outputs\api_full_current_repaired3_20260610` |
| Total rows | 1352 |
| Pipeline completion / trace PASS rate | 1345 / 1352 = 99.48% |
| Objective answer accuracy | 1251 / 1282 = 97.58% |
| Numeric accuracy | 1223 / 1254 = 97.53% |
| Symbolic formula accuracy | 25 / 25 = 100% |
| Choice accuracy | 3 / 3 = 100% |
| Concept text accuracy | 68 / 70 = 97.14% |

Trace PASS means the pipeline completed and produced an internally valid trace. It is not the same as final answer correctness; final accuracy is measured by the objective answer evaluator.

## Final Submission Stabilization

Print the final PowerShell regression commands:

```powershell
python scripts/final_submission_check.py
```

Run syntax checks:

```powershell
python scripts/final_submission_check.py --run-syntax
```

Run the final full hybrid Type2 evaluation:

```powershell
$env:DSPY_MODEL="openai/qwen3-8b-awq"
$env:DSPY_API_BASE="http://localhost:8002/v1"
$env:DSPY_API_KEY="EMPTY"

python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only.csv `
  --output-dir outputs\final_full_hybrid `
  --query-type type2 `
  --type2-solver-mode hybrid
```

After a final full run, generate error and false-pass analysis:

```powershell
python scripts\analyze_api_results.py `
  --results outputs\final_full_hybrid\api_results.jsonl `
  --output-dir outputs\final_full_hybrid\error_analysis `
  --sample-limit 50
```

See `FINAL_REPORT.md`, `FINAL_REPORT_SUMMARY.md`, and `FINAL_SUBMISSION_CHECKLIST.md` for the final reports, submission checklist, current metrics, reproduction commands, and model-compliance notes.

## Model Compliance

The final hybrid configuration uses local vLLM with `qwen3-8b-awq`, an open-source model within the <=8B constraint. No GPT, Claude, Gemini, or other closed hosted models are required for the final Type2 run.

## Pipeline Overview

### Type 2 (Physics)

```
Question text
  -> parse
  -> formula retrieval
  -> deterministic symbolic solve
  -> step verification
  -> optional diagnosis / guarded repair
  -> final-answer validation
  -> response
```

Type2 is deterministic-first. In hybrid mode, the system calls the local vLLM-backed DSPy fallback only when deterministic solving fails or needs guarded repair. Correct deterministic traces are not sent to the LLM.

### Type 1 (Logic)

```
Premises + Question
  → Parse (extract premises, classify question format)
  → Route: Yes/No → Z3 solver | MCQ/Open → LLM reasoner
  → Verify (optional confidence boost)
  → Response
```

## Configuration

Key parameters in `SolverConfig` (`config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beam_n` | 3 | Formula search breadth |
| `type1_enabled` | True | Enable Type 1 pipeline |
| `type2_enabled` | True | Enable Type 2 pipeline |
| `type1_use_z3` | True | Use Z3 for Yes/No questions |
| `stage0_use_llm_fallback` | True | LLM recovery for failed parses |
| `stage0_cache_enabled` | True | Reuse cached Stage 0 results |

## Evaluation scripts

```bash
# Analyze Stage 0 parse failures
python scripts/analyze_stage0_failures.py

# Evaluate Stage 1 formula retrieval
python scripts/eval_stage1_retrieval.py

# Diagnose LLM fallback behavior
python scripts/diagnose_llm_fallback.py
```
