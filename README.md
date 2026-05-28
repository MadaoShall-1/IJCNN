# Education System — Physics & Logic Solver

An AI-powered educational solver that handles two types of queries:
- **Type 1**: Logic-based reasoning (premises + yes/no, MCQ, or open-ended questions)
- **Type 2**: Physics calculation (multi-stage pipeline: parse → formula retrieval → symbolic solve → diagnose → repair → response)

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

The system uses [DSPy](https://github.com/stanfordnlp/dspy) for LLM calls. Set environment variables before starting:

```bash
# Windows PowerShell
$env:DSPY_MODEL = "openai/qwen3-8b"
$env:DSPY_API_BASE = "http://localhost:8000/v1"
$env:DSPY_API_KEY = "your-api-key"

# Linux / macOS
export DSPY_MODEL="openai/qwen3-8b"
export DSPY_API_BASE="http://localhost:8000/v1"
export DSPY_API_KEY="your-api-key"
```

Or edit `config.py` directly via `SolverConfig`.

### 3. (Optional) Local model

If using a local model (e.g. Qwen3-8B with llama.cpp or vLLM), start the model server first, then point `DSPY_API_BASE` to it.

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
    "question": "A resistor of 10 ohm is connected to a 5V battery. Find the current.",
    "id": "P001"
  }'
```

### Type 1 — Logic problem

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "type1",
    "premises-NL": ["All birds can fly", "A penguin is a bird"],
    "questions": [{"text": "Can a penguin fly?", "format": "yes_no"}]
  }'
```

### Batch processing (Stage 0 parser)

```bash
python scripts/run_stage0_on_dataset.py \
  --dataset Dataset/Physics_Problems_Text_Only.csv \
  --output-dir outputs/stage0_with_llm_v2 \
  --use-llm-fallback \
  --limit 100
```

## Pipeline Overview

### Type 2 (Physics) — 6 Stages

```
Question text
  → Stage 0: Parse (extract quantities, target, domain)
  → Stage 1: Formula retrieval (beam search, n=3)
  → Stage 2: Symbolic solve (SymPy substitution)
  → Stage 4: Diagnose (if solve failed)
  → Stage 5: Repair (formula substitution retry)
  → Stage 6: Response assembly (answer + explanation)
```

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
