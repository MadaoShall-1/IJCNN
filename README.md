# EXACT 2026 — Team superNB

Solution for the **2nd International XAI Challenge for Transparent Educational
Question-Answering (EXACT 2026, IEEE IJCNN)**: a single public `/predict`
endpoint that answers logic-based educational queries (Type 1) and physics
calculation problems (Type 2) with explanations and structured reasoning.

## Architecture

```
POST /predict  (root api.py — the only HTTP server in the project)
  |
  +-- type == "type1"  ->  type1/IJCNN-Qiwei
  |     retained WM/SSM/Transformer classifier  (choice questions)
  |     vLLM reasoner                           (free-form questions)
  |     vLLM premise selector                   (premises_used / P2)
  |
  +-- type == "type2"  ->  type2/ staged physics pipeline
        Stage 0 parse -> formula retrieval (beam) -> deterministic SymPy
        solve -> step verification -> diagnose / guarded repair ->
        final-answer validation -> response assembly
        (vLLM used only when the deterministic path fails)
```

Both pipelines share **one** vLLM server running `Qwen/Qwen3-8B-AWQ`
(served as `qwen3-8b-awq`), keeping the total LLM parameters loaded at any
moment within the 8B-class limit. The embedding retriever
(BAAI/bge-small-en-v1.5, 33M) and the retained Type-1 reasoner (<10M) are
non-LLM components.

## Results

| Metric | Result |
|---|---|
| Type 2 objective answer accuracy (full official dataset, 1352 problems) | **97.58%** (1251/1282) |
| Type 1 P1 answer accuracy (93 official questions) | **80.6%** (75/93) |
| Type 1 P2 premises_used (exact-set / mean Jaccard, 92 gold questions) | **22.8% / 0.634** (vs 3.3% / 0.356 heuristic baseline) |
| Latency (50-question simulated grading run) | avg ~3 s, max 17 s, 0 timeouts (60 s limit) |

## Repository layout

```
api.py                    # unified FastAPI /predict + /health (single server)
type1/IJCNN-Qiwei/        # Type 1: retained model, training & inference code
  ijcnn_qiwei/type1_predictor.py          # answering chain (no HTTP)
  ijcnn_qiwei/type1_retained_predictor.py # weights loader / inference
  type1_backtracking_trace_best_model.json# retained model weights
type2/                    # Type 2: staged physics pipeline (gitlink ->
                          #   branch final-submission-type2 in this repo)
  type2/pipeline.py       # pipeline capability module (no HTTP)
  parser/  type2/stage*.py  scripts/
solution.pdf  urls.txt  notation_mapping.csv  superNB.zip   # submission package
TYPE2_SUBMISSION_TODO.md  # deployment runbook & submission checklist
```

## Running

```powershell
# 1. vLLM (docker, host port 8002, model qwen3-8b-awq)
docker start exact-vllm        # or: cd type2; docker compose -f docker-compose.vllm.yml up -d

# 2. unified API (reads type1/IJCNN-Qiwei/.env automatically)
python -m uvicorn api:app --host 0.0.0.0 --port 8080

# 3. health check — expect type1_rag_backend "bge" and type2_solver_mode "dspy_llm"
curl http://localhost:8080/health
```

Example request (unified EXACT schema):

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{
  "query_id": "T2_0001", "type": "type2",
  "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
  "premises": [], "options": []
}'
# -> [{"query_id":"T2_0001","answer":"5","unit":"A","explanation":"...","premises_used":[],"reasoning":{"type":"cot","steps":[...]}}]
```

Evaluation helpers: `run_type2_sample.py` (50-sample Type 2 regression),
`type1/IJCNN-Qiwei/scripts/eval_type1_records.py` (Type 1 dataset eval),
`type2/scripts/run_api_on_dataset.py` (full Type 2 dataset run).

## Branches

- `main` / `final-submission` — full project (this snapshot)
- `final-submission-type2` — type2 pipeline history (referenced as gitlink)
- `Ye`, `submission-version` — development history
