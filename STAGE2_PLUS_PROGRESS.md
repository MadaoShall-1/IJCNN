# Stage 2+ Progress

## Scope

This branch continues from the completed Stage 0 parser and Stage 1 formula retrieval work. The focus is Type 2 Stage 2 and later:

- Stage 2/3 solution execution and verification
- Stage 4 trace diagnosis
- Stage 5 repair
- Stage 6 response assembly
- vLLM/DSPy local model smoke testing
- reproducible Docker setup for the local vLLM server

## Implemented Changes

### Deterministic Stage 2/3 Fallback

`type2/stage2.py` now includes `DeterministicSolveTrace`, a no-LLM solver that:

- initializes the VSO from Stage 0 known quantities and constants
- executes `formula_application` steps with SymPy when the selected formula is checkable
- writes computed outputs back into the VSO
- copies final values for `conclusion` steps
- preserves parser output aliases such as `R_total` while also storing canonical formula variables when useful

The API now uses this deterministic solver when DSPy is unavailable or no DSPy model is configured. This avoids returning empty/stub Type 2 responses when vLLM is not running.

### DSPy/vLLM Path Hardening

When `DSPY_MODEL` is configured, the API enters `dspy_llm` mode. In that mode, formula steps that SymPy can solve are still solved deterministically before falling back to LLM generation. This keeps simple numeric physics problems stable while preserving the LLM path for steps that need it.

### Stage 5 Repair Fix

`type2/stage5.py` now preserves original Stage 0 known quantities when the first wrong step is the first step in the trace. Previously, repair rollback could produce an empty VSO in that case, causing repair to fail even when an alternative formula was available.

### vLLM Docker Setup

Added a reproducible local vLLM setup:

- `docker/vllm/Dockerfile`
- `docker-compose.vllm.yml`
- `docker/vllm/README.md`

Default local model:

```text
Qwen/Qwen2.5-0.5B-Instruct
served as qwen2.5-0.5b
```

Start command:

```powershell
docker compose -f docker-compose.vllm.yml up --build -d
```

DSPy environment:

```powershell
$env:DSPY_MODEL = "openai/qwen2.5-0.5b"
$env:DSPY_API_BASE = "http://localhost:8000/v1"
$env:DSPY_API_KEY = "EMPTY"
```

## Verification Scripts

Added:

- `scripts/verify_type2_deterministic.py`
- `scripts/verify_vllm_smoke.py`

`verify_type2_deterministic.py` covers:

- Stage 2/3 deterministic SymPy solve
- parser output aliases used by downstream steps
- constant-table variables not being used as canonical problem values
- Stage 5 repair when the first wrong step is the first trace step
- Stage 4 diagnosis for formula selection and chain propagation
- Stage 6 response assembly fields

`verify_vllm_smoke.py` checks:

- vLLM server is reachable at `DSPY_API_BASE`
- served model matches `DSPY_MODEL`
- API loads in `dspy_llm` mode
- a simple Type 2 problem returns `PASS` and `0.5 A`

## Validation Results

Commands run successfully:

```powershell
python scripts\verify_type2_deterministic.py
python scripts\verify_vllm_smoke.py
python -m compileall api.py type2 scripts\verify_type2_deterministic.py scripts\verify_vllm_smoke.py
```

Observed deterministic result:

```text
Ran 7 tests
OK
```

Observed vLLM smoke result:

```text
solver_mode: dspy_llm
answer: 0.5 A
confidence: 0.975
trace_status: PASS
```

## vLLM Model Notes

`Qwen/Qwen2.5-0.5B-Instruct` runs successfully on the local RTX 4070 Laptop GPU with 8 GB VRAM.

`Qwen/Qwen3-8B-AWQ` was also tested. It is public and downloaded, but did not fully start on the 8 GB GPU:

- first attempt failed because requested GPU memory exceeded available free memory
- second attempt loaded most weights with `awq_marlin`, `--enforce-eager`, and `max_model_len=1024`
- final failure was KV cache allocation: no available memory for cache blocks

Current stable local vLLM target remains `qwen2.5-0.5b`.

## Current Status

Stage 2 through Stage 6 now have a working deterministic baseline and targeted regression checks. vLLM integration is wired and smoke-tested through Docker with the 0.5B model. The deterministic path remains the stable fallback when no local model is configured.

