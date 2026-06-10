# vLLM Server

This Dockerfile serves an OpenAI-compatible vLLM endpoint for DSPy.

Default model:

```text
Qwen/Qwen3-8B
served as qwen3-8b
```

Start:

```powershell
docker compose -f docker-compose.vllm.yml up --build -d
```

If another container or app already uses host port `8000`, publish vLLM on a
different host port:

```powershell
$env:VLLM_HOST_PORT = "8002"
docker compose -f docker-compose.vllm.yml up --build -d
$env:DSPY_API_BASE = "http://localhost:8002/v1"
```

Check:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models
python scripts\verify_vllm_smoke.py
```

Configure the app to use it:

```powershell
$env:DSPY_MODEL = "openai/qwen3-8b"
$env:DSPY_API_BASE = "http://localhost:8000/v1"
$env:DSPY_API_KEY = "EMPTY"
```

Stop:

```powershell
docker compose -f docker-compose.vllm.yml down
```

To try another model, override the environment before `docker compose up`.
For example, a quantized 8B attempt:

```powershell
$env:VLLM_MODEL = "Qwen/Qwen3-8B-AWQ"
$env:VLLM_SERVED_MODEL_NAME = "qwen3-8b-awq"
$env:VLLM_MAX_MODEL_LEN = "1024"
$env:VLLM_GPU_MEMORY_UTILIZATION = "0.75"
$env:VLLM_EXTRA_ARGS = "--quantization awq_marlin --enforce-eager"
docker compose -f docker-compose.vllm.yml up --build -d
```

On the 8 GB RTX 4070 Laptop GPU tested here, `Qwen3-8B-AWQ` loaded most of
the weights but failed to allocate KV cache. The default is now `Qwen3-8B`,
but you may need a larger GPU or a more aggressive quantized/runtime setup.
