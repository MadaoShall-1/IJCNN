FROM vllm/vllm-openai:latest

ENV VLLM_MODEL=Qwen/Qwen3-8B-AWQ
ENV VLLM_SERVED_MODEL_NAME=qwen3-8b-awq
ENV VLLM_MAX_MODEL_LEN=2048
ENV VLLM_GPU_MEMORY_UTILIZATION=0.60
ENV VLLM_EXTRA_ARGS="--enforce-eager --generation-config vllm --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-chunked-prefill --no-async-scheduling"
ENV VLLM_USE_V1=0
ENV VLLM_USE_V2_MODEL_RUNNER=0
ENV VLLM_USE_FLASHINFER_SAMPLER=0

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["vllm serve \"$VLLM_MODEL\" --served-model-name \"$VLLM_SERVED_MODEL_NAME\" --host 0.0.0.0 --port 8000 --max-model-len \"$VLLM_MAX_MODEL_LEN\" --gpu-memory-utilization \"$VLLM_GPU_MEMORY_UTILIZATION\" $VLLM_EXTRA_ARGS"]
