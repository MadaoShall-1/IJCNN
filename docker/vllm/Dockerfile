FROM vllm/vllm-openai:latest

ENV VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV VLLM_SERVED_MODEL_NAME=qwen2.5-0.5b
ENV VLLM_MAX_MODEL_LEN=1024
ENV VLLM_GPU_MEMORY_UTILIZATION=0.65
ENV VLLM_EXTRA_ARGS=

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["vllm serve \"$VLLM_MODEL\" --served-model-name \"$VLLM_SERVED_MODEL_NAME\" --host 0.0.0.0 --port 8000 --max-model-len \"$VLLM_MAX_MODEL_LEN\" --gpu-memory-utilization \"$VLLM_GPU_MEMORY_UTILIZATION\" $VLLM_EXTRA_ARGS"]
