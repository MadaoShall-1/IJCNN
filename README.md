# IJCNN-Qiwei

Type 1 reasoning pipeline with one retained architecture:

```text
Stage 0: semantic parsing and retrieval
Stage 1: evidence graph construction
Stage 2: external causal/world-model imagination
Transformer Brain: internal latent imagination and answer distribution
Stage 3: final gate and evaluation
```

No Z3 path is used. The project keeps the semantic parser, evidence graph,
causal imagination, and local transformer-style brain world model.

## Architecture

Stage 0 normalizes the Type 1 payload, builds semantic units, and retrieves
structured evidence.

Stage 1 builds a lightweight evidence graph from FOL/NL premises.

Stage 2 treats facts as state variables and rules as action-effect
interventions. It imagines causal transitions, premise couplings, and option
probability distributions.

The Transformer Brain receives those externally imagined candidate states and
runs the local architecture below for `N` imagination layers:

```text
AdaLN(condition)
Block-Wise SSM
Scale + residual
AdaLN(condition)
Multi-Head Frame Local Attention
Scale + residual
AdaLN(condition)
FFN
Scale + residual
```

It then produces:

```text
answer_distribution
external_imagination_states
internal_imagination_steps
final_answer
```

The retained training path uses the Transformer Brain candidate latents and
answer distribution as the main evaluator. It does not report or optimize a
separate comparison path.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a smoke test:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --limit 20 \
  --local-files-only
```

Run the full Type 1 pipeline:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --output type1_pipeline_eval_results.json \
  --summary-output type1_pipeline_eval_summary.json \
  --local-files-only
```

Train the retained Transformer Brain readout with an 8/2 split and validate it:

```bash
python run_type1_brain_training.py \
  --input ../Logic_Based_Educational_Queries.json \
  --local-files-only \
  --epochs 100 \
  --learning-rate 0.025 \
  --output type1_brain_train_eval_results.json \
  --summary-output type1_brain_train_eval_summary.json \
  --model-output type1_trained_brain_readout.json
```

Useful pipeline controls:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --causal-max-steps 6 \
  --causal-top-k 4 \
  --premise-coupling-top-k 8 \
  --recurrent-planning-rounds 4 \
  --enable-global-option-distribution \
  --enable-transformer-brain-world-model \
  --transformer-brain-imagination-layers 2 \
  --transformer-brain-attention-heads 4 \
  --transformer-brain-frame-local-window 4 \
  --transformer-brain-ssm-block-size 4
```

Saved results include the full transformer brain trace under:

```text
transformer_brain_world_model
```
