# IJCNN-Qiwei

Object-oriented Type 1 preprocessing and classification pipeline.

It preserves the existing method:

- SentenceTransformer semantic embeddings
- KMeans coarse clustering
- KNN fine answer-class classification
- TF-IDF cluster keywords and statistics

It also adds deterministic Type 1 interfaces for each question:

- `type1_question_format`: `multiple_choice`, `yes_no_judgment`, `open_ended`, or `logic_query`
- `type1_reasoning_task`: `conclusion_selection`, `entailment_judgment`, `truth_judgment`, etc.
- `type1_logic_structures`: conditional chain, universal rule, negation, numeric constraint, etc.
- `type1_solver_route`: semantic retrieval routes such as `semantic_mcq_retrieval` or `semantic_entailment_retrieval`
- `type1_api_payload`: normalized payload for a Type 1 solver API

Run from this folder:

```bash
pip install -r requirements.txt

python run_type1_preprocessing.py \
  --input ../Logic_Based_Educational_Queries.json \
  --output processed_logic_queries_classified.json \
  --stats-output logic_query_classification_statistics.json
```

Use a locally cached embedding model:

```bash
python run_type1_preprocessing.py --local-files-only
```

## Type 1 Stage 0 Parser

Stage 0 uses the semantic parsing and hybrid retrieval route:

1. Normalize the Type 1 payload.
2. Use Qwen2.5-1.5B-style semantic decomposition to split text into fine-grained units.
3. Initialize a structured attribute base from semantic tree nodes.
4. Map semantic units with BGE vectors.
5. Match query semantics against the structured attribute base.
6. Branch matched attributes to structured output and preserve unmatched text as natural-language stream.
7. Run a deterministic gate over match quality and answer shape.

Install dependencies:

```bash
pip install -r requirements.txt
```

Programmatic interface:

```python
from ijcnn_qiwei import SemanticHybridConfig, Type1SemanticHybridParser

parser = Type1SemanticHybridParser(SemanticHybridConfig())
result = parser.parse({
    "premises-NL": ["If A then B.", "A."],
    "question": "Does B follow according to the premises?"
})
```

## Type 1 Auto Evaluation

Run the Stage 0 parser on the full labeled Type 1 dataset and compute accuracy:

```bash
python run_type1_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --output type1_stage0_eval_results.json \
  --summary-output type1_stage0_eval_summary.json
```

By default, failed or weak semantic matches are handled by a local Stage 0
semantic agent. The agent rescans retrieved evidence and premises, then
rescoring options or yes/no decisions without asking an LLM to produce the
answer.

Disable the agent for pure semantic-retrieval evaluation:

```bash
python run_type1_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --disable-agent-fallback
```

Smoke test a small subset first:

```bash
python run_type1_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --limit 5
```

The summary includes overall accuracy, gate pass rate, structured hit rate,
mean top match score, error rate, answer confusion, and grouped metrics by
question format, reasoning task, solver route, and answer source. The
`stage0_agent` section shows how often the local semantic agent was used.

## Semantic Tree RAG

An optional Type 1 parser enhancement builds a semantic tree before reasoning:

- Qwen2.5-1.5B-Instruct decomposes premises/questions into semantic units.
- BGE embeds semantic nodes and retrieves relevant evidence.
- The retrieved structured evidence is used as the Stage 0 parser output and
  can be fed to later RAG reasoning stages.

Inspect the semantic tree for one question:

```bash
python run_semantic_tree_rag.py \
  --input ../Logic_Based_Educational_Queries.json \
  --record-index 0 \
  --question-index 1 \
  --segmenter-api-base http://localhost:8001/v1 \
  --embedding-model BAAI/bge-small-en-v1.5
```

This debug command is only for inspecting the semantic tree. The main Type 1
path is the batch evaluation command above.

If Qwen/BGE are unavailable, the code falls back to deterministic segmentation
and a small hash-vector index so the interfaces remain testable.

## Type 1 Stage 0-3 Pipeline

After Stage 0, the Type 1-only multi-stage pipeline adds:

- Stage 1: build an evidence graph from the Type 1 FOL/NL premises.
- Stage 2: run a causal inference world model over action-effect rules.
- Stage 3: fuse the Stage 2 answer with the Stage 0 semantic result and gate it.

The causal world model treats facts as environment state variables and rules as
text actions/interventions. A rule application is represented as
`do(cause_variables) -> effect_variable`. During rollout, each imagined
action-effect transition matches semantic evidence against logical atoms. The
imagination space also mathematically couples premises with operations such as
intersection, union, ordered pair, and reverse pair. The final decision is made
from an answer probability distribution:

```text
answer_distribution = { Yes, Uncertain, No }
```

`causal_score = P(Yes) - P(No)` is kept only as a compatibility/debug value.
The planner is recurrent: it repeatedly chooses the strongest imagined
trajectory or premise coupling, updates a belief distribution, and stops when
confidence or convergence is reached. Saved results include
`causal_inference.belief_states` for inspecting each update round.

Run the full Type 1 pipeline:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --output type1_pipeline_eval_results.json \
  --summary-output type1_pipeline_eval_summary.json
```

For a fast smoke test:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --limit 5 \
  --local-files-only \
  --disable-agent-fallback
```

Use `--include-evidence-graph` when you want the saved result file to contain
the full Stage 1 facts, rules, derived facts, and proof chains.

Control the causal world model:

```bash
python run_type1_pipeline_evaluation.py \
  --input ../Logic_Based_Educational_Queries.json \
  --causal-max-steps 6 \
  --causal-top-k 4 \
  --causal-yes-probability-threshold 0.45 \
  --causal-no-probability-threshold 0.45 \
  --premise-coupling-top-k 8 \
  --recurrent-planning-rounds 4 \
  --belief-confidence-threshold 0.72 \
  --belief-convergence-delta 0.035
```
