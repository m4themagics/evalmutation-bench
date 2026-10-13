# EvalMutationBench

EvalMutationBench tests the tests: can an LLM evaluation or release-gate system detect and localize a known regression in a RAG/tool-agent trace without rejecting a valid semantic change?

The benchmark is deliberately different from agent capability benchmarks. It evaluates the **evaluator**, using paired control/candidate traces and controlled mutations with known provenance.

## Status

`v0.1.0` is an executable benchmark foundation:

- 300 deterministic trace pairs across customer support, procurement, and travel operations;
- 180 development, 60 validation, and 60 label-free challenge records;
- 19 mutation/control operators across retrieval, answer, tools, and state/policy layers;
- strict Pydantic contracts and exported JSON Schemas;
- deterministic, pointwise, pairwise, Ragas-compatible, and DeepEval-compatible adapter entry points;
- scoring, cluster-bootstrap confidence intervals, reports, submission validation, and CI;
- Label Studio configuration for three-reviewer annotation.

The checked-in development and validation labels are **synthetic oracle labels**, not human judgments. Human calibration and external model runs are release work, not fabricated results. See [ROADMAP.md](ROADMAP.md).

## Quick start

Requirements: Python 3.12 or 3.13 and `uv`.

```bash
uv sync --extra dev
uv run evalmut generate-dataset --output-dir data
uv run evalmut validate-dataset data/development.jsonl data/validation.jsonl data/challenge.jsonl
uv run evalmut run \
  --dataset data/development.jsonl \
  --adapter deterministic \
  --output-dir artifacts/deterministic-dev
uv run evalmut score \
  --dataset data/development.jsonl \
  --predictions artifacts/deterministic-dev/predictions.jsonl \
  --output artifacts/deterministic-dev/metrics.json
uv run evalmut report \
  --metrics artifacts/deterministic-dev/metrics.json \
  --adapter deterministic \
  --dataset-name development \
  --output artifacts/deterministic-dev/report.md
```

Run all local acceptance checks:

```bash
make verify
```

## LLM judges: BYOK or local

Install the optional adapter and point LiteLLM at a provider or an OpenAI-compatible local endpoint:

```bash
uv sync --extra llm
export EVALMUT_MODEL='openai/your-model-id'
export EVALMUT_API_BASE='http://127.0.0.1:8000/v1'  # optional
uv run evalmut run --dataset data/development.jsonl --adapter pairwise
```

Provider API keys are read by LiteLLM from the environment. They are never accepted as CLI values or written to manifests.

## Stable evaluator interface

An adapter receives one `TracePair` and returns one `EvaluatorPrediction`:

- regression probability and decision;
- predicted failure family and severity;
- explanation and evidence paths;
- latency and optional provider cost.

Built-ins are `deterministic`, `pointwise`, `pairwise`, `ragas`, and `deepeval`. A custom adapter can be loaded as `module:factory`; the factory receives the parsed YAML configuration and returns `EvaluatorAdapter`.

`ragas` and `deepeval` are compatibility entry points using the stable EvalMutationBench pair contract and structured judge. Framework-native metric bridges should be implemented as plugins because their metric APIs and semantics change independently of this benchmark.

## Dataset design

Every scenario group produces three variants: one valid semantic control and two injected regressions. All variants of a scenario remain in one split.

| Split | Scenario groups | Pairs | Public gold |
|---|---:|---:|---|
| Development | 60 | 180 | Synthetic oracle |
| Validation | 20 | 60 | Synthetic oracle |
| Challenge | 20 | 60 | No |

The challenge file contains inputs only. Submissions contain predictions and an immutable run manifest; maintainers score them against adjudicated labels outside the public repository. No untrusted submission code is executed.

## Metrics

The report keeps complementary evidence separate:

- mutation detection rate and specificity;
- balanced accuracy and Matthews correlation coefficient;
- failure-family macro-F1;
- severity-weighted recall;
- Brier score and expected calibration error;
- p50/p95 latency and cost per 1,000 pairs.

Confidence intervals use cluster bootstrap by scenario group, not by correlated variants.

## Reference pipeline

The core package includes a deterministic state-machine trace generator and a dependency-free BM25 baseline. The optional `reference` extra adds a checkpoint-friendly LangGraph graph plus Sentence Transformers dense retrieval and CrossEncoder reranking:

```bash
uv sync --extra reference
```

These are reference systems, not benchmark dependencies. An evaluator can consume the JSONL traces without LangGraph, PyTorch, a vector database, or network access.

## Publication

- `make site` builds the static leaderboard used by the GitHub Pages workflow.
- `python scripts/publish_hf.py --repo-id owner/name` performs a no-write Hugging Face dry-run.
- Adding `--execute` requires `HF_TOKEN` and publishes only development and validation splits; challenge labels are never uploaded.

## Scope and limitations

- The v0.1 corpus is synthetic and does not establish production safety.
- Oracle mutation labels prove generator intent, not human construct validity.
- Deterministic baselines know benchmark invariants and are not general-purpose judges.
- A high score does not certify access control, transactional correctness, or operational reliability.
- Public task content creates contamination risk; benchmark version and dataset hash are mandatory in every result.

## Licenses

Code is licensed under Apache-2.0. Dataset files under `data/` are licensed under CC BY 4.0; see [DATA_LICENSE](DATA_LICENSE).

