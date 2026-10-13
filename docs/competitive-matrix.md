# Competitive boundary

| Project | Primary object evaluated | EvalMutationBench difference |
|---|---|---|
| tau-bench / tau2 / tau3 | Agent task completion under policy and tools | Evaluates whether an evaluator detects a controlled regression |
| LayerRAG-Bench | Agentic RAG reliability under cross-layer faults | Treats paired faults as probes of evaluation-system sensitivity and specificity |
| MEMERAG | Meta-evaluation of RAG answer metrics | Adds tool trajectories, state transitions, policy/approval failures, and release decisions |
| Regression-gate tools | Detect change between application versions | Supplies a framework-neutral, labeled mutation corpus for testing those gates |

The benchmark does not claim that mutation testing replaces production monitoring or human review.

