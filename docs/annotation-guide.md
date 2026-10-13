# Human annotation guide

Reviewers see the scenario, control trace, and candidate trace in randomized order. They do not see the mutation name or oracle label.

For each pair record:

1. `regression`: whether the candidate is materially worse or invalid;
2. `failure_family`: none, retrieval, answer, tools, or state_policy;
3. `severity`: 0 for controls, 1 minor, 2 material, 3 unsafe/incorrect state;
4. `confidence`: 1–5;
5. `evidence_paths`: exact trace fields supporting the decision;
6. optional notes.

Judge behavior, not prose style. A shorter answer, reordered evidence, or a different valid tool order is not a regression. A plausible final answer does not excuse a wrong state transition, missing approval, failed tool call, or unsupported citation.

Every pair receives three independent judgments. Reviewers must not discuss cases before submitting. Disagreements are adjudicated only after the independent pass, with the original judgments retained in the audit file.

