from __future__ import annotations

from pathlib import Path
from typing import Any


def render_markdown(metrics: dict[str, Any], adapter: str, dataset: str) -> str:
    ci = metrics.get("balanced_accuracy_ci_95", {})

    def fmt(value: object) -> str:
        if value is None:
            return "not reported"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# EvalMutationBench report",
        "",
        f"- Adapter: `{adapter}`",
        f"- Dataset: `{dataset}`",
        f"- Scored pairs: {metrics['n']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "balanced_accuracy",
        "mutation_detection_rate",
        "specificity",
        "false_positive_rate",
        "mcc",
        "localization_macro_f1",
        "severity_weighted_recall",
        "brier_score",
        "ece_10",
        "cost_per_1000_pairs_usd",
        "latency_p50_ms",
        "latency_p95_ms",
    ):
        lines.append(f"| {key} | {fmt(metrics.get(key))} |")
    if ci:
        lines.extend(
            [
                "",
                f"Balanced accuracy 95% cluster-bootstrap CI: "
                f"[{fmt(ci.get('low'))}, {fmt(ci.get('high'))}]",
            ]
        )
    lines.extend(
        [
            "",
            "Cost is reported only when every scored prediction contains provider cost data.",
            "Challenge labels are excluded from public scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, metrics: dict[str, Any], adapter: str, dataset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(metrics, adapter, dataset), encoding="utf-8")
