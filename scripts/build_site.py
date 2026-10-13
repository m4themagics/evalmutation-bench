# ruff: noqa: E501 - keeping the generated HTML readable is more useful than wrapping literals.
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def main() -> None:
    payload = json.loads((ROOT / "results/leaderboard.json").read_text(encoding="utf-8"))
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(entry['system'])}</td>"
        f"<td>{fmt(entry['balanced_accuracy'])}</td>"
        f"<td>{fmt(entry['specificity'])}</td>"
        f"<td>{fmt(entry['localization_macro_f1'])}</td>"
        f"<td>{fmt(entry['cost_per_1000_pairs_usd'])}</td>"
        f"<td>{fmt(entry['latency_p95_ms'])}</td>"
        "</tr>"
        for entry in payload["entries"]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EvalMutationBench</title>
  <style>
    body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 1080px; margin: 3rem auto; padding: 0 1rem; color: #172033; }}
    h1 {{ font-size: clamp(2rem, 6vw, 4rem); margin-bottom: .25rem; }}
    .lede {{ font-size: 1.2rem; max-width: 70ch; }}
    .notice {{ background: #fff6d8; border-left: 4px solid #c88900; padding: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d8dce5; padding: .7rem; text-align: left; }}
    th {{ background: #f3f5f9; }}
    code {{ background: #f3f5f9; padding: .1rem .3rem; }}
  </style>
</head>
<body>
  <h1>EvalMutationBench</h1>
  <p class="lede">A mutation-testing benchmark for LLM evaluation and release-gate systems.</p>
  <p class="notice"><strong>v{payload["benchmark_version"]}:</strong> results use {payload["label_status"]} labels on the {payload["dataset_split"]} split. They are engineering baselines, not human-validated model rankings.</p>
  <h2>What is measured</h2>
  <p>Can an evaluator detect and localize an injected RAG/agent regression without rejecting a valid semantic change?</p>
  <h2>Leaderboard</h2>
  <table>
    <thead><tr><th>System</th><th>Balanced accuracy</th><th>Specificity</th><th>Localization F1</th><th>Cost / 1k</th><th>p95 ms</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p>Scores require a dataset version, hash, immutable manifest, and complete prediction file. No aggregate score hides specificity, localization, cost, or latency.</p>
  <h2>Reproduce</h2>
  <pre><code>make verify</code></pre>
</body>
</html>
"""
    output = ROOT / "site/index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
