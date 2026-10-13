from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .adapters import load_adapter
from .annotation import agreement_report, export_label_studio_tasks
from .dataset import BENCHMARK_VERSION, validate_split_isolation, write_dataset
from .io import dataset_sha256, load_pairs, load_predictions, write_json, write_jsonl
from .metrics import cluster_bootstrap_ci, score_predictions
from .models import EvaluatorPrediction, FailureFamily, HumanJudgment, RunManifest
from .report import write_report
from .submission import validate_submission as check_submission

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise typer.BadParameter("config must contain a YAML object")
    return payload


@app.command("generate-dataset")
def generate_dataset(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data"),
) -> None:
    counts = write_dataset(output_dir)
    typer.echo(json.dumps(counts, indent=2))


@app.command("validate-dataset")
def validate_dataset(paths: Annotated[list[Path], typer.Argument()]) -> None:
    pairs = [pair for path in paths for pair in load_pairs(path)]
    validate_split_isolation(pairs)
    ids = [pair.pair_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise typer.BadParameter("pair IDs are not unique")
    typer.echo(f"valid: {len(pairs)} pairs across {len({p.scenario_group for p in pairs})} groups")


@app.command("export-annotation-tasks")
def export_annotation_tasks(
    dataset: Annotated[Path, typer.Option("--dataset")],
    output: Annotated[Path, typer.Option("--output")] = Path("annotation/tasks.json"),
) -> None:
    pairs = load_pairs(dataset)
    export_label_studio_tasks(pairs, output)
    typer.echo(f"wrote {len(pairs)} Label Studio tasks to {output}")


@app.command("agreement")
def agreement(
    judgments: Annotated[Path, typer.Option("--judgments")],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/agreement.json"),
    reviewers_per_pair: Annotated[int, typer.Option("--reviewers-per-pair")] = 3,
) -> None:
    rows: list[HumanJudgment] = []
    with judgments.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(HumanJudgment.model_validate_json(line))
    result = agreement_report(rows, reviewers_per_pair=reviewers_per_pair)
    write_json(output, result)
    typer.echo(json.dumps(result, indent=2))


@app.command("run")
def run(
    dataset: Annotated[Path, typer.Option("--dataset")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("artifacts/run"),
    adapter: Annotated[str, typer.Option("--adapter")] = "deterministic",
    config: Annotated[Path | None, typer.Option("--config")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    pairs = load_pairs(dataset)
    adapter_config = _config(config)
    if model:
        adapter_config["model"] = model
    evaluator = load_adapter(adapter, adapter_config)
    predictions: list[EvaluatorPrediction] = []
    for pair in pairs:
        try:
            predictions.append(evaluator.evaluate(pair))
        except Exception as exc:
            predictions.append(
                EvaluatorPrediction(
                    pair_id=pair.pair_id,
                    regression_probability=0.5,
                    predicted_regression=False,
                    predicted_failure_family=FailureFamily.NONE,
                    predicted_severity=0,
                    explanation="Evaluation failed.",
                    evidence_paths=[],
                    latency_ms=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    manifest = RunManifest(
        benchmark_version=BENCHMARK_VERSION,
        run_id=str(uuid.uuid4()),
        adapter=evaluator.name,
        adapter_version=evaluator.version,
        model=model,
        dataset_sha256=dataset_sha256(dataset),
        dataset_split=pairs[0].split,
        seed=seed,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        config=adapter_config,
    )
    write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
    errors = sum(item.error is not None for item in predictions)
    typer.echo(f"wrote {len(predictions)} predictions to {output_dir}; errors={errors}")


@app.command("score")
def score(
    dataset: Annotated[Path, typer.Option("--dataset")],
    predictions: Annotated[Path, typer.Option("--predictions")],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/metrics.json"),
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples")] = 1000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    pairs = load_pairs(dataset)
    outputs = load_predictions(predictions)
    metrics = score_predictions(pairs, outputs)
    metrics["balanced_accuracy_ci_95"] = cluster_bootstrap_ci(
        pairs, outputs, samples=bootstrap_samples, seed=seed
    )
    write_json(output, metrics)
    typer.echo(json.dumps(metrics, indent=2))


@app.command("report")
def report(
    metrics: Annotated[Path, typer.Option("--metrics")],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/report.md"),
    adapter: Annotated[str, typer.Option("--adapter")] = "unknown",
    dataset: Annotated[str, typer.Option("--dataset-name")] = "unknown",
) -> None:
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    write_report(output, payload, adapter=adapter, dataset=dataset)
    typer.echo(f"wrote {output}")


@app.command("validate-submission")
def validate_submission(
    dataset: Annotated[Path, typer.Option("--dataset")],
    predictions: Annotated[Path, typer.Option("--predictions")],
) -> None:
    result = check_submission(load_pairs(dataset), load_predictions(predictions))
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
