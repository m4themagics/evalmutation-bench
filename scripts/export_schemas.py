from __future__ import annotations

import json
from pathlib import Path

from evalmutationbench.models import (
    EvaluatorPrediction,
    HumanJudgment,
    RunManifest,
    TracePair,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "schemas"

MODELS = {
    "trace-pair.schema.json": TracePair,
    "evaluator-prediction.schema.json": EvaluatorPrediction,
    "run-manifest.schema.json": RunManifest,
    "human-judgment.schema.json": HumanJudgment,
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        path = OUTPUT / filename
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
