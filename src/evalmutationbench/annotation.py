from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import HumanJudgment, TracePair


def export_label_studio_tasks(pairs: list[TracePair], path: Path) -> None:
    tasks = []
    for pair in pairs:
        tasks.append(
            {
                "id": pair.pair_id,
                "data": {
                    "pair_id": pair.pair_id,
                    "scenario": json.dumps(pair.scenario.model_dump(mode="json"), indent=2),
                    "control": json.dumps(pair.control.model_dump(mode="json"), indent=2),
                    "candidate": json.dumps(pair.candidate.model_dump(mode="json"), indent=2),
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _complete_matrix(
    judgments: list[HumanJudgment], value: str, reviewers_per_pair: int
) -> tuple[list[str], list[list[str]]]:
    by_pair: dict[str, list[HumanJudgment]] = defaultdict(list)
    for judgment in judgments:
        by_pair[judgment.pair_id].append(judgment)
    duplicate_keys = Counter((item.pair_id, item.reviewer_id) for item in judgments)
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate reviewer judgments: {duplicates[:5]}")
    incomplete = [pair_id for pair_id, rows in by_pair.items() if len(rows) != reviewers_per_pair]
    if incomplete:
        raise ValueError(f"pairs without exactly {reviewers_per_pair} judgments: {incomplete[:5]}")
    pair_ids = sorted(by_pair)
    matrix = [[str(getattr(item, value)) for item in by_pair[pair_id]] for pair_id in pair_ids]
    return pair_ids, matrix


def fleiss_kappa_nominal(
    judgments: list[HumanJudgment], value: str = "regression", reviewers_per_pair: int = 3
) -> float:
    _, matrix = _complete_matrix(judgments, value, reviewers_per_pair)
    categories = sorted({category for row in matrix for category in row})
    n = reviewers_per_pair
    observed = []
    totals = Counter(category for row in matrix for category in row)
    for row in matrix:
        counts = Counter(row)
        observed.append(sum(count * count for count in counts.values()) - n)
    p_bar = sum(observed) / (len(matrix) * n * (n - 1))
    p_e = sum((totals[category] / (len(matrix) * n)) ** 2 for category in categories)
    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_bar, 1.0) else float("nan")
    return (p_bar - p_e) / (1 - p_e)


def agreement_report(judgments: list[HumanJudgment], reviewers_per_pair: int = 3) -> dict[str, Any]:
    pair_ids, regression_matrix = _complete_matrix(judgments, "regression", reviewers_per_pair)
    _complete_matrix(judgments, "failure_family", reviewers_per_pair)
    unanimous = sum(len(set(row)) == 1 for row in regression_matrix)
    return {
        "pairs": len(pair_ids),
        "judgments": len(judgments),
        "reviewers_per_pair": reviewers_per_pair,
        "regression_fleiss_kappa": fleiss_kappa_nominal(
            judgments, "regression", reviewers_per_pair
        ),
        "failure_family_fleiss_kappa": fleiss_kappa_nominal(
            judgments, "failure_family", reviewers_per_pair
        ),
        "regression_unanimous_rate": unanimous / len(pair_ids),
    }
