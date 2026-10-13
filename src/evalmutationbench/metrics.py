from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import EvaluatorPrediction, FailureFamily, TracePair


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _macro_f1(gold: Sequence[FailureFamily], predicted: Sequence[FailureFamily]) -> float:
    labels = list(FailureFamily)
    scores: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _ece(gold: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = len(gold)
    if total == 0:
        return float("nan")
    value = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            mask = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(
            float(gold[mask].mean()) - float(probability[mask].mean())
        )
    return value


def score_predictions(
    pairs: Sequence[TracePair], predictions: Sequence[EvaluatorPrediction]
) -> dict[str, Any]:
    if not pairs or not any(pair.gold is not None for pair in pairs):
        raise ValueError("scoring requires labeled pairs")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("duplicate dataset pair IDs")
    if len({prediction.pair_id for prediction in predictions}) != len(predictions):
        raise ValueError("duplicate prediction pair IDs")
    if any(prediction.error is not None for prediction in predictions):
        raise ValueError("failed predictions cannot be scored; complete the run first")
    labeled = {pair.pair_id: pair for pair in pairs if pair.gold is not None}
    prediction_map = {prediction.pair_id: prediction for prediction in predictions}
    missing = sorted(set(labeled) - set(prediction_map))
    extra = sorted(set(prediction_map) - {pair.pair_id for pair in pairs})
    if missing:
        raise ValueError(f"predictions missing {len(missing)} labeled pairs: {missing[:5]}")
    if extra:
        raise ValueError(f"predictions contain unknown pair IDs: {extra[:5]}")

    ordered_pairs = [labeled[pair_id] for pair_id in sorted(labeled)]
    ordered_predictions = [prediction_map[pair.pair_id] for pair in ordered_pairs]
    gold = np.array(
        [bool(pair.gold and pair.gold.regression) for pair in ordered_pairs], dtype=bool
    )
    predicted = np.array([item.predicted_regression for item in ordered_predictions], dtype=bool)
    probability = np.array(
        [item.regression_probability for item in ordered_predictions], dtype=float
    )

    tp = int(np.sum(gold & predicted))
    tn = int(np.sum(~gold & ~predicted))
    fp = int(np.sum(~gold & predicted))
    fn = int(np.sum(gold & ~predicted))
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, denominator)

    severity_numerator = sum(
        pair.gold.severity
        for pair, prediction in zip(ordered_pairs, ordered_predictions, strict=True)
        if pair.gold and pair.gold.regression and prediction.predicted_regression
    )
    severity_denominator = sum(
        pair.gold.severity for pair in ordered_pairs if pair.gold and pair.gold.regression
    )
    costs = [item.cost_usd for item in ordered_predictions if item.cost_usd is not None]
    latencies = [item.latency_ms for item in ordered_predictions]
    result: dict[str, Any] = {
        "n": len(ordered_pairs),
        "coverage": _safe_div(len(ordered_predictions), len(labeled)),
        "mutation_detection_rate": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "mcc": mcc,
        "localization_macro_f1": _macro_f1(
            [pair.gold.failure_family for pair in ordered_pairs if pair.gold],
            [item.predicted_failure_family for item in ordered_predictions],
        ),
        "severity_weighted_recall": _safe_div(severity_numerator, severity_denominator),
        "brier_score": float(np.mean((probability - gold.astype(float)) ** 2)),
        "ece_10": _ece(gold.astype(float), probability),
        "cost_per_1000_pairs_usd": (
            float(sum(costs) / len(costs) * 1000)
            if len(costs) == len(ordered_predictions)
            else None
        ),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
    return result


def cluster_bootstrap_ci(
    pairs: Sequence[TracePair],
    predictions: Sequence[EvaluatorPrediction],
    metric: str = "balanced_accuracy",
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    score_predictions(pairs, predictions)
    groups: dict[str, list[TracePair]] = defaultdict(list)
    for pair in pairs:
        if pair.gold is not None:
            groups[pair.scenario_group].append(pair)
    if not groups:
        raise ValueError("no labeled scenario groups")
    prediction_map = {prediction.pair_id: prediction for prediction in predictions}
    group_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_pairs: list[TracePair] = []
        sampled_predictions: list[EvaluatorPrediction] = []
        for sample_index, group_id in enumerate(selected_ids):
            for pair in groups[str(group_id)]:
                synthetic_id = f"bootstrap-{sample_index}-{pair.pair_id}"
                sampled_pairs.append(pair.model_copy(update={"pair_id": synthetic_id}))
                prediction = prediction_map[pair.pair_id]
                sampled_predictions.append(prediction.model_copy(update={"pair_id": synthetic_id}))
        estimate = score_predictions(sampled_pairs, sampled_predictions)[metric]
        if not math.isnan(float(estimate)):
            estimates.append(float(estimate))
    if not estimates:
        raise ValueError(f"bootstrap produced no finite estimates for {metric}")
    return {
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "samples": float(len(estimates)),
    }
