from __future__ import annotations

import pytest

from evalmutationbench.adapters.deterministic import DeterministicAdapter
from evalmutationbench.dataset import build_pairs
from evalmutationbench.metrics import cluster_bootstrap_ci, score_predictions


def test_deterministic_baseline_scores_and_bootstraps() -> None:
    pairs = [pair for pair in build_pairs() if pair.gold is not None]
    predictions = [DeterministicAdapter().evaluate(pair) for pair in pairs]
    metrics = score_predictions(pairs, predictions)
    assert metrics["n"] == 240
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["cost_per_1000_pairs_usd"] == 0.0
    ci = cluster_bootstrap_ci(pairs, predictions, samples=20, seed=7)
    assert 0 <= ci["low"] <= ci["high"] <= 1


def test_missing_prediction_fails_closed() -> None:
    pairs = [pair for pair in build_pairs() if pair.gold is not None][:2]
    prediction = DeterministicAdapter().evaluate(pairs[0])
    with pytest.raises(ValueError, match="missing"):
        score_predictions(pairs, [prediction])
