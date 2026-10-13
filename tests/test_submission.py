from __future__ import annotations

import pytest

from evalmutationbench.adapters.deterministic import DeterministicAdapter
from evalmutationbench.dataset import build_pairs
from evalmutationbench.submission import validate_submission


def test_complete_submission() -> None:
    pairs = build_pairs()[:3]
    predictions = [DeterministicAdapter().evaluate(pair) for pair in pairs]
    result = validate_submission(pairs, predictions)
    assert result == {"expected": 3, "predictions": 3, "missing": 0, "errors": 0}


def test_duplicate_submission_rejected() -> None:
    pair = build_pairs()[0]
    prediction = DeterministicAdapter().evaluate(pair)
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission([pair], [prediction, prediction])
