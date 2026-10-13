from __future__ import annotations

from collections import Counter

from .models import EvaluatorPrediction, TracePair


def validate_submission(
    pairs: list[TracePair], predictions: list[EvaluatorPrediction], require_complete: bool = True
) -> dict[str, int]:
    expected = {pair.pair_id for pair in pairs}
    counts = Counter(item.pair_id for item in predictions)
    duplicates = sorted(pair_id for pair_id, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate prediction IDs: {duplicates[:5]}")
    unknown = sorted(set(counts) - expected)
    if unknown:
        raise ValueError(f"unknown prediction IDs: {unknown[:5]}")
    missing = sorted(expected - set(counts))
    if require_complete and missing:
        raise ValueError(f"missing prediction IDs: {missing[:5]}")
    errors = sum(item.error is not None for item in predictions)
    return {
        "expected": len(expected),
        "predictions": len(predictions),
        "missing": len(missing),
        "errors": errors,
    }
