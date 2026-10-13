from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalmutationbench.dataset import build_pairs
from evalmutationbench.models import FailureFamily, GoldLabel, MutationType


def test_all_generated_pairs_validate() -> None:
    pairs = build_pairs()
    assert len(pairs) == 300
    assert all(pair.model_validate(pair.model_dump()) for pair in pairs)


def test_control_label_rejects_failure_family() -> None:
    with pytest.raises(ValidationError):
        GoldLabel(
            regression=False,
            failure_family=FailureFamily.ANSWER,
            mutation_type=MutationType.SEMANTIC_PARAPHRASE,
            severity=0,
            label_source="synthetic_oracle",
            evidence_paths=[],
        )


def test_challenge_has_no_gold() -> None:
    challenge = [pair for pair in build_pairs() if pair.split.value == "challenge"]
    assert len(challenge) == 60
    assert all(pair.gold is None for pair in challenge)
