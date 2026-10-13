from __future__ import annotations

from evalmutationbench.annotation import agreement_report
from evalmutationbench.models import FailureFamily, HumanJudgment


def test_perfect_agreement() -> None:
    judgments = [
        HumanJudgment(
            pair_id=f"pair-{pair}",
            reviewer_id=f"reviewer-{reviewer}",
            regression=bool(pair),
            failure_family=FailureFamily.ANSWER if pair else FailureFamily.NONE,
            severity=2 if pair else 0,
            confidence=5,
            evidence_paths=["candidate.final_answer"],
        )
        for pair in range(2)
        for reviewer in range(3)
    ]
    report = agreement_report(judgments)
    assert report["judgments"] == 6
    assert report["regression_fleiss_kappa"] == 1.0
    assert report["failure_family_fleiss_kappa"] == 1.0
    assert report["regression_unanimous_rate"] == 1.0
