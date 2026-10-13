from __future__ import annotations

from collections import Counter, defaultdict

from evalmutationbench.dataset import build_pairs, validate_split_isolation
from evalmutationbench.models import Domain, Split


def test_expected_split_sizes_and_isolation() -> None:
    pairs = build_pairs()
    validate_split_isolation(pairs)
    assert Counter(pair.split for pair in pairs) == {
        Split.DEVELOPMENT: 180,
        Split.VALIDATION: 60,
        Split.CHALLENGE: 60,
    }
    owners: dict[str, set[Split]] = defaultdict(set)
    for pair in pairs:
        owners[pair.scenario_group].add(pair.split)
    assert all(len(splits) == 1 for splits in owners.values())


def test_domains_and_control_rate() -> None:
    pairs = build_pairs()
    assert {pair.scenario.domain for pair in pairs} == set(Domain)
    public = [pair for pair in pairs if pair.gold is not None]
    controls = [pair for pair in public if pair.gold and not pair.gold.regression]
    assert len(controls) / len(public) >= 0.20
