from __future__ import annotations

from evalmutationbench.dataset import build_scenario
from evalmutationbench.models import MutationType
from evalmutationbench.mutations import SPECS, apply_mutation


def _changed_sections(before: dict[str, object], after: dict[str, object]) -> set[str]:
    ignored = {"trace_id"}
    return {key for key in before if key not in ignored and before[key] != after[key]}


def test_every_mutation_changes_declared_sections_only() -> None:
    _, control = build_scenario(1)
    for mutation, spec in SPECS.items():
        candidate = apply_mutation(control, mutation)
        changed = _changed_sections(control.model_dump(), candidate.model_dump())
        declared = {path.split(".", maxsplit=1)[1] for path in spec.evidence_paths}
        assert changed
        assert changed <= declared, f"{mutation}: changed={changed}, declared={declared}"


def test_mutations_do_not_modify_input() -> None:
    _, control = build_scenario(2)
    snapshot = control.model_dump_json()
    for mutation in MutationType:
        apply_mutation(control, mutation)
        assert control.model_dump_json() == snapshot
