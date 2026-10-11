from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from security_steering.schemas import AttackClass, SteeringExample, Variant


@dataclass(frozen=True)
class DatasetReport:
    total_examples: int
    variant_counts: dict[str, int]
    attack_class_counts: dict[str, int]
    family_count: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: str | Path) -> list[SteeringExample]:
    examples: list[SteeringExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                examples.append(SteeringExample.from_dict(payload))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return examples


def write_jsonl(path: str | Path, examples: Iterable[SteeringExample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def group_by_family(
    examples: Iterable[SteeringExample],
) -> dict[str, list[SteeringExample]]:
    families: dict[str, list[SteeringExample]] = defaultdict(list)
    for example in examples:
        families[example.family_id].append(example)
    return dict(families)


def validate_examples(examples: Iterable[SteeringExample]) -> DatasetReport:
    materialized = list(examples)
    errors: list[str] = []
    if not materialized:
        errors.append("dataset cannot be empty")
    example_ids = [example.example_id for example in materialized]
    duplicate_ids = sorted(
        example_id for example_id, count in Counter(example_ids).items() if count > 1
    )
    for example_id in duplicate_ids:
        errors.append(f"duplicate example_id: {example_id}")

    families = group_by_family(materialized)
    for family_id, family_examples in sorted(families.items()):
        if len({e.user_task for e in family_examples}) != 1:
            errors.append(f"family {family_id} has inconsistent user tasks")
        splits = {e.metadata.get("split") for e in family_examples} - {None}
        if len(splits) > 1:
            errors.append(f"family {family_id} crosses dataset splits")
        variants = {example.variant for example in family_examples}
        attack_examples = [
            example for example in family_examples if example.variant == Variant.ATTACK.value
        ]
        if attack_examples and Variant.SAFE.value not in variants:
            errors.append(f"family {family_id} has attack examples but no safe pair")
        for attack_example in attack_examples:
            if attack_example.attack_class == AttackClass.NONE.value:
                errors.append(f"{attack_example.example_id} attack class cannot be none")
            if attack_example.malicious_span is None:
                errors.append(f"{attack_example.example_id} is missing malicious_span")

    variant_counts = Counter(example.variant for example in materialized)
    attack_class_counts = Counter(example.attack_class for example in materialized)
    return DatasetReport(
        total_examples=len(materialized),
        variant_counts=dict(sorted(variant_counts.items())),
        attack_class_counts=dict(sorted(attack_class_counts.items())),
        family_count=len(families),
        errors=errors,
    )
