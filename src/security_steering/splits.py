"""Family/payload grouped splits; metadata never enters the model prompt."""

from __future__ import annotations

import copy
import re
from collections import defaultdict

import numpy as np

from security_steering.dataset import group_by_family, validate_examples
from security_steering.io import digest
from security_steering.schemas import SteeringExample

SPLITS = ("train", "validation", "test")


def leakage_keys(example: SteeringExample) -> set[str]:
    keys = {"family:" + example.family_id}
    if example.metadata.get("split_group"):
        keys.add("group:" + str(example.metadata["split_group"]))
    if example.malicious_span is not None:
        span = example.malicious_span
        segment = next(s for s in example.context_segments if s.segment_id == span.segment_id)
        payload = re.sub(r"\s+", " ", segment.text[span.start_char : span.end_char]).strip().lower()
        keys.add("payload:" + digest(payload))
    keys.add(
        "content:" + digest([example.user_task, [s.to_dict() for s in example.context_segments]])
    )
    return keys


def assert_disjoint(examples: list[SteeringExample]) -> None:
    seen: dict[str, str] = {}
    for example in examples:
        split = example.metadata.get("split")
        if split not in SPLITS:
            raise ValueError(f"{example.example_id}: explicit train/validation/test split required")
        for key in leakage_keys(example):
            if key in seen and seen[key] != split:
                raise ValueError(f"split leakage: {key} is in {seen[key]} and {split}")
            seen[key] = split


def split_examples(
    examples: list[SteeringExample],
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[SteeringExample]:
    report = validate_examples(examples)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("three positive split fractions must sum to one")
    families = group_by_family(examples)
    parent = {f: f for f in families}

    def root(f: str) -> str:
        while f != parent[f]:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    seen: dict[str, str] = {}
    for example in examples:
        for key in leakage_keys(example):
            if key in seen:
                left, right = sorted((root(example.family_id), root(seen[key])))
                parent[right] = left
            seen[key] = example.family_id
    groups: dict[str, list[SteeringExample]] = defaultdict(list)
    for example in examples:
        groups[root(example.family_id)].append(example)
    assignments: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        locked = {e.metadata.get("split") for e in members} - {None}
        if len(locked) > 1 or not locked.issubset(SPLITS):
            raise ValueError(f"conflicting/invalid official splits in group {key}")
        if locked:
            assignments[key] = next(iter(locked))
        else:
            classes = sorted({e.attack_class for e in members} - {"none"})
            strata[",".join(classes) or "benign"].append(key)
    rng = np.random.default_rng(seed)
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
        n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
        while n_val + n_test >= n:
            if n_val >= n_test:
                n_val -= 1
            else:
                n_test -= 1
        n_train = n - n_val - n_test
        for i, key in enumerate(keys):
            assignments[key] = (
                "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
            )
    output = copy.deepcopy(examples)
    for example in output:
        example.metadata["split"] = assignments[root(example.family_id)]
    assert_disjoint(output)
    if {e.metadata["split"] for e in output} != set(SPLITS):
        raise ValueError("need enough independent families to populate all three splits")
    return output


def select_split(examples: list[SteeringExample], split: str) -> list[SteeringExample]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    assert_disjoint(examples)
    result = [e for e in examples if e.metadata["split"] == split]
    if not result:
        raise ValueError(f"empty {split} split")
    return result
