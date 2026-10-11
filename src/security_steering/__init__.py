"""Source-aware activation steering tools for prompt-injection research."""

from security_steering.controller import (
    aggregate_risk,
    compose_steering_vector,
    risk_gated_alpha,
)
from security_steering.dataset import (
    DatasetReport,
    group_by_family,
    load_jsonl,
    validate_examples,
    write_jsonl,
)
from security_steering.hidden_states import candidate_layers, pool_anchor_states
from security_steering.schemas import (
    AttackClass,
    ContextSegment,
    ExpectedMemoryAction,
    ExpectedToolAction,
    SourceType,
    Span,
    SteeringExample,
    TrustLevel,
    Variant,
)
from security_steering.seed import build_seed_examples
from security_steering.vectors import (
    add_steering,
    mean_difference_vector,
    normalize_vector,
    probe_direction_vector,
    projection_remove,
)

__all__ = [
    "AttackClass",
    "ContextSegment",
    "DatasetReport",
    "ExpectedMemoryAction",
    "ExpectedToolAction",
    "SourceType",
    "Span",
    "SteeringExample",
    "TrustLevel",
    "Variant",
    "add_steering",
    "aggregate_risk",
    "build_seed_examples",
    "candidate_layers",
    "compose_steering_vector",
    "group_by_family",
    "load_jsonl",
    "mean_difference_vector",
    "normalize_vector",
    "pool_anchor_states",
    "probe_direction_vector",
    "projection_remove",
    "risk_gated_alpha",
    "validate_examples",
    "write_jsonl",
]
