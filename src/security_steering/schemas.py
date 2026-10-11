from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ValueEnum(str, Enum):
    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


class Variant(ValueEnum):
    SAFE = "safe"
    ATTACK = "attack"
    HARD_BENIGN = "hard_benign"


class AttackClass(ValueEnum):
    NONE = "none"
    OVERRIDE = "override"
    REVEAL = "reveal"
    EXFILTRATION = "exfiltration"
    TOOL_MISUSE = "tool_misuse"
    MEMORY_POISONING = "memory_poisoning"
    SOURCE_CONFUSION = "source_confusion"
    OBFUSCATION = "obfuscation"


class SourceType(ValueEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    TOOL_OUTPUT = "tool_output"
    RETRIEVED_DOCUMENT = "retrieved_document"
    WEBPAGE = "webpage"
    MEMORY = "memory"
    AGENT_MESSAGE = "agent_message"


class TrustLevel(ValueEnum):
    TRUSTED_INSTRUCTION = "trusted_instruction"
    USER_INTENT = "user_intent"
    UNTRUSTED_DATA = "untrusted_data"
    PRIVATE_DATA = "private_data"


class AllowedRole(ValueEnum):
    INSTRUCTION = "instruction"
    EVIDENCE = "evidence"
    DATA = "data"
    MEMORY_FACT = "memory_fact"


class ExpectedToolAction(ValueEnum):
    ALLOWED_TOOL = "allowed_tool"
    BLOCKED_TOOL = "blocked_tool"
    ASK_CONFIRMATION = "ask_confirmation"
    NO_TOOL = "no_tool"


class ExpectedMemoryAction(ValueEnum):
    WRITE = "write"
    DO_NOT_WRITE = "do_not_write"
    QUARANTINE = "quarantine"


SOURCE_RISK_PRIORS: dict[str, float] = {
    SourceType.SYSTEM.value: 0.00,
    SourceType.DEVELOPER.value: 0.05,
    SourceType.USER.value: 0.10,
    SourceType.MEMORY.value: 0.40,
    SourceType.RETRIEVED_DOCUMENT.value: 0.55,
    SourceType.WEBPAGE.value: 0.65,
    SourceType.TOOL_OUTPUT.value: 0.60,
    SourceType.AGENT_MESSAGE.value: 0.50,
}


def _coerce_enum(enum_cls: type[ValueEnum], value: str | ValueEnum, field_name: str) -> str:
    if isinstance(value, enum_cls):
        return value.value
    if isinstance(value, str) and value in enum_cls.values():
        return value
    allowed = ", ".join(sorted(enum_cls.values()))
    raise ValueError(f"{field_name} must be one of: {allowed}")


def _validate_probability(value: float, field_name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")
    return numeric


@dataclass
class Span:
    segment_id: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("span.segment_id cannot be empty")
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError("span must satisfy 0 <= start_char <= end_char")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Span":
        return cls(
            segment_id=str(data["segment_id"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
        )


@dataclass
class ContextSegment:
    segment_id: str
    text: str
    source_type: str
    trust_level: str
    allowed_role: str
    can_trigger_tool: bool = False
    can_update_memory: bool = False
    risk_prior: float | None = None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("context segment id cannot be empty")
        if not self.text:
            raise ValueError(f"context segment {self.segment_id!r} cannot be empty")
        self.source_type = _coerce_enum(SourceType, self.source_type, "source_type")
        self.trust_level = _coerce_enum(TrustLevel, self.trust_level, "trust_level")
        self.allowed_role = _coerce_enum(AllowedRole, self.allowed_role, "allowed_role")
        if self.risk_prior is None:
            self.risk_prior = SOURCE_RISK_PRIORS[self.source_type]
        self.risk_prior = _validate_probability(self.risk_prior, "risk_prior")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextSegment":
        return cls(
            segment_id=str(data["segment_id"]),
            text=str(data["text"]),
            source_type=str(data["source_type"]),
            trust_level=str(data["trust_level"]),
            allowed_role=str(data["allowed_role"]),
            can_trigger_tool=bool(data.get("can_trigger_tool", False)),
            can_update_memory=bool(data.get("can_update_memory", False)),
            risk_prior=data.get("risk_prior"),
        )


@dataclass
class SteeringExample:
    example_id: str
    family_id: str
    variant: str
    user_task: str
    context_segments: list[ContextSegment]
    expected_safe_behavior: str
    expected_tool_action: str
    expected_memory_action: str
    attack_class: str = AttackClass.NONE.value
    malicious_span: Span | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id cannot be empty")
        if not self.family_id:
            raise ValueError("family_id cannot be empty")
        if not self.user_task:
            raise ValueError("user_task cannot be empty")
        self.variant = _coerce_enum(Variant, self.variant, "variant")
        self.attack_class = _coerce_enum(AttackClass, self.attack_class, "attack_class")
        self.expected_tool_action = _coerce_enum(
            ExpectedToolAction,
            self.expected_tool_action,
            "expected_tool_action",
        )
        self.expected_memory_action = _coerce_enum(
            ExpectedMemoryAction,
            self.expected_memory_action,
            "expected_memory_action",
        )
        self.context_segments = [
            segment if isinstance(segment, ContextSegment) else ContextSegment.from_dict(segment)
            for segment in self.context_segments
        ]
        segment_ids = [s.segment_id for s in self.context_segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("context segment ids must be unique within an example")
        if isinstance(self.malicious_span, Mapping):
            self.malicious_span = Span.from_dict(self.malicious_span)
        if self.variant == Variant.ATTACK.value and self.attack_class == AttackClass.NONE.value:
            raise ValueError("attack examples must use a non-none attack_class")
        if self.variant != Variant.ATTACK.value and self.attack_class != AttackClass.NONE.value:
            raise ValueError("non-attack examples must use attack_class='none'")
        if self.variant == Variant.ATTACK.value and self.malicious_span is None:
            raise ValueError("attack examples must identify malicious_span")
        if self.malicious_span is not None:
            self._validate_span()

    def _validate_span(self) -> None:
        segment_by_id = {segment.segment_id: segment for segment in self.context_segments}
        assert self.malicious_span is not None
        if self.malicious_span.start_char == self.malicious_span.end_char:
            raise ValueError("malicious_span must be nonempty")
        segment = segment_by_id.get(self.malicious_span.segment_id)
        if segment is None:
            raise ValueError(
                f"malicious_span references missing segment {self.malicious_span.segment_id!r}"
            )
        if self.malicious_span.end_char > len(segment.text):
            raise ValueError("malicious_span extends beyond its segment text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "variant": self.variant,
            "user_task": self.user_task,
            "context_segments": [segment.to_dict() for segment in self.context_segments],
            "expected_safe_behavior": self.expected_safe_behavior,
            "expected_tool_action": self.expected_tool_action,
            "expected_memory_action": self.expected_memory_action,
            "attack_class": self.attack_class,
            "malicious_span": None
            if self.malicious_span is None
            else self.malicious_span.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SteeringExample":
        return cls(
            example_id=str(data["example_id"]),
            family_id=str(data["family_id"]),
            variant=str(data["variant"]),
            user_task=str(data["user_task"]),
            context_segments=[
                ContextSegment.from_dict(segment) for segment in data.get("context_segments", [])
            ],
            expected_safe_behavior=str(data["expected_safe_behavior"]),
            expected_tool_action=str(data["expected_tool_action"]),
            expected_memory_action=str(data["expected_memory_action"]),
            attack_class=str(data.get("attack_class", AttackClass.NONE.value)),
            malicious_span=None
            if data.get("malicious_span") is None
            else Span.from_dict(data["malicious_span"]),
            metadata=dict(data.get("metadata", {})),
        )
