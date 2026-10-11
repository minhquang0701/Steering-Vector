from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from security_steering.dataset import load_jsonl
from security_steering.hidden_states import candidate_layers


def collect_hidden_state_records(
    *,
    dataset_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    max_examples: int | None,
) -> None:
    from security_steering.collection import collect_records
    from security_steering.model import ModelBackend

    examples = load_jsonl(dataset_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        # Keep entire families even if this slightly exceeds the requested limit.
        families = {e.family_id for e in examples[:max_examples]}
        examples = [e for e in examples if e.family_id in families]
    backend = ModelBackend(model_name, device=device, cache_dir="artifacts/model-cache")
    collect_records(backend, examples, output_path, layers=candidate_layers(backend.num_layers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect paired train states at residual and MLP sites."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collect_hidden_state_records(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        max_examples=args.max_examples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
