from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from security_steering.dataset import load_jsonl, validate_examples, write_jsonl
from security_steering.seed import build_seed_examples


def _cmd_build_seed_data(args: argparse.Namespace) -> int:
    examples = build_seed_examples()
    write_jsonl(args.output, examples)
    print(f"wrote {len(examples)} seed examples to {args.output}")
    return 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    examples = load_jsonl(args.path)
    report = validate_examples(examples)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _cmd_compute_vectors(args: argparse.Namespace) -> int:
    from security_steering.bank import fit_bank

    bank = fit_bank(
        args.records,
        args.output,
        methods=args.methods,
        normalize=not args.no_normalize,
        layer=args.layer,
    )
    print(f"wrote {len(bank.entries)} layer/site-specific vectors to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="security-steering",
        description="Source-aware security steering: data, vectors, experiments and adaptive alpha.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_seed = subparsers.add_parser("build-seed-data", help="write the seed JSONL dataset")
    build_seed.add_argument("output", type=Path)
    build_seed.set_defaults(func=_cmd_build_seed_data)

    validate = subparsers.add_parser("validate-data", help="validate a JSONL dataset")
    validate.add_argument("path", type=Path)
    validate.set_defaults(func=_cmd_validate_data)

    vectors = subparsers.add_parser(
        "compute-vectors",
        help="compute class steering vectors from collected hidden-state records",
    )
    vectors.add_argument(
        "records", type=Path, help="NPZ with states, attack_class, variant, family_id"
    )
    vectors.add_argument("output", type=Path, help="NPZ vector bank output path")
    vectors.add_argument("--layer", type=int, default=None, help="optional layer id to filter")
    vectors.add_argument("--no-normalize", action="store_true", help="keep raw mean differences")
    from security_steering.bank import METHODS

    vectors.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    vectors.set_defaults(func=_cmd_compute_vectors)

    from security_steering.commands import add_research_commands

    add_research_commands(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
