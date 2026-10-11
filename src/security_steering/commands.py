from __future__ import annotations

import json
from pathlib import Path

from security_steering.dataset import load_jsonl, validate_examples, write_jsonl
from security_steering.splits import split_examples


def _build_data(args):
    from security_steering.synthetic import build_controlled_data

    examples = build_controlled_data(args.families_per_class, args.seed)
    if args.classes:
        families = {e.family_id for e in examples if e.attack_class in args.classes}
        examples = [e for e in examples if e.family_id in families]
    from security_steering.benchmarks import import_bipia, import_injecagent

    if args.bipia_dir:
        for split in ("train", "test"):
            examples.extend(
                import_bipia(
                    args.bipia_dir / "benchmark/email" / f"{split}.jsonl",
                    args.bipia_dir / "benchmark" / f"text_attack_{split}.json",
                    limit=args.benchmark_limit,
                    upstream_split=split,
                )
            )
    if args.injecagent_dir:
        for kind in ("dh", "ds"):
            examples.extend(
                import_injecagent(
                    args.injecagent_dir / "data" / f"test_cases_{kind}_base.json",
                    limit=args.benchmark_limit,
                )
            )
    for path in args.llm_data:
        additions = load_jsonl(path)
        if not args.include_unreviewed and any(
            e.metadata.get("review_status") != "approved" for e in additions
        ):
            raise ValueError(
                "LLM candidates require review_status=approved; "
                "use --include-unreviewed only for labelled exploratory runs"
            )
        examples.extend(additions)
    examples = split_examples(examples, seed=args.seed)
    write_jsonl(args.output, examples)
    print(json.dumps(validate_examples(examples).__dict__, indent=2))
    return 0


def _fetch(args):
    from security_steering.benchmarks import download_benchmarks

    result = download_benchmarks(args.output, args.names, args.revision)
    print(
        f"downloaded {len(result['sources'])} resources; manifest: {args.output / 'manifest.json'}"
    )
    return 0


def _import(args):
    from security_steering.benchmarks import import_bipia, import_injecagent

    if args.benchmark == "bipia":
        if args.attacks is None:
            raise ValueError("BIPIA import requires --attacks")
        examples = import_bipia(
            args.input, args.attacks, limit=args.limit, upstream_split=args.upstream_split
        )
    else:
        examples = import_injecagent(args.input, limit=args.limit)
    write_jsonl(args.output, examples)
    print(f"wrote {len(examples)} static replay examples (not native benchmark evaluation)")
    return 0


def _split(args):
    examples = [e for path in args.inputs for e in load_jsonl(path)]
    write_jsonl(args.output, split_examples(examples, seed=args.seed))
    return 0


def _model(args):
    from security_steering.model import ModelBackend

    return ModelBackend(
        args.model,
        device=args.device,
        revision=args.revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        threads=args.threads,
    )


def _collect(args):
    from security_steering.collection import collect_records
    from security_steering.hidden_states import candidate_layers

    backend = _model(args)
    metadata = collect_records(
        backend,
        load_jsonl(args.dataset),
        args.output,
        layers=args.layers or candidate_layers(backend.num_layers),
        sites=args.sites,
        split=args.split,
        pooling=args.pooling,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _generate(args):
    from security_steering.augmentation import generate_candidates

    examples = generate_candidates(_model(args), args.output, count=args.count, seed=args.seed)
    print(f"wrote {len(examples)} candidate examples; review before dataset inclusion")
    return 0 if examples else 1


def _run(args):
    from security_steering.experiments import load_config, run_pipeline

    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    manifest = run_pipeline(config, resume=args.resume)
    print(f"completed {'smoke test' if manifest['is_smoke'] else 'experiment'}")
    return 0


def _report(args):
    from security_steering.reporting import build_report

    build_report(args.run_dir)
    print(args.run_dir / "report.md")
    return 0


def _model_arguments(parser):
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/model-cache"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)


def add_research_commands(subparsers):
    from security_steering.synthetic import CLASSES

    build = subparsers.add_parser(
        "build-data", help="build paired and grouped hybrid research data"
    )
    build.add_argument("output", type=Path)
    build.add_argument("--families-per-class", type=int, default=6)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--classes", nargs="+", choices=CLASSES)
    build.add_argument("--bipia-dir", type=Path)
    build.add_argument("--injecagent-dir", type=Path)
    build.add_argument("--benchmark-limit", type=int, default=30)
    build.add_argument("--llm-data", nargs="*", type=Path, default=[])
    build.add_argument("--include-unreviewed", action="store_true")
    build.set_defaults(func=_build_data)
    fetch = subparsers.add_parser(
        "fetch-benchmarks", help="download versioned public benchmark data"
    )
    fetch.add_argument("output", type=Path)
    fetch.add_argument(
        "--names", nargs="+", choices=["bipia", "injecagent"], default=["bipia", "injecagent"]
    )
    fetch.add_argument("--revision", default="main")
    fetch.set_defaults(func=_fetch)
    ingest = subparsers.add_parser("import-benchmark", help="normalize a local benchmark snapshot")
    ingest.add_argument("benchmark", choices=["bipia", "injecagent"])
    ingest.add_argument("input", type=Path)
    ingest.add_argument("output", type=Path)
    ingest.add_argument("--attacks", type=Path)
    ingest.add_argument("--limit", type=int)
    ingest.add_argument("--upstream-split", choices=["train", "test"], default="train")
    ingest.set_defaults(func=_import)
    split = subparsers.add_parser("split-data", help="merge and split families without leakage")
    split.add_argument("output", type=Path)
    split.add_argument("inputs", nargs="+", type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)
    collect = subparsers.add_parser("collect-states", help="capture layer/site activations")
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--layers", nargs="+", type=int)
    collect.add_argument(
        "--sites", nargs="+", choices=["residual", "mlp"], default=["residual", "mlp"]
    )
    collect.add_argument("--pooling", choices=["last", "mean"], default="last")
    collect.add_argument("--split", choices=["train", "validation", "test"], default="train")
    _model_arguments(collect)
    collect.set_defaults(func=_collect)
    generate = subparsers.add_parser(
        "generate-data", help="generate LLM candidate pairs for review"
    )
    generate.add_argument("output", type=Path)
    generate.add_argument("--count", type=int, default=7)
    _model_arguments(generate)
    generate.set_defaults(func=_generate)
    run = subparsers.add_parser("run", help="run all computational milestones from a config")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--output-dir", type=Path, help="override the configured artifact directory")
    run.set_defaults(func=_run)
    report = subparsers.add_parser(
        "report", help="regenerate Markdown/CSV/LaTeX/plots from results"
    )
    report.add_argument("run_dir", type=Path)
    report.set_defaults(func=_report)
