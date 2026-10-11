"""Reproducible baseline, placement/representation sweeps, controls, and bandit training."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from security_steering.bank import METHODS, VectorBank, fit_bank
from security_steering.collection import capture_states, collect_records
from security_steering.dataset import load_jsonl, validate_examples, write_jsonl
from security_steering.evaluation import paired_family_bootstrap, reward, score_response, summarize
from security_steering.hidden_states import candidate_layers
from security_steering.intervention import Intervention
from security_steering.io import digest, file_digest, timestamp, write_json
from security_steering.learning import NeuralAlphaPolicy, RiskProbe, binary_auc, context_features
from security_steering.model import ModelBackend
from security_steering.splits import assert_disjoint, select_split


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    mode: str = "fixed"
    method: str = "mean_difference"
    layers: tuple[int, ...] = ()
    sites: tuple[str, ...] = ("residual",)
    alpha: float = 0.0
    position: str = "decision"
    scale: str = "absolute"
    seed: int = 42
    vector_kind: str = "global"


def interventions_for(spec, bank, example, candidate_layer_ids, *, alpha=None):
    if spec.mode == "baseline":
        return [], []
    value = spec.alpha if alpha is None else alpha
    layer_ids = list(spec.layers)
    rng = np.random.default_rng(int(digest([spec.seed, example.family_id])[:16], 16))
    if spec.mode == "random_layer":
        layer_ids = sorted(rng.choice(candidate_layer_ids, len(layer_ids), replace=False).tolist())
    output = []
    for layer in layer_ids:
        for site in spec.sites:
            vector = bank.get(spec.method, layer, site, spec.vector_kind).copy()
            if spec.mode == "random_direction":
                direction_rng = np.random.default_rng(
                    int(digest([spec.seed, layer, site])[:16], 16)
                )
                direction = direction_rng.normal(size=vector.shape)
                vector = direction / np.linalg.norm(direction) * np.linalg.norm(vector)
            elif spec.mode == "sign_flip":
                vector = -vector
            # Split the intervention budget across sites/layers for matched comparisons.
            output.append(
                Intervention(
                    layer,
                    site,
                    vector,
                    float(value) / len(layer_ids) / len(spec.sites),
                    spec.position,
                    spec.scale,
                )
            )
    return output, layer_ids


class ExperimentRunner:
    def __init__(self, backend, bank, output_dir, layers, *, max_new_tokens=48):
        bank.check_backend(backend)
        self.backend, self.bank = backend, bank
        self.output_dir, self.layers = Path(output_dir), layers
        self.max_new_tokens = max_new_tokens
        self.path = self.output_dir / "responses.jsonl"
        self.cache = {}
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            for line in lines:
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["cache_key"]] = row
        self.feature_cache = {}

    def features(self, example, probe):
        key = (example.example_id, probe.layer)
        if key not in self.feature_cache:
            state = capture_states(self.backend, example, [probe.layer], ["residual"])[
                (probe.layer, "residual")
            ]
            self.feature_cache[key] = context_features(example, probe.predict(state))
        return self.feature_cache[key]

    def run_one(self, example, spec, *, alpha=None):
        effective = spec.alpha if alpha is None else alpha
        key = digest(
            [
                asdict(spec),
                example.to_dict(),
                effective,
                self.bank.metadata["bank_id"],
                self.max_new_tokens,
            ]
        )
        if key in self.cache:
            return self.cache[key]
        interventions, layers = interventions_for(
            spec, self.bank, example, self.layers, alpha=alpha
        )
        generation = self.backend.generate(
            example, interventions, max_new_tokens=self.max_new_tokens
        )
        row = {
            "cache_key": key,
            "run_name": spec.name,
            "spec": asdict(spec),
            "example_id": example.example_id,
            "family_id": example.family_id,
            "variant": example.variant,
            "attack_class": example.attack_class,
            "split": example.metadata["split"],
            "origin": example.metadata.get("origin", "unknown"),
            "selected_layers": layers,
            "alpha": float(effective),
            **asdict(generation),
            **score_response(example, generation.text),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.cache[key] = row
        return row

    def evaluate(self, examples, spec, *, policy=None, probe=None):
        rows = []
        for example in examples:
            alpha = None
            if spec.mode in {"adaptive", "risk_gated"}:
                features = self.features(example, probe)
                alpha = (
                    policy.choose(features)[1]
                    if spec.mode == "adaptive"
                    else float(features[0] * spec.alpha)
                )
            rows.append(self.run_one(example, spec, alpha=alpha))
        return rows


def selection_score(summary):
    def value(key, default):
        result = summary[key]["value"]
        return default if result is None else result

    return (
        value("joint_success", -1),
        -value("attack_success_rate", 1),
        value("benign_task_success", -1),
        -summary["mean_alpha"],
    )


def train_policy(runner, examples, spec, probe, config, output_path):
    if any(e.metadata["split"] != "train" for e in examples):
        raise ValueError("controller training cannot consume validation/test examples")
    eligible = [
        e
        for e in examples
        if e.metadata.get("evaluation", {}).get("security_check") == "controlled"
        and e.metadata.get("evaluation", {}).get("answers")
    ]
    if not eligible:
        raise ValueError("no train examples have complete task and security rewards")
    policy = NeuralAlphaPolicy(
        config["alphas"],
        seed=config.get("seed", 42),
        learning_rate=config.get("learning_rate", 0.02),
        hidden=config.get("hidden", 16),
    )
    epochs = config.get("epochs", 3)
    if epochs < 1:
        raise ValueError("controller epochs must be positive")
    rng = np.random.default_rng(config.get("seed", 42))
    trajectory = []
    training_spec = replace(spec, name="bandit_train", mode="fixed")
    for epoch in range(epochs):
        for index in rng.permutation(len(eligible)):
            example = eligible[index]
            features = runner.features(example, probe)
            action, alpha, probability = policy.choose(features, explore=True)
            row = runner.run_one(example, training_spec, alpha=alpha)
            observed = reward(row, alpha, **config.get("reward", {}))
            policy.update(features, action, observed)
            trajectory.append(
                {
                    "epoch": epoch,
                    "example_id": example.example_id,
                    "split": "train",
                    "features": features.tolist(),
                    "action": action,
                    "alpha": alpha,
                    "propensity": probability,
                    "reward": observed,
                    "response_key": row["cache_key"],
                }
            )
        print(f"controller epoch {epoch + 1}/{epochs}: {policy.updates} updates", flush=True)
    policy.save(
        output_path,
        probe,
        {
            "model_identity": runner.backend.identity,
            "bank_id": runner.bank.metadata["bank_id"],
            "spec": asdict(spec),
            "fit_split": "train",
            "training_families": sorted({e.family_id for e in eligible}),
            "training_config": config,
        },
    )
    write_json(Path(output_path).with_suffix(".trajectory.json"), trajectory)
    return policy


def load_config(path):
    path = Path(path)
    if path.suffix == ".json":
        config = json.loads(path.read_text())
    else:
        import yaml

        config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    allowed = {
        "dataset",
        "output_dir",
        "model",
        "layers",
        "sites",
        "methods",
        "alphas",
        "positions",
        "scale",
        "seed",
        "random_seeds",
        "max_new_tokens",
        "pooling",
        "controller",
        "utility_tolerance",
        "multi_layer",
        "vector_kind",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key in ("dataset", "output_dir", "model"):
        if key not in config:
            raise ValueError(f"missing configuration key {key}")
    return config


def run_pipeline(config, *, resume=False):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    signature = digest({"config": config, "dataset_sha256": file_digest(config["dataset"])})
    if any(output.iterdir()):
        if not resume or not manifest_path.exists():
            raise ValueError("output directory is nonempty; use a new directory or --resume")
        if json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("resume config/dataset differs from the saved run")
    examples = load_jsonl(config["dataset"])
    report = validate_examples(examples)
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    assert_disjoint(examples)
    train, validation, test = [
        select_split(examples, split) for split in ("train", "validation", "test")
    ]
    backend = ModelBackend(**config["model"])
    layers = config.get("layers") or candidate_layers(backend.num_layers)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(
            not isinstance(layer, int) or not 0 <= layer < backend.num_layers for layer in layers
        )
    ):
        raise ValueError("layers must be unique zero-based decoder block indices")
    sites = config.get("sites", ["residual", "mlp", "both"])
    if not sites or any(s not in {"residual", "mlp", "both"} for s in sites):
        raise ValueError("sites must be residual/mlp/both")
    methods = config.get("methods", list(METHODS))
    alphas = config.get("alphas", [0, 1, 5])
    if (
        not alphas
        or any(not np.isfinite(a) or a < 0 for a in alphas)
        or not any(a > 0 for a in alphas)
    ):
        raise ValueError(
            "alpha grid needs finite nonnegative values and at least one positive alpha"
        )
    positions = config.get("positions", ["decision"])
    if not positions or any(p not in {"decision", "all", "prefill", "decode"} for p in positions):
        raise ValueError("invalid positions")
    scale = config.get("scale", "absolute")
    if scale not in {"absolute", "relative"}:
        raise ValueError("scale must be absolute or relative")
    seed = config.get("seed", 42)
    manifest = {
        "signature": signature,
        "config": config,
        "started_at": timestamp(),
        "model_identity": backend.identity,
        "model_name": backend.name,
        "is_smoke": backend.is_smoke,
        "status": "running",
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    import platform
    from importlib.metadata import version

    manifest["environment"] = {
        "python": platform.python_version(),
        **{name: version(name) for name in ("numpy", "torch", "transformers")},
    }
    manifest["source_hashes"] = {
        path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["model_identity"] != backend.identity:
            raise ValueError("resolved model identity changed since previous run")
        if previous.get("source_hashes") != manifest["source_hashes"]:
            raise ValueError(
                "implementation changed since previous run; use a new output directory"
            )
        manifest["started_at"] = previous["started_at"]
    write_json(manifest_path, manifest)
    snapshot = output / "dataset.jsonl"
    if not snapshot.exists():
        write_jsonl(snapshot, examples)
    records_path, bank_path = output / "train_states.npz", output / "vectors.npz"
    if not records_path.exists():
        print(f"collecting {len(train)} training examples at layers {layers}", flush=True)
        collect_records(
            backend,
            examples,
            records_path,
            layers=layers,
            sites=["residual", "mlp"] if set(sites) - {"residual"} else ["residual"],
            pooling=config.get("pooling", "last"),
        )
    bank = (
        VectorBank.load(bank_path)
        if bank_path.exists()
        else fit_bank(records_path, bank_path, methods=methods)
    )
    bank.check_backend(backend)
    runner = ExperimentRunner(
        backend, bank, output, layers, max_new_tokens=config.get("max_new_tokens", 48)
    )
    baseline = ExperimentSpec("baseline", mode="baseline", seed=seed)
    base_val = runner.evaluate(validation, baseline)
    base_summary = summarize(base_val)
    candidates = []
    layer_sets = [(layer,) for layer in layers]
    if config.get("multi_layer", False) and len(layers) > 1:
        layer_sets.append(tuple(layers))
    for method, site, layer_set, alpha, position in itertools.product(
        methods, sites, layer_sets, [a for a in alphas if a > 0], positions
    ):
        name = f"{method}-{site}-l{'_'.join(map(str, layer_set))}-a{alpha}-{position}"
        spec = ExperimentSpec(
            name,
            method=method,
            layers=layer_set,
            sites=("residual", "mlp") if site == "both" else (site,),
            alpha=alpha,
            position=position,
            scale=scale,
            seed=seed,
            vector_kind=config.get("vector_kind", "global"),
        )
        rows = runner.evaluate(validation, spec)
        summary = summarize(rows)
        candidates.append((spec, summary))
        print(f"validation {name}: joint={summary['joint_success']['value']}", flush=True)
    tolerance = config.get("utility_tolerance", 0.05)
    if not 0 <= tolerance <= 1:
        raise ValueError("utility_tolerance must be in [0, 1]")
    base_utility = base_summary["benign_task_success"]["value"]
    if base_utility is None:
        raise ValueError("validation requires benign task rubrics for placement selection")
    feasible = [
        (s, m)
        for s, m in candidates
        if m["benign_task_success"]["value"] is not None
        and m["benign_task_success"]["value"] >= base_utility - tolerance
    ]
    best_spec, best_summary = max(feasible or candidates, key=lambda pair: selection_score(pair[1]))
    selection = {
        "selected_on": "validation",
        "best_fixed": asdict(best_spec),
        "utility_constraint_satisfied": bool(feasible),
        "baseline": base_summary,
        "candidates": [{"spec": asdict(s), "summary": m} for s, m in candidates],
        "best_fixed_summary": best_summary,
        "recommend_fixed_over_baseline": bool(feasible)
        and selection_score(best_summary) > selection_score(base_summary),
    }
    write_json(output / "selection.json", selection)
    probe = RiskProbe.fit(records_path, best_spec.layers[0])
    policy = None
    controller_config = config.get("controller", {"enabled": True})
    if controller_config.get("enabled", True):
        policy_path = output / "controller.npz"
        training_config = {
            "alphas": alphas if 0 in alphas else [0, *alphas],
            "seed": seed,
            **{k: v for k, v in controller_config.items() if k != "enabled"},
        }
        if policy_path.exists():
            policy, probe, policy_metadata = NeuralAlphaPolicy.load(policy_path)
            if (
                policy_metadata["model_identity"] != backend.identity
                or policy_metadata["bank_id"] != bank.metadata["bank_id"]
                or digest(policy_metadata["spec"]) != digest(asdict(best_spec))
                or policy_metadata["training_config"] != training_config
            ):
                raise ValueError(
                    "controller checkpoint does not match this model/bank/training plan"
                )
        else:
            policy = train_policy(runner, train, best_spec, probe, training_config, policy_path)
    # Freeze the full evaluation plan before any test responses are observed.
    specs = [baseline, replace(best_spec, name="selected_fixed")]
    for random_seed in config.get("random_seeds", [seed, seed + 1, seed + 2]):
        specs.append(
            replace(
                best_spec,
                name=f"random_layer_s{random_seed}",
                mode="random_layer",
                seed=random_seed,
            )
        )
        specs.append(
            replace(
                best_spec,
                name=f"random_direction_s{random_seed}",
                mode="random_direction",
                seed=random_seed,
            )
        )
    specs.append(replace(best_spec, name="sign_flip", mode="sign_flip"))
    specs.append(replace(best_spec, name="risk_gated", mode="risk_gated"))
    if policy:
        specs.append(replace(best_spec, name="adaptive_alpha", mode="adaptive"))
    write_json(output / "test_plan.json", [asdict(s) for s in specs])
    summaries, base_test = {}, None
    for spec in specs:
        rows = runner.evaluate(test, spec, policy=policy, probe=probe)
        summaries[spec.name] = summarize(rows)
        summaries[spec.name]["by_attack_class"] = {
            kind: summarize([r for r in rows if r["attack_class"] == kind])
            for kind in sorted({r["attack_class"] for r in rows})
        }
        if spec.mode == "baseline":
            base_test = rows
        else:
            summaries[spec.name]["paired_joint_delta"] = paired_family_bootstrap(base_test, rows)
        print(f"test {spec.name}: {len(rows)} examples", flush=True)
    risks = [float(runner.features(e, probe)[0]) for e in test]
    labels = [int(e.variant == "attack") for e in test]
    manifest.update(
        {
            "status": "complete",
            "finished_at": timestamp(),
            "probe_auc_test": binary_auc(labels, risks),
            "probe_false_positive_rate_at_0_5": float(
                np.mean([risk >= 0.5 for risk, label in zip(risks, labels) if label == 0])
            ),
            "test_family_count": len({e.family_id for e in test}),
        }
    )
    write_json(output / "summary.json", summaries)
    write_json(manifest_path, manifest)
    from security_steering.reporting import build_report

    build_report(output)
    return manifest
