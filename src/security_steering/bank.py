"""Per-layer, per-site prototypes and defensive directions, fitted on train only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from security_steering.io import digest
from security_steering.vectors import normalize_vector

METHODS = ("mean_difference", "paired_mean", "paired_pca", "ridge_probe")


def representative(safe, attack, paired_safe, paired_attack, method, *, ridge=1.0):
    if method == "mean_difference":
        return safe.mean(axis=0) - attack.mean(axis=0)
    differences = paired_safe - paired_attack
    mean = differences.mean(axis=0)
    if method == "paired_mean":
        return mean
    if method == "paired_pca":
        # Uncentered paired differences; centered PCA would discard the desired mean shift.
        _, _, right = np.linalg.svd(differences, full_matrices=False)
        axis = right[0]
        return axis if np.dot(axis, mean) >= 0 else -axis
    if method == "ridge_probe":
        x = np.concatenate([paired_safe, paired_attack]).astype(np.float64)
        x -= x.mean(axis=0)
        labels = np.concatenate([np.ones(len(paired_safe)), -np.ones(len(paired_attack))])
        # Dual ridge avoids allocating a hidden_dim x hidden_dim matrix.
        return x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), labels)
    raise ValueError(f"unknown representative method {method!r}")


class VectorBank:
    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict):
        self.arrays, self.metadata = arrays, metadata
        self.entries = metadata["entries"]

    def get(self, method: str, layer: int, site: str, kind: str = "global") -> np.ndarray:
        key = f"{method}__{site}__l{layer}__{kind}"
        if key not in self.arrays:
            raise ValueError(f"missing vector {key}; collect and fit this layer/site/class first")
        return self.arrays[key]

    def check_backend(self, backend):
        if self.metadata["model_identity"] != backend.identity:
            raise ValueError("vector bank was extracted from a different model/revision/dtype/seed")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays, metadata=np.asarray(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            return cls({k: data[k].copy() for k in data.files if k != "metadata"}, metadata)


def fit_bank(records_path, output_path, *, methods=METHODS, normalize=True, layer=None):
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError(f"methods must be from {METHODS}")
    with np.load(records_path, allow_pickle=False) as archive:
        records = {k: archive[k].copy() for k in archive.files}
    required = {
        "states",
        "layer",
        "site",
        "split",
        "family_id",
        "example_id",
        "variant",
        "attack_class",
        "metadata",
    }
    if not required.issubset(records):
        raise ValueError("legacy/unlabelled records: recollect with explicit sites and train split")
    if set(records["split"].tolist()) != {"train"}:
        raise ValueError("vector fitting only accepts train records")
    states = records["states"].astype(np.float64)
    if states.ndim != 2 or not len(states) or not np.isfinite(states).all():
        raise ValueError("states must be a nonempty finite [records, hidden_dim] matrix")
    for key in required - {"metadata", "states"}:
        if records[key].shape != (len(states),):
            raise ValueError(f"record shape mismatch for {key}")
    identities = list(zip(records["example_id"], records["layer"], records["site"]))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate example/layer/site records")
    arrays, entries, skipped = {}, [], []
    variants, families, classes = [records[k] for k in ("variant", "family_id", "attack_class")]
    for layer_id in sorted(set(records["layer"].tolist())):
        if layer is not None and layer != layer_id:
            continue
        for site in sorted(set(records["site"].tolist())):
            group = (records["layer"] == layer_id) & (records["site"] == site)
            for kind in ["global", *sorted(set(classes[group].tolist()) - {"none"})]:
                attack_mask = group & (variants == "attack")
                if kind != "global":
                    attack_mask &= classes == kind
                if not attack_mask.any():
                    continue
                paired_safe, paired_attack = [], []
                safe_mask = np.zeros(len(states), dtype=bool)
                for family in sorted(set(families[attack_mask].tolist())):
                    safe_family = group & (families == family) & (variants == "safe")
                    if not safe_family.any():
                        raise ValueError(f"missing safe pair for {family}/{layer_id}/{site}")
                    safe_mask |= safe_family
                    paired_safe.append(states[safe_family].mean(axis=0))
                    paired_attack.append(states[attack_mask & (families == family)].mean(axis=0))
                safe, attack = states[safe_mask], states[attack_mask]
                prefix = f"{site}__l{layer_id}__{kind}"
                arrays[f"safe_mean__{prefix}"] = safe.mean(axis=0)
                arrays[f"attack_mean__{prefix}"] = attack.mean(axis=0)
                for method in methods:
                    vector = representative(
                        safe, attack, np.asarray(paired_safe), np.asarray(paired_attack), method
                    )
                    norm = float(np.linalg.norm(vector))
                    key = f"{method}__{prefix}"
                    if norm < 1e-12:
                        skipped.append({"key": key, "reason": "zero separation"})
                        continue
                    arrays[key] = normalize_vector(vector) if normalize else vector
                    entries.append(
                        {
                            "key": key,
                            "method": method,
                            "site": site,
                            "layer": layer_id,
                            "kind": kind,
                            "raw_norm": norm,
                            "families": len(paired_safe),
                            "safe_count": len(safe),
                            "attack_count": len(attack),
                        }
                    )
    if not entries:
        raise ValueError("no nonzero vectors could be fitted")
    metadata = json.loads(str(records["metadata"].item()))
    metadata.update(
        {
            "entries": entries,
            "normalized": normalize,
            "skipped": skipped,
            "training_families": sorted(set(families.tolist())),
            "fit_split": "train",
            "methods": list(methods),
        }
    )
    metadata["bank_id"] = digest(
        {"metadata": metadata, "arrays": {k: v.tolist() for k, v in arrays.items()}}
    )
    bank = VectorBank(arrays, metadata)
    bank.save(output_path)
    return bank
