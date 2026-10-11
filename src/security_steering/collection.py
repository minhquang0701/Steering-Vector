from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from security_steering.intervention import hidden_tensor, site_module
from security_steering.io import digest
from security_steering.prompting import PROMPT_VERSION
from security_steering.splits import select_split


def capture_states(backend, example, layers, sites, *, pooling="last"):
    import torch

    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be last or mean")
    captured = {}
    handles = []
    try:
        for layer in layers:
            for site in sites:

                def hook(_module, _args, output, key=(layer, site)):
                    states = hidden_tensor(output)[0]
                    pooled = states[-1] if pooling == "last" else states.mean(dim=0)
                    captured[key] = pooled.detach().float().cpu().numpy().copy()

                handles.append(site_module(backend.model, layer, site).register_forward_hook(hook))
        with torch.inference_mode():
            backend.model(**backend.encode(example), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def collect_records(
    backend, examples, output, *, layers, sites=("residual", "mlp"), split="train", pooling="last"
):
    if len(set(layers)) != len(layers) or len(set(sites)) != len(sites) or not layers or not sites:
        raise ValueError("nonempty unique layers/sites required")
    selected = select_split(examples, split)
    records = {
        k: []
        for k in (
            "states",
            "example_id",
            "family_id",
            "variant",
            "attack_class",
            "layer",
            "site",
            "split",
        )
    }
    for example in selected:
        captured = capture_states(backend, example, layers, sites, pooling=pooling)
        for (layer, site), state in captured.items():
            records["states"].append(state)
            for key in ("example_id", "family_id", "variant", "attack_class"):
                records[key].append(getattr(example, key))
            records["layer"].append(layer)
            records["site"].append(site)
            records["split"].append(split)
    metadata = {
        "format_version": 2,
        "model_name": backend.name,
        "model_identity": backend.identity,
        "hidden_dim": backend.hidden_dim,
        "prompt_version": PROMPT_VERSION,
        "pooling": pooling,
        "representation": "prompt_condition_not_observed_attack_following",
        "dataset_hash": digest([e.to_dict() for e in examples]),
        "is_smoke": backend.is_smoke,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: np.asarray(v) for k, v in records.items()},
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata
