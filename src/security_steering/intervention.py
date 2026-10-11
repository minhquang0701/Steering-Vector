"""Scoped PyTorch hooks at the exact same sites used for vector extraction."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

import numpy as np


def transformer_blocks(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        current = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None and len(current):
            return current
    raise ValueError("unsupported decoder: expected Llama/Qwen/GPT2/GPT-NeoX/OPT blocks")


def site_module(model, layer: int, site: str):
    blocks = transformer_blocks(model)
    if not 0 <= layer < len(blocks):
        raise ValueError(f"layer {layer} outside [0, {len(blocks)})")
    block = blocks[layer]
    if site == "residual":
        return block
    if site == "mlp" and hasattr(block, "mlp"):
        return block.mlp
    if site == "mlp" and hasattr(block, "fc2"):
        return block.fc2
    raise ValueError(f"unsupported intervention site {site!r} on {type(block).__name__}")


def hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


@dataclass(frozen=True)
class Intervention:
    layer: int
    site: str
    vector: np.ndarray
    alpha: float
    position: str = "decision"
    scale: str = "absolute"


class SteeringHooks(AbstractContextManager):
    """decision=last token on every forward, prefill/decode are explicit phases.

    Each example is unpadded and generated with one beam. No in-place updates or
    weight changes. Hooks are removed even if generation or registration fails.
    """

    def __init__(self, model, interventions: list[Intervention]):
        self.model = model
        self.interventions = interventions
        self.handles = []
        self.calls: dict[str, int] = {}

    def __enter__(self):
        import torch

        keys = [(i.layer, i.site) for i in self.interventions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate layer/site interventions")
        try:
            for spec in self.interventions:
                if spec.position not in {"decision", "all", "prefill", "decode"}:
                    raise ValueError(f"unknown token position {spec.position}")
                if spec.scale not in {"absolute", "relative"}:
                    raise ValueError(f"unknown scale {spec.scale}")
                vector = np.asarray(spec.vector)
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    raise ValueError("vector must be finite and one-dimensional")
                if not np.isfinite(spec.alpha):
                    raise ValueError("alpha must be finite")
                module = site_module(self.model, spec.layer, spec.site)
                key = f"{spec.site}:{spec.layer}"
                self.calls[key] = 0

                def hook(_module, _args, output, spec=spec, vector=vector, key=key):
                    hidden = hidden_tensor(output)
                    call = self.calls[key]
                    self.calls[key] += 1
                    if hidden.ndim != 3 or hidden.shape[-1] != len(vector):
                        raise ValueError("activation/vector shape mismatch")
                    if (
                        spec.alpha == 0
                        or (spec.position == "prefill" and call > 0)
                        or (spec.position == "decode" and call == 0)
                    ):
                        return output
                    direction = torch.as_tensor(vector, device=hidden.device, dtype=hidden.dtype)
                    target = hidden if spec.position == "all" else hidden[:, -1:, :]
                    scale = 1.0
                    if spec.scale == "relative":
                        scale = target.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                        direction = direction / direction.norm().clamp_min(1e-12)
                    updated = target + spec.alpha * scale * direction
                    if spec.position == "all":
                        result = updated
                    else:
                        result = torch.cat([hidden[:, :-1, :], updated], dim=1)
                    return (result, *output[1:]) if isinstance(output, tuple) else result

                self.handles.append(module.register_forward_hook(hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
