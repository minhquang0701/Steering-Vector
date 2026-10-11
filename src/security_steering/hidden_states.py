from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_LAYER_FRACTIONS = (0.25, 0.40, 0.50, 0.60, 0.75, 0.90)


def candidate_layers(
    num_layers: int,
    fractions: Sequence[float] = DEFAULT_LAYER_FRACTIONS,
) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    layers: list[int] = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"layer fraction must be in (0, 1], got {fraction}")
        layer = min(num_layers - 1, max(0, round((num_layers - 1) * fraction)))
        if layer not in layers:
            layers.append(layer)
    return layers


def pool_anchor_states(hidden_states: np.ndarray, anchor_indices: Sequence[int]) -> np.ndarray:
    states = np.asarray(hidden_states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"hidden_states must be shaped [tokens, hidden_dim], got {states.shape}")
    if not anchor_indices:
        raise ValueError("anchor_indices cannot be empty")
    token_count = states.shape[0]
    normalized_indices: list[int] = []
    for index in anchor_indices:
        resolved = index if index >= 0 else token_count + index
        if not 0 <= resolved < token_count:
            raise IndexError(f"anchor index {index} is outside token range {token_count}")
        normalized_indices.append(resolved)
    return states[normalized_indices].mean(axis=0)
