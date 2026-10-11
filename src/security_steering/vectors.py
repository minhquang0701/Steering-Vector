from __future__ import annotations

from typing import Mapping

import numpy as np


ArrayLike = np.ndarray | list[float] | list[list[float]]


def _as_2d(states: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(states, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def normalize_vector(vector: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("vector must be finite and one-dimensional")
    norm = float(np.linalg.norm(array))
    if norm <= eps:
        raise ValueError("cannot normalize a zero vector")
    return array / norm


def mean_difference_vector(
    safe_states: ArrayLike,
    attack_states: ArrayLike,
    *,
    normalize: bool = True,
) -> np.ndarray:
    safe = _as_2d(safe_states, "safe_states")
    attack = _as_2d(attack_states, "attack_states")
    if safe.shape[1] != attack.shape[1]:
        raise ValueError(
            "safe_states and attack_states must have the same hidden dimension, "
            f"got {safe.shape[1]} and {attack.shape[1]}"
        )
    vector = safe.mean(axis=0) - attack.mean(axis=0)
    return normalize_vector(vector) if normalize else vector


def probe_direction_vector(weights: ArrayLike, *, normalize: bool = True) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    vector = -weights_array
    return normalize_vector(vector) if normalize else vector


def add_steering(
    hidden_states: ArrayLike,
    steering_vector: ArrayLike,
    *,
    alpha: float = 1.0,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    states = np.asarray(hidden_states, dtype=np.float64)
    vector = np.asarray(steering_vector, dtype=np.float64)
    if states.shape[-1] != vector.shape[-1]:
        raise ValueError(
            f"hidden dimension mismatch: states={states.shape[-1]} vector={vector.shape[-1]}"
        )
    update = float(alpha) * vector
    if mask is None:
        return states + update
    mask_array = np.asarray(mask, dtype=np.float64)
    return states + np.expand_dims(mask_array, axis=-1) * update


def projection_remove(
    hidden_states: ArrayLike,
    attack_axis: ArrayLike,
    *,
    strength: float = 1.0,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    states = np.asarray(hidden_states, dtype=np.float64)
    axis = normalize_vector(attack_axis)
    if states.shape[-1] != axis.shape[-1]:
        raise ValueError(
            f"hidden dimension mismatch: states={states.shape[-1]} axis={axis.shape[-1]}"
        )
    projection = np.tensordot(states, axis, axes=([-1], [0]))
    positive_projection = np.maximum(projection, 0.0)
    removal = float(strength) * np.expand_dims(positive_projection, axis=-1) * axis
    if mask is not None:
        mask_array = np.asarray(mask, dtype=np.float64)
        removal = np.expand_dims(mask_array, axis=-1) * removal
    return states - removal


def vector_bank_from_states(
    states_by_class: Mapping[str, Mapping[str, ArrayLike]],
    *,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for attack_class, class_states in states_by_class.items():
        if "safe" not in class_states or "attack" not in class_states:
            raise ValueError(f"{attack_class} must include 'safe' and 'attack' arrays")
        vectors[attack_class] = mean_difference_vector(
            class_states["safe"],
            class_states["attack"],
            normalize=normalize,
        )
    return vectors
