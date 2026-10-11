from __future__ import annotations

from typing import Mapping

import numpy as np

from security_steering.vectors import normalize_vector


def risk_gated_alpha(
    risk: float | np.ndarray,
    *,
    alpha_max: float = 1.0,
    threshold: float = 0.50,
    sharpness: float = 12.0,
) -> float | np.ndarray:
    risk_array = np.asarray(risk, dtype=np.float64)
    alpha = float(alpha_max) / (1.0 + np.exp(-float(sharpness) * (risk_array - threshold)))
    if alpha.ndim == 0:
        return float(alpha)
    return alpha


def aggregate_risk(
    *,
    probe_risk: float = 0.0,
    text_risk: float = 0.0,
    source_risk: float = 0.0,
    action_risk: float = 0.0,
) -> float:
    return max(float(probe_risk), float(text_risk), float(source_risk), float(action_risk))


def compose_steering_vector(
    vector_bank: Mapping[str, np.ndarray],
    class_risks: Mapping[str, float],
    *,
    alpha_max: float = 1.0,
    threshold: float = 0.50,
    sharpness: float = 12.0,
    normalize: bool = False,
) -> np.ndarray:
    if not vector_bank:
        raise ValueError("vector_bank cannot be empty")

    hidden_dim = next(iter(vector_bank.values())).shape[-1]
    composed = np.zeros(hidden_dim, dtype=np.float64)
    for attack_class, risk in class_risks.items():
        vector = vector_bank.get(attack_class)
        if vector is None:
            continue
        if vector.shape[-1] != hidden_dim:
            raise ValueError("all steering vectors must share the same hidden dimension")
        alpha = risk_gated_alpha(
            risk,
            alpha_max=alpha_max,
            threshold=threshold,
            sharpness=sharpness,
        )
        composed += float(alpha) * vector

    if normalize and np.linalg.norm(composed) > 0:
        return normalize_vector(composed)
    return composed
