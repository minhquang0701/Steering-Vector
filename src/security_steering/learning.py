"""Train-only risk probe and a small neural contextual-bandit alpha policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from security_steering.io import digest


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


@dataclass
class RiskProbe:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    layer: int

    @classmethod
    def fit(cls, records_path, layer, *, steps=200, learning_rate=0.03):
        with np.load(records_path, allow_pickle=False) as data:
            if set(data["split"].tolist()) != {"train"}:
                raise ValueError("risk probe must be fitted on train only")
            mask = (data["layer"] == layer) & (data["site"] == "residual")
            x = data["states"][mask].astype(np.float64)
            y = (data["variant"][mask] == "attack").astype(float)
        if len(np.unique(y)) < 2:
            raise ValueError("risk probe needs both benign and attack training examples")
        center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
        x = np.clip((x - center) / scale, -10, 10) / np.sqrt(x.shape[1])
        weights, bias = np.zeros(x.shape[1]), 0.0
        for _ in range(steps):
            error = sigmoid(x @ weights + bias) - y
            weights -= learning_rate * (x.T @ error / len(y) + 1e-3 * weights)
            bias -= learning_rate * error.mean()
        return cls(center, scale, weights, float(bias), layer)

    def predict(self, state):
        x = np.clip((np.asarray(state) - self.center) / self.scale, -10, 10)
        return float(sigmoid(x @ self.weights / np.sqrt(len(self.weights)) + self.bias))

    def arrays(self):
        return {
            "probe_center": self.center,
            "probe_scale": self.scale,
            "probe_weights": self.weights,
            "probe_bias": np.array(self.bias),
            "probe_layer": np.array(self.layer),
        }

    @classmethod
    def from_arrays(cls, arrays):
        return cls(
            arrays["probe_center"],
            arrays["probe_scale"],
            arrays["probe_weights"],
            float(arrays["probe_bias"]),
            int(arrays["probe_layer"]),
        )


def context_features(example, risk: float) -> np.ndarray:
    """Observable features only: no variant, class, span, answer, or grading metadata."""
    segments = example.context_segments
    untrusted = [s for s in segments if s.trust_level == "untrusted_data"]
    text_length = len(example.user_task) + sum(len(s.text) for s in segments)
    return np.array(
        [
            risk,
            np.log1p(text_length) / 10,
            len(untrusted) / max(1, len(segments)),
            float(any(s.trust_level == "private_data" for s in segments)),
            float(any(s.source_type == "tool_output" for s in untrusted)),
            float(any(s.source_type == "memory" for s in untrusted)),
        ],
        dtype=np.float64,
    )


class NeuralAlphaPolicy:
    """REINFORCE for a one-step MDP (contextual bandit), not sequence-level PPO.

    Categorical alpha grid includes zero. A running reward baseline reduces
    variance; selection probabilities are logged for audit/off-policy analysis.
    """

    def __init__(self, alphas, *, features=6, hidden=16, seed=42, learning_rate=0.02):
        self.alphas = np.asarray(alphas, dtype=float)
        if (
            self.alphas.ndim != 1
            or len(self.alphas) < 2
            or 0 not in self.alphas
            or len(np.unique(self.alphas)) != len(self.alphas)
            or not np.isfinite(self.alphas).all()
            or (self.alphas < 0).any()
        ):
            raise ValueError(
                "alpha actions must be unique finite nonnegative values including zero"
            )
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, 0.1, (features, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = self.rng.normal(0, 0.1, (hidden, len(alphas)))
        self.b2 = np.zeros(len(alphas))
        self.learning_rate = learning_rate
        self.baseline = 0.0
        self.updates = 0

    def probabilities(self, features):
        features = np.asarray(features, dtype=float)
        if features.shape != (self.w1.shape[0],) or not np.isfinite(features).all():
            raise ValueError("invalid policy features")
        hidden = np.tanh(features @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def choose(self, features, *, explore=False):
        p = self.probabilities(features)
        action = int(self.rng.choice(len(p), p=p)) if explore else int(np.argmax(p))
        return action, float(self.alphas[action]), float(p[action])

    def update(self, features, action, observed_reward):
        if not 0 <= action < len(self.alphas) or not np.isfinite(observed_reward):
            raise ValueError("invalid bandit observation")
        probabilities = self.probabilities(features)
        hidden = np.tanh(features @ self.w1 + self.b1)
        gradient = -probabilities
        gradient[action] += 1
        advantage = np.clip(observed_reward - self.baseline, -5, 5)
        gradient *= advantage
        grad_hidden = (self.w2 @ gradient) * (1 - hidden**2)
        rate = self.learning_rate
        self.w2 += rate * np.outer(hidden, gradient)
        self.b2 += rate * gradient
        self.w1 += rate * np.outer(features, grad_hidden)
        self.b1 += rate * grad_hidden
        self.updates += 1
        self.baseline += (observed_reward - self.baseline) / min(self.updates, 100)

    def save(self, path, probe: RiskProbe, metadata: dict):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **metadata,
            "algorithm": "neural_contextual_bandit_reinforce",
            "learning_rate": self.learning_rate,
            "baseline": self.baseline,
            "updates": self.updates,
            "rng_state": self.rng.bit_generator.state,
        }
        metadata["policy_id"] = digest(
            {
                **metadata,
                "w1": self.w1.tolist(),
                "w2": self.w2.tolist(),
                "b1": self.b1.tolist(),
                "b2": self.b2.tolist(),
            }
        )
        np.savez_compressed(
            path,
            alphas=self.alphas,
            w1=self.w1,
            w2=self.w2,
            b1=self.b1,
            b2=self.b2,
            **probe.arrays(),
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        metadata = json.loads(str(arrays["metadata"].item()))
        obj = cls(
            arrays["alphas"],
            features=arrays["w1"].shape[0],
            hidden=arrays["w1"].shape[1],
            learning_rate=metadata["learning_rate"],
        )
        for name in ("w1", "w2", "b1", "b2"):
            setattr(obj, name, arrays[name])
        obj.baseline, obj.updates = metadata["baseline"], metadata["updates"]
        obj.rng.bit_generator.state = metadata["rng_state"]
        return obj, RiskProbe.from_arrays(arrays), metadata


def binary_auc(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    positive, negative = probabilities[labels == 1], probabilities[labels == 0]
    if not len(positive) or not len(negative):
        return None
    # Pairwise comparisons, including half credit for ties; small research sets.
    return float(
        ((positive[:, None] > negative).sum() + 0.5 * (positive[:, None] == negative).sum())
        / (len(positive) * len(negative))
    )
