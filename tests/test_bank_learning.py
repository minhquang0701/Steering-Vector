from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from security_steering.bank import METHODS, VectorBank, fit_bank, representative
from security_steering.learning import NeuralAlphaPolicy, RiskProbe, binary_auc


class BankTests(unittest.TestCase):
    def records(self, path, split="train"):
        np.savez_compressed(
            path,
            states=np.array([[2, 0], [0, 0], [0, 4], [0, 0.0]]),
            example_id=np.array(["s", "a", "s", "a"]),
            family_id=np.array(["f"] * 4),
            variant=np.array(["safe", "attack"] * 2),
            attack_class=np.array(["none", "override"] * 2),
            layer=np.array([0, 0, 1, 1]),
            site=np.array(["residual"] * 4),
            split=np.array([split] * 4),
            metadata=np.array(json.dumps({"model_identity": "test"})),
        )

    def test_layers_are_never_averaged_and_prototypes_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, output = Path(tmp) / "states.npz", Path(tmp) / "bank.npz"
            self.records(records)
            bank = fit_bank(records, output)
            bank = VectorBank.load(output)
        np.testing.assert_allclose(bank.get("mean_difference", 0, "residual"), [1, 0])
        np.testing.assert_allclose(bank.get("mean_difference", 1, "residual"), [0, 1])
        np.testing.assert_allclose(bank.arrays["safe_mean__residual__l1__global"], [0, 4])

    def test_test_records_cannot_fit_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "states.npz"
            self.records(records, split="test")
            with self.assertRaisesRegex(ValueError, "train"):
                fit_bank(records, Path(tmp) / "bank.npz")

    def test_methods_orient_toward_safe(self):
        safe, attack = np.array([[2.0, 1], [3.0, 1]]), np.array([[0.0, 1], [1.0, 1]])
        for method in METHODS:
            vector = representative(safe, attack, safe, attack, method)
            self.assertGreater(vector @ (safe - attack).mean(axis=0), 0)

    def test_model_mismatch_is_rejected(self):
        bank = VectorBank({}, {"entries": [], "model_identity": "first"})
        with self.assertRaisesRegex(ValueError, "different model"):
            bank.check_backend(type("Backend", (), {"identity": "second"})())


class LearningTests(unittest.TestCase):
    def test_bandit_learns_a_rewarded_action(self):
        policy = NeuralAlphaPolicy([0, 1, 3], seed=4, learning_rate=0.08)
        features = np.array([0.8, 0.5, 0.3, 1, 1, 0])
        before = policy.probabilities(features)[1]
        for _ in range(800):
            action, _, _ = policy.choose(features, explore=True)
            policy.update(features, action, 2.0 if action == 1 else 0)
        self.assertGreater(policy.probabilities(features)[1], before + 0.5)
        self.assertEqual(policy.choose(features)[1], 1)

    def test_policy_checkpoint_roundtrip_and_rng(self):
        policy = NeuralAlphaPolicy([0, 2])
        probe = RiskProbe(np.zeros(3), np.ones(3), np.ones(3), 0.0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.npz"
            policy.save(path, probe, {"fit_split": "train"})
            loaded, other_probe, metadata = NeuralAlphaPolicy.load(path)
        features = np.ones(6)
        np.testing.assert_equal(policy.probabilities(features), loaded.probabilities(features))
        self.assertEqual(
            policy.choose(features, explore=True), loaded.choose(features, explore=True)
        )
        self.assertEqual(other_probe.predict(np.ones(3)), probe.predict(np.ones(3)))
        self.assertEqual(metadata["fit_split"], "train")

    def test_bad_alpha_grid_and_nonfinite_features_are_rejected(self):
        for grid in ([1, 2], [0, -1], [0, float("nan")], [0, 1, 1]):
            with self.assertRaises(ValueError):
                NeuralAlphaPolicy(grid)
        with self.assertRaises(ValueError):
            NeuralAlphaPolicy([0, 1]).probabilities(np.full(6, float("nan")))

    def test_auc_ties_and_missing_class(self):
        self.assertEqual(binary_auc([0, 1], [0.1, 0.9]), 1)
        self.assertEqual(binary_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertIsNone(binary_auc([0, 0], [0.1, 0.2]))
