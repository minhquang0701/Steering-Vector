from __future__ import annotations

import unittest

import numpy as np

from security_steering.controller import compose_steering_vector, risk_gated_alpha
from security_steering.hidden_states import candidate_layers, pool_anchor_states
from security_steering.vectors import (
    add_steering,
    mean_difference_vector,
    probe_direction_vector,
    projection_remove,
)


class VectorTests(unittest.TestCase):
    def test_mean_difference_points_from_attack_to_safe(self) -> None:
        safe = np.array([[2.0, 0.0], [2.0, 2.0]])
        attack = np.array([[0.0, 0.0], [0.0, 2.0]])

        vector = mean_difference_vector(safe, attack, normalize=False)

        np.testing.assert_allclose(vector, np.array([2.0, 0.0]))

    def test_probe_direction_is_defensive_negative_weight(self) -> None:
        vector = probe_direction_vector(np.array([3.0, 4.0]))

        np.testing.assert_allclose(vector, np.array([-0.6, -0.8]))

    def test_projection_remove_only_removes_positive_attack_component(self) -> None:
        hidden = np.array([[2.0, 1.0], [-2.0, 1.0]])
        cleaned = projection_remove(hidden, np.array([1.0, 0.0]))

        np.testing.assert_allclose(cleaned, np.array([[0.0, 1.0], [-2.0, 1.0]]))

    def test_add_steering_supports_token_mask(self) -> None:
        hidden = np.zeros((3, 2))
        vector = np.array([1.0, -1.0])
        steered = add_steering(hidden, vector, alpha=2.0, mask=np.array([1.0, 0.0, 1.0]))

        np.testing.assert_allclose(steered, np.array([[2.0, -2.0], [0.0, 0.0], [2.0, -2.0]]))

    def test_risk_gated_alpha_increases_with_risk(self) -> None:
        self.assertLess(risk_gated_alpha(0.1), risk_gated_alpha(0.9))

    def test_compose_steering_vector_uses_class_risks(self) -> None:
        bank = {
            "override": np.array([1.0, 0.0]),
            "reveal": np.array([0.0, 1.0]),
        }
        composed = compose_steering_vector(
            bank,
            {"override": 0.9, "reveal": 0.1},
            alpha_max=1.0,
            threshold=0.5,
        )

        self.assertGreater(composed[0], composed[1])

    def test_candidate_layers_are_unique_and_in_range(self) -> None:
        layers = candidate_layers(32)

        self.assertEqual(layers, sorted(set(layers)))
        self.assertTrue(all(0 <= layer < 32 for layer in layers))

    def test_pool_anchor_states_averages_selected_tokens(self) -> None:
        states = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
        pooled = pool_anchor_states(states, [0, -1])

        np.testing.assert_allclose(pooled, np.array([3.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
