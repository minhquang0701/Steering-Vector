from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from security_steering.dataset import write_jsonl
from security_steering.experiments import ExperimentSpec, interventions_for, run_pipeline
from security_steering.bank import VectorBank
from security_steering.splits import split_examples
from security_steering.synthetic import build_controlled_data
from test_intervention import ML_AVAILABLE

import numpy as np


class ControlTests(unittest.TestCase):
    def test_random_controls_are_reproducible_and_matched(self):
        bank = VectorBank(
            {f"mean_difference__residual__l{i}__global": np.ones(4) for i in range(6)},
            {"entries": []},
        )
        example = build_controlled_data()[1]
        spec = ExperimentSpec("random", mode="random_layer", layers=(1, 2), alpha=2)
        first, layers = interventions_for(spec, bank, example, list(range(6)))
        second, other = interventions_for(spec, bank, example, list(range(6)))
        self.assertEqual(layers, other)
        self.assertEqual(len(set(layers)), 2)
        self.assertEqual(sum(item.alpha for item in first), 2)
        np.testing.assert_equal(first[0].vector, second[0].vector)


@unittest.skipUnless(ML_AVAILABLE, "install ml extra")
class PipelineTests(unittest.TestCase):
    def test_end_to_end_and_resume_reject_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = build_controlled_data(3)
            families = {e.family_id for e in data if e.attack_class == "override"}
            data = split_examples([e for e in data if e.family_id in families])
            dataset = tmp / "data.jsonl"
            write_jsonl(dataset, data)
            config = {
                "dataset": str(dataset),
                "output_dir": str(tmp / "run"),
                "model": {"name": "tiny-random", "threads": 1},
                "layers": [0, 2],
                "sites": ["residual", "both"],
                "methods": ["mean_difference"],
                "alphas": [0, 1],
                "max_new_tokens": 2,
                "random_seeds": [9],
                "controller": {"epochs": 1},
            }
            manifest = run_pipeline(config)
            self.assertTrue(manifest["is_smoke"])
            output = tmp / "run"
            self.assertTrue((output / "controller.npz").is_file())
            self.assertTrue((output / "results_table.tex").is_file())
            summaries = json.loads((output / "summary.json").read_text())
            self.assertIn("adaptive_alpha", summaries)
            self.assertIn("random_layer_s9", summaries)
            before = (output / "responses.jsonl").read_text()
            run_pipeline(config, resume=True)
            self.assertEqual(before, (output / "responses.jsonl").read_text())
            with self.assertRaisesRegex(ValueError, "differs"):
                run_pipeline({**config, "max_new_tokens": 3}, resume=True)
