from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from security_steering.dataset import load_jsonl, validate_examples, write_jsonl
from security_steering.schemas import Variant
from security_steering.seed import build_seed_examples


class DatasetTests(unittest.TestCase):
    def test_seed_examples_validate(self) -> None:
        examples = build_seed_examples()
        report = validate_examples(examples)

        self.assertTrue(report.ok, report.errors)
        self.assertGreaterEqual(report.variant_counts[Variant.SAFE.value], 7)
        self.assertGreaterEqual(report.variant_counts[Variant.ATTACK.value], 7)
        self.assertGreaterEqual(report.variant_counts[Variant.HARD_BENIGN.value], 3)

    def test_jsonl_round_trip(self) -> None:
        examples = build_seed_examples()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed.jsonl"
            write_jsonl(path, examples)
            loaded = load_jsonl(path)

        self.assertEqual(
            [example.to_dict() for example in examples],
            [example.to_dict() for example in loaded],
        )


if __name__ == "__main__":
    unittest.main()
