from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from security_steering.augmentation import generate_candidates
from security_steering.dataset import validate_examples


class AugmentationTests(unittest.TestCase):
    def test_generated_paraphrases_record_provenance_and_need_review(self):
        class Backend:
            is_smoke = False
            identity = "test-generator-identity"
            name = "test-generator"

            def generate_messages(self, messages, **kwargs):
                instruction = messages[-1]["content"].split("\n", 1)[1]
                return SimpleNamespace(text="Updated instruction: " + instruction)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidates.jsonl"
            examples = generate_candidates(Backend(), output, count=7)
            self.assertTrue(output.with_suffix(".generation.json").is_file())
        self.assertEqual(len(examples), 21)
        self.assertTrue(validate_examples(examples).ok)
        self.assertTrue(all(e.metadata["review_status"] == "pending" for e in examples))
        self.assertTrue(all(e.metadata["generator_identity"] == Backend.identity for e in examples))

    def test_no_fabricated_fallback_on_failed_generations(self):
        backend = SimpleNamespace(
            is_smoke=False,
            identity="x",
            name="x",
            generate_messages=lambda *a, **kw: SimpleNamespace(text=""),
        )
        with tempfile.TemporaryDirectory() as tmp:
            examples = generate_candidates(backend, Path(tmp) / "empty.jsonl", count=2)
        self.assertEqual(examples, [])

    def test_random_model_cannot_be_reported_as_llm_data(self):
        with self.assertRaisesRegex(ValueError, "pretrained"):
            generate_candidates(SimpleNamespace(is_smoke=True), "unused.jsonl")
