from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from security_steering.collection import capture_states
from security_steering.intervention import Intervention, SteeringHooks, site_module
from security_steering.model import ModelBackend
from security_steering.synthetic import build_controlled_data


@unittest.skipUnless(ML_AVAILABLE, "install the ml extra to test actual transformer hooks")
class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_layer=2,
                n_head=2,
                n_embd=16,
                n_positions=64,
                vocab_size=32,
                resid_pdrop=0,
                attn_pdrop=0,
                embd_pdrop=0,
            )
        ).eval()
        self.inputs = torch.tensor([[2, 5, 8, 13]])
        self.vector = np.arange(16, dtype=float) / 16

    def logits(self):
        with torch.inference_mode():
            return self.model(self.inputs, use_cache=False).logits.clone()

    def test_zero_alpha_is_exact_noop_and_nonzero_changes_logits(self):
        before = self.logits()
        with SteeringHooks(self.model, [Intervention(0, "residual", self.vector, 0)]):
            torch.testing.assert_close(before, self.logits(), rtol=0, atol=0)
        with SteeringHooks(self.model, [Intervention(0, "residual", self.vector, 2)]):
            self.assertFalse(torch.allclose(before[:, -1], self.logits()[:, -1]))
        torch.testing.assert_close(before, self.logits(), rtol=0, atol=0)

    def test_capture_uses_raw_last_block_output_not_final_norm(self):
        backend = ModelBackend("tiny-random", max_input_tokens=2048, threads=1)
        example = build_controlled_data()[0]
        captured = capture_states(backend, example, [2], ["residual"])[(2, "residual")]
        with torch.inference_mode():
            result = backend.model(**backend.encode(example), output_hidden_states=True)
        normalized = result.hidden_states[-1][0, -1].numpy()
        self.assertFalse(np.allclose(captured, normalized))

    def test_mlp_and_residual_both_modify_without_changing_weights(self):
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        baseline = self.logits()
        with SteeringHooks(
            self.model,
            [Intervention(0, "residual", self.vector, 1), Intervention(0, "mlp", self.vector, 1)],
        ) as hooks:
            self.assertFalse(torch.allclose(baseline, self.logits()))
            self.assertEqual(hooks.calls, {"residual:0": 1, "mlp:0": 1})
        for name, param in self.model.named_parameters():
            torch.testing.assert_close(before[name], param, rtol=0, atol=0)

    def test_hooks_cleaned_on_generation_exception_and_registration_error(self):
        module = site_module(self.model, 0, "residual")
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with SteeringHooks(self.model, [Intervention(0, "residual", self.vector, 1)]):
                raise RuntimeError("simulated")
        self.assertEqual(len(module._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with SteeringHooks(
                self.model,
                [
                    Intervention(0, "residual", self.vector, 1),
                    Intervention(99, "residual", self.vector, 1),
                ],
            ):
                pass
        self.assertEqual(len(module._forward_hooks), 0)

    def test_prefill_and_decode_control_phases_by_call_count(self):
        baseline = self.logits()
        with SteeringHooks(
            self.model, [Intervention(0, "residual", self.vector, 1, position="decode")]
        ):
            torch.testing.assert_close(baseline, self.logits(), rtol=0, atol=0)
            self.assertFalse(torch.allclose(baseline, self.logits()))
        with SteeringHooks(
            self.model, [Intervention(0, "residual", self.vector, 1, position="prefill")]
        ):
            self.assertFalse(torch.allclose(baseline, self.logits()))
            torch.testing.assert_close(baseline, self.logits(), rtol=0, atol=0)

    def test_decision_hook_does_not_change_earlier_tokens(self):
        baseline = self.logits()
        with SteeringHooks(self.model, [Intervention(0, "residual", self.vector, 1)]):
            after = self.logits()
        torch.testing.assert_close(baseline[:, :-1], after[:, :-1], rtol=0, atol=0)

    def test_bad_vectors_and_duplicate_sites_rejected(self):
        with self.assertRaises(ValueError):
            with SteeringHooks(self.model, [Intervention(0, "residual", np.ones(17), 1)]):
                self.logits()
        with self.assertRaises(ValueError):
            with SteeringHooks(self.model, [Intervention(0, "residual", self.vector, 1)] * 2):
                pass
