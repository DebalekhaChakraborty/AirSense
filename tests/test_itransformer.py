"""Behavioural tests for the in-repository iTransformer-style forecaster.

Phase 12A. Written after the study completed. These tests exercise the
existing frozen implementation and change none of it.

``src/models/itransformer_forecaster.py`` carried no behavioural tests: the
only occurrence of "iTransformer" in the suite was a model name inside an
exclusion list. Its three sibling architectures each had dedicated coverage.

Scope note, stated because it bounds what these tests mean: the module is an
**in-repository adaptation** of the inverted formulation, not a reproduction
of the authors' implementation. Nothing here demonstrates equivalence to the
published iTransformer, and no test should be read as doing so. What is
asserted is that this repository's model behaves as this repository documents.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.itransformer_forecaster import (                  # noqa: E402
    ITransformerForecaster)
from src.models.neural_features import (                          # noqa: E402
    CONTEXT_HOURS, dynamic_channel_names, static_channel_names)

STATICS = len(static_channel_names())
R1_CHANNELS = len(dynamic_channel_names("R1"))
R2_CHANNELS = len(dynamic_channel_names("R2"))
TARGET_TOKEN = dynamic_channel_names("R2").index("value:PM2.5")


def build(channels=R2_CHANNELS, target_index=TARGET_TOKEN, d_model=64):
    return ITransformerForecaster(channels, STATICS, target_index,
                                  d_model=d_model)


class TestShapeContract(unittest.TestCase):
    """A and B: accepted input shapes and emitted output shape."""

    def test_a_output_is_one_scalar_per_batch_row(self):
        model = build().eval()
        with torch.no_grad():
            out = model(torch.randn(7, CONTEXT_HOURS, R2_CHANNELS),
                        torch.randn(7, STATICS))
        self.assertEqual(tuple(out.shape), (7,))

    def test_a_batch_of_one(self):
        model = build().eval()
        with torch.no_grad():
            out = model(torch.zeros(1, CONTEXT_HOURS, R2_CHANNELS),
                        torch.zeros(1, STATICS))
        self.assertEqual(tuple(out.shape), (1,))

    def test_b_accepts_the_r1_regime_channel_count(self):
        model = build(channels=R1_CHANNELS,
                      target_index=dynamic_channel_names("R1").index(
                          "value:PM2.5")).eval()
        with torch.no_grad():
            out = model(torch.randn(3, CONTEXT_HOURS, R1_CHANNELS),
                        torch.randn(3, STATICS))
        self.assertEqual(tuple(out.shape), (3,))

    def test_b_wrong_context_length_is_rejected(self):
        model = build().eval()
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                model(torch.randn(2, CONTEXT_HOURS + 1, R2_CHANNELS),
                      torch.randn(2, STATICS))

    def test_b_wrong_static_width_is_rejected(self):
        model = build().eval()
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                model(torch.randn(2, CONTEXT_HOURS, R2_CHANNELS),
                      torch.randn(2, STATICS + 1))


class TestVariatesAreTokens(unittest.TestCase):
    """C: the inverted formulation, asserted structurally."""

    def test_c_variate_embedding_consumes_the_time_dimension(self):
        model = build()
        self.assertIsInstance(model.variate_embedding, nn.Linear)
        self.assertEqual(model.variate_embedding.in_features, CONTEXT_HOURS)
        self.assertEqual(model.variate_embedding.out_features, model.d_model)

    def test_c_channel_count_does_not_change_parameter_count(self):
        """The signature of inversion: variates are tokens, not features.

        R1 has 31 dynamic channels and R2 has 46. Under a time-token model the
        parameter count would differ; under variate tokens it cannot, because
        the channel count sets the token-sequence length only.
        """
        r1 = build(channels=R1_CHANNELS,
                   target_index=dynamic_channel_names("R1").index(
                       "value:PM2.5"))
        r2 = build(channels=R2_CHANNELS)
        self.assertNotEqual(R1_CHANNELS, R2_CHANNELS)
        self.assertEqual(r1.parameter_count(), r2.parameter_count())

    def test_c_encoded_sequence_length_equals_the_channel_count(self):
        model = build().eval()
        with torch.no_grad():
            variates = torch.randn(2, CONTEXT_HOURS, R2_CHANNELS).transpose(1, 2)
            tokens = model.variate_embedding(variates)
            encoded = model.encoder(tokens)
        self.assertEqual(tuple(tokens.shape), (2, R2_CHANNELS, model.d_model))
        self.assertEqual(tuple(encoded.shape), (2, R2_CHANNELS, model.d_model))

    def test_c_attention_mixes_across_variates_not_time(self):
        """Perturbing one variate's history can move another variate's token."""
        model = build().eval()
        base_seq = torch.zeros(1, CONTEXT_HOURS, R2_CHANNELS)
        moved = base_seq.clone()
        other = (TARGET_TOKEN + 1) % R2_CHANNELS
        moved[:, :, other] += 5.0
        with torch.no_grad():
            a = model(base_seq, torch.zeros(1, STATICS))
            b = model(moved, torch.zeros(1, STATICS))
        self.assertFalse(torch.equal(a, b))


class TestTargetIndex(unittest.TestCase):
    """D: token selection, and how an invalid index behaves."""

    def test_d_target_index_is_recorded_and_selects_that_token(self):
        model = build().eval()
        self.assertEqual(model.target_index, TARGET_TOKEN)
        with torch.no_grad():
            seq = torch.randn(2, CONTEXT_HOURS, R2_CHANNELS)
            static = torch.randn(2, STATICS)
            encoded = model.encoder(
                model.variate_embedding(seq.transpose(1, 2)))
            expected = model.head(
                torch.cat([encoded[:, model.target_index, :], static],
                          dim=1)).squeeze(-1)
            torch.testing.assert_close(model(seq, static), expected)

    def test_d_out_of_range_target_index_is_exposed_not_absorbed(self):
        model = build(target_index=R2_CHANNELS).eval()
        with self.assertRaises(IndexError):
            with torch.no_grad():
                model(torch.randn(1, CONTEXT_HOURS, R2_CHANNELS),
                      torch.zeros(1, STATICS))

    def test_d_negative_target_index_wraps_silently(self):
        """Documents observed behaviour, and does not endorse it.

        Python indexing makes a negative ``target_index`` select from the end
        of the variate axis rather than raise. Every production call site
        passes ``names.index("value:PM2.5")``, which is non-negative, so the
        study never exercised this path. Recorded so the behaviour is visible
        rather than assumed.
        """
        wrapped = build(target_index=-1).eval()
        explicit = build(target_index=R2_CHANNELS - 1).eval()
        explicit.load_state_dict(wrapped.state_dict())
        seq = torch.randn(2, CONTEXT_HOURS, R2_CHANNELS)
        static = torch.randn(2, STATICS)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(seq, static),
                                       explicit(seq, static))


class TestUnconstrainedOutput(unittest.TestCase):
    """E: negative predictions must remain expressible."""

    def test_e_head_contains_no_squashing_nonlinearity(self):
        model = build()
        kinds = [type(m).__name__ for m in model.head]
        for banned in ("ReLU", "Softplus", "Sigmoid", "Hardtanh", "Clamp"):
            self.assertNotIn(banned, kinds)
        self.assertEqual(kinds[-1], "Linear")

    def test_e_a_negative_bias_yields_a_negative_prediction(self):
        model = build().eval()
        final = model.head[-1]
        with torch.no_grad():
            final.bias.fill_(-50.0)
            out = model(torch.zeros(4, CONTEXT_HOURS, R2_CHANNELS),
                        torch.zeros(4, STATICS))
        self.assertTrue(bool((out < 0).all()))

    def test_e_no_clamp_or_clip_call_in_the_module_code(self):
        """Scans executable code only, with word boundaries.

        The module docstring contains the word "clipping" while promising the
        opposite, so a naive substring scan over the whole file reports a
        constraint that is not there. The docstring is stripped and the
        remaining call sites are matched on word boundaries.
        """
        import ast
        import re
        source = (PROJECT_ROOT / "src" / "models"
                  / "itransformer_forecaster.py").read_text()
        tree = ast.parse(source)
        doc = ast.get_docstring(tree) or ""
        code = source.replace(doc, "")
        calls = {n.func.attr for n in ast.walk(ast.parse(code))
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)}
        for banned in ("clamp", "clamp_", "clip", "relu", "softplus",
                       "sigmoid", "hardtanh"):
            self.assertNotIn(banned, calls)
            self.assertIsNone(
                re.search(r"\b" + banned + r"\s*\(", code, re.I),
                "unexpected %s( call in executable code" % banned)


class TestNoDecoderOrFutureInput(unittest.TestCase):
    """F: encoder-only, and no interface accepting future values."""

    def test_f_no_decoder_attribute(self):
        model = build()
        self.assertFalse(hasattr(model, "decoder"))

    def test_f_no_decoder_submodule_registered(self):
        model = build()
        names = [n for n, _ in model.named_modules()]
        self.assertFalse(any("decoder" in n.lower() for n in names))
        self.assertTrue(any(isinstance(m, nn.TransformerEncoder)
                            for _, m in model.named_modules()))

    def test_f_forward_accepts_only_history_and_static(self):
        import inspect
        params = list(inspect.signature(
            ITransformerForecaster.forward).parameters)
        self.assertEqual(params, ["self", "sequence", "static"])

    def test_f_no_positional_encoding_across_variate_tokens(self):
        model = build()
        names = [n for n, _ in model.named_modules()]
        self.assertFalse(any("pos" in n.lower() for n in names))


class TestParameterCountAndDeterminism(unittest.TestCase):
    """G and H."""

    def test_g_parameter_count_matches_the_registered_parameters(self):
        model = build()
        self.assertEqual(model.parameter_count(),
                         sum(p.numel() for p in model.parameters()))

    def test_g_parameter_count_includes_non_trainable_parameters(self):
        model = build()
        model.variate_embedding.weight.requires_grad_(False)
        self.assertEqual(model.parameter_count(),
                         sum(p.numel() for p in model.parameters()))
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        self.assertGreater(model.parameter_count(), trainable)

    def test_g_frozen_configuration_reproduces_the_recorded_count(self):
        """Structural cross-check against artifacts/phase5_prevalidation_freeze.

        This recomputes an architectural property, not a scientific metric.
        """
        import json
        freeze = json.loads((PROJECT_ROOT / "artifacts"
                             / "phase5_prevalidation_freeze.json").read_text())
        for regime, channels in (("iTransformer_R1", R1_CHANNELS),
                                 ("iTransformer_R2", R2_CHANNELS)):
            recorded = freeze["models"][regime]["parameter_count"]
            model = build(channels=channels, target_index=0, d_model=64)
            self.assertEqual(model.parameter_count(), recorded)

    def test_h_eval_mode_inference_is_deterministic(self):
        model = build().eval()
        seq = torch.randn(5, CONTEXT_HOURS, R2_CHANNELS)
        static = torch.randn(5, STATICS)
        with torch.no_grad():
            first = model(seq, static)
            second = model(seq, static)
        self.assertTrue(torch.equal(first, second))

    def test_h_train_mode_dropout_is_active(self):
        """The converse: determinism is a property of eval mode, not the model."""
        model = build().train()
        seq = torch.randn(64, CONTEXT_HOURS, R2_CHANNELS)
        static = torch.randn(64, STATICS)
        torch.manual_seed(0)
        with torch.no_grad():
            a = model(seq, static)
            b = model(seq, static)
        self.assertFalse(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
