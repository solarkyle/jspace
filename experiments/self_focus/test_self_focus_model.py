"""Model-dependent correctness tests. Loads weights once, runs short passes.

These are the load-bearing ones from the brief: an intervention harness that
silently perturbs a capture, leaks a hook, or lets one trial's cache reach
another would invalidate every downstream measurement.

Run: python -m unittest experiments.self_focus.test_self_focus_model
Skipped automatically when no CUDA device is present.
"""

import unittest

from experiments.self_focus import core

try:
    import torch
    CUDA = torch.cuda.is_available()
except Exception:
    CUDA = False

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = core.Model(core.config())
    return _MODEL


PROMPT = "Copy this exactly: the lamp is on the table."
OTHER_PROMPT = "Copy this exactly: a kite drifted over the harbour."


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestHookFidelity(unittest.TestCase):
    """A capture must observe, and a zero-dose patch must be a no-op."""

    def test_zero_strength_patch_equals_no_patch(self) -> None:
        m = model()
        layers = [17]
        cont = m.tokenizer("the lamp is on the table.", add_special_tokens=False)["input_ids"]
        plain = m.capture(PROMPT, layers, forced_continuation_ids=cont)
        direction = torch.ones(2560)
        zero = m.capture(PROMPT, layers, forced_continuation_ids=cont,
                         patch={"layer": 17, "direction": direction, "alpha": 0.0,
                                "positions": [5, 6, 7]})
        a = plain["layers"][17]["residual_norms"]
        b = zero["layers"][17]["residual_norms"]
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=4,
                                   msg="a zero-strength patch changed the residuals")

    def test_a_real_patch_is_visible_at_the_patched_positions(self) -> None:
        """Otherwise test_zero_strength_patch_equals_no_patch passes on a dead hook.

        Forward hooks fire in registration order, so the patcher is registered
        before the recorders. With the old order the recorder saw the pre-patch
        output and a live patch looked like a no-op.
        """
        m = model()
        positions = [5, 6, 7]
        plain = m.capture(PROMPT, [17], positions=positions)
        direction = torch.randn(2560, generator=torch.Generator().manual_seed(1))
        patched = m.capture(PROMPT, [17], positions=positions,
                            patch={"layer": 17, "direction": direction, "alpha": 0.25,
                                   "positions": positions})
        a = plain["layers"][17]["residual_norms"]
        b = patched["layers"][17]["residual_norms"]
        self.assertTrue(any(abs(x - y) > 1e-3 for x, y in zip(a, b)),
                        "the patched layer did not reflect its own patch")

    def test_a_patch_propagates_downstream(self) -> None:
        """A patch at layer L must reach a later layer at later positions.

        This is the property an intervention experiment depends on: the patched
        content has to actually take part in the rest of the computation.
        """
        m = model()
        cont = m.tokenizer("the lamp is on the table.", add_special_tokens=False)["input_ids"]
        plain = m.capture(PROMPT, [23], forced_continuation_ids=cont)
        direction = torch.randn(2560, generator=torch.Generator().manual_seed(2))
        patched = m.capture(PROMPT, [23], forced_continuation_ids=cont,
                            patch={"layer": 17, "direction": direction, "alpha": 0.5,
                                   "positions": [5, 6, 7]})
        a = plain["layers"][23]["residual_norms"]
        b = patched["layers"][23]["residual_norms"]
        self.assertTrue(any(abs(x - y) > 1e-3 for x, y in zip(a, b)),
                        "a layer-17 patch never reached layer 23")

    def test_a_patch_at_the_patched_layer_leaves_other_positions_alone(self) -> None:
        """Documents the mechanism, so a future reader does not call it a bug.

        Patching layer L at positions P changes layer L's output only at P. Other
        positions at that same layer are untouched until the next layer attends
        across them, which is why interventions are read downstream.
        """
        m = model()
        later = [12, 13, 14]   # in range for this 20-token prompt
        plain = m.capture(PROMPT, [17], positions=later)
        direction = torch.randn(2560, generator=torch.Generator().manual_seed(3))
        patched = m.capture(PROMPT, [17], positions=later,
                            patch={"layer": 17, "direction": direction, "alpha": 0.5,
                                   "positions": [5, 6, 7]})
        for x, y in zip(plain["layers"][17]["residual_norms"],
                        patched["layers"][17]["residual_norms"]):
            self.assertAlmostEqual(x, y, places=4)

    def test_capture_does_not_alter_the_forward_pass(self) -> None:
        m = model()
        bare = m.generate(PROMPT, 8)
        # capture hooks return output untouched, so generation must be identical
        m.capture(PROMPT, [10, 17, 23, 30])
        after = m.generate(PROMPT, 8)
        self.assertEqual(bare["gen_token_ids"], after["gen_token_ids"])


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestHookHygiene(unittest.TestCase):
    def _hook_counts(self, m):
        return [len(b._forward_hooks) for b in m.blocks]

    def test_no_hooks_remain_after_a_normal_capture(self) -> None:
        m = model()
        before = self._hook_counts(m)
        m.capture(PROMPT, [10, 17])
        self.assertEqual(self._hook_counts(m), before)

    def test_no_hooks_remain_after_an_exception_inside_the_forward(self) -> None:
        m = model()
        before = self._hook_counts(m)
        # a direction of the wrong width raises inside the patch hook
        with self.assertRaises(Exception):
            m.capture(PROMPT, [17],
                      patch={"layer": 17, "direction": torch.ones(7), "alpha": 0.5,
                             "positions": [3]})
        self.assertEqual(self._hook_counts(m), before,
                         "a hook survived an exception and would contaminate later trials")


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestTrialIsolation(unittest.TestCase):
    def test_no_cache_carries_between_trials(self) -> None:
        m = model()
        first = m.generate(PROMPT, 10)
        m.generate(OTHER_PROMPT, 10)
        repeat = m.generate(PROMPT, 10)
        self.assertEqual(first["gen_token_ids"], repeat["gen_token_ids"],
                         "an intervening trial changed a later identical trial")

    def test_each_trial_renders_a_fresh_single_turn_prompt(self) -> None:
        m = model()
        a = m.generate(PROMPT, 4)
        b = m.generate(PROMPT, 4)
        self.assertEqual(a["rendered_prompt"], b["rendered_prompt"])
        self.assertEqual(a["prompt_token_ids"], b["prompt_token_ids"])


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestNoFutureLeakage(unittest.TestCase):
    def test_changing_later_tokens_cannot_move_an_earlier_readout(self) -> None:
        """Causal attention guarantees this; the test guards the plumbing."""
        m = model()
        base = m.tokenizer("one two three four five six",
                           add_special_tokens=False)["input_ids"]
        altered = list(base)
        altered[-2:] = m.tokenizer(" zebra kite", add_special_tokens=False)["input_ids"][:2]
        self.assertNotEqual(list(base), altered, "fixture failed to change a later token")

        rendered = m.render(PROMPT)
        n_prompt = len(m.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        early = [n_prompt - 1, n_prompt]

        a = m.capture(PROMPT, [17], forced_continuation_ids=base, positions=early)
        b = m.capture(PROMPT, [17], forced_continuation_ids=altered, positions=early)
        for x, y in zip(a["layers"][17]["residual_norms"], b["layers"][17]["residual_norms"]):
            self.assertAlmostEqual(x, y, places=4,
                                   msg="an earlier readout moved when only later tokens changed")


@unittest.skipUnless(CUDA, "needs a CUDA device")
class TestPromptLimit(unittest.TestCase):
    def test_out_of_range_positions_are_rejected_not_clipped(self) -> None:
        m = model()
        with self.assertRaises(ValueError):
            m.capture(PROMPT, [17], positions=[10_000])

    def test_oversized_prompts_are_rejected_not_truncated(self) -> None:
        m = model()
        huge = "word " * (int(core.config()["max_prompt_tokens"]) + 200)
        with self.assertRaises(ValueError):
            m.generate(huge, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
