"""Fast tests for the Self-Focus Lab. No model load.

Model-dependent tests (hook agreement, hook removal, cache isolation,
future-token invariance) live in test_self_focus_model.py because they need
weights on the GPU.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.self_focus import core


class TestStableIdentity(unittest.TestCase):
    def test_ids_are_stable_across_processes(self) -> None:
        # hashlib, not hash(): the builtin is salted per process, so a trial id
        # built from it would differ between runs and break resume.
        a = core.stable_id("x", 1, {"k": "v"})
        b = core.stable_id("x", 1, {"k": "v"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, core.stable_id("x", 2, {"k": "v"}))

    def test_id_depends_on_every_part(self) -> None:
        base = core.stable_id("cond", "task", "fp")
        self.assertNotEqual(base, core.stable_id("cond", "task", "fp2"))
        self.assertNotEqual(base, core.stable_id("cond2", "task", "fp"))


class TestLocalOnly(unittest.TestCase):
    def test_shipped_config_authorizes_no_spending(self) -> None:
        cfg = core.config()
        self.assertEqual(cfg["cloud_spend_authorization_usd"], 0.0)
        self.assertFalse(cfg["allow_remote"])
        self.assertFalse(cfg["allow_paid_judge"])
        core.assert_local_only(cfg)

    def test_remote_or_paid_config_is_refused(self) -> None:
        cfg = core.config()
        for key, value in (("allow_remote", True), ("allow_paid_judge", True),
                           ("cloud_spend_authorization_usd", 5.0)):
            bad = dict(cfg)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                core.assert_local_only(bad)

    def test_dry_run_cannot_start_remote_compute(self) -> None:
        from experiments.self_focus import runner
        self.assertEqual(runner.main(["preflight", "--dry-run"]), 0)
        self.assertEqual(runner.main(["demo", "--dry-run"]), 0)


class TestPromptAssembly(unittest.TestCase):
    def setUp(self) -> None:
        self.conds = core.conditions()

    def test_every_condition_shares_the_outer_instruction_and_task(self) -> None:
        task = "TASK-SENTINEL"
        for name in self.conds["conditions"]:
            prompt = core.build_prompt(name, task, self.conds)
            self.assertIn(self.conds["shared_outer"], prompt)
            self.assertIn(task, prompt)
            self.assertIn("--- instruction ---", prompt)

    def test_condition_id_is_never_shown_to_the_model(self) -> None:
        for name in self.conds["conditions"]:
            prompt = core.build_prompt(name, "task", self.conds)
            self.assertNotIn(name, prompt,
                             f"condition id {name} leaked into its own prompt")

    def test_only_jspace_conditions_carry_the_explanation(self) -> None:
        explanation = self.conds["jspace_explanation"]
        for name, spec in self.conds["conditions"].items():
            prompt = core.build_prompt(name, "task", self.conds)
            self.assertEqual(explanation in prompt, spec["uses_jspace_explanation"], name)

    def test_instruction_text_is_verbatim_from_the_protocol(self) -> None:
        for name, spec in self.conds["conditions"].items():
            self.assertIn(spec["instruction"], core.build_prompt(name, "t", self.conds))

    def test_registered_contrasts_reference_real_conditions(self) -> None:
        for a, b in self.conds["primary_contrasts"]:
            self.assertIn(a, self.conds["conditions"])
            self.assertIn(b, self.conds["conditions"])


class TestTrialStore(unittest.TestCase):
    def test_resume_skips_done_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            store = core.TrialStore(path, "fp-1")
            store.write({"trial_id": "a", "x": 1})
            store.close()

            reopened = core.TrialStore(path, "fp-1")
            self.assertTrue(reopened.has("a"))
            with self.assertRaises(RuntimeError):
                reopened.write({"trial_id": "a", "x": 2})
            reopened.close()

    def test_changed_provenance_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            store = core.TrialStore(path, "fp-1")
            store.write({"trial_id": "a"})
            store.close()
            with self.assertRaises(RuntimeError):
                core.TrialStore(path, "fp-DIFFERENT")

    def test_rows_are_flushed_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            store = core.TrialStore(path, "fp")
            store.write({"trial_id": "a"})
            # readable before close: a crash must not lose completed trials
            rows = core.read_trials(path)
            self.assertEqual(len(rows), 1)
            store.close()


class TestBudget(unittest.TestCase):
    def test_budget_accumulates_and_reports(self) -> None:
        b = core.Budget(minutes=1.0)
        with b:
            pass
        self.assertGreaterEqual(b.used_s, 0.0)
        self.assertFalse(b.exhausted())
        self.assertIn("remaining_minutes", b.summary())

    def test_zero_budget_is_immediately_exhausted(self) -> None:
        self.assertTrue(core.Budget(minutes=0.0).exhausted())


class TestLayerSelection(unittest.TestCase):
    def test_at_most_four_layers_from_the_lens_band(self) -> None:
        cfg = core.config()
        band = list(range(10, 31))
        out = core.select_layers(42, cfg, band)
        self.assertLessEqual(len(out["layers"]), cfg["max_recorded_layers"])
        self.assertTrue(set(out["layers"]).issubset(band))
        self.assertEqual(out["layers"], sorted(out["layers"]))
        self.assertTrue(out["zero_based"])

    def test_works_without_a_lens(self) -> None:
        out = core.select_layers(42, core.config(), None)
        self.assertLessEqual(len(out["layers"]), 4)
        self.assertIn("no lens", out["basis"])

    def test_missing_lens_is_reported_not_silently_ignored(self) -> None:
        band, info = core.lens_band_from("definitely/not/here.pt", 42)
        self.assertIsNone(band)
        self.assertFalse(info["available"])
        self.assertIn("reason", info)


class TestFixedContinuations(unittest.TestCase):
    def test_two_neutral_passages_are_present_verbatim(self) -> None:
        conds = core.conditions()
        self.assertEqual(len(conds["fixed_continuations"]), 2)
        self.assertTrue(conds["fixed_continuations"][0].startswith("The small wooden box"))
        self.assertTrue(conds["fixed_continuations"][1].startswith("A blue notebook"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
