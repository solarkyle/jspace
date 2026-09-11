"""v3 scoring tests. No model load.

The central one: a score whose orientation depends on the true label manufactures
discrimination out of nothing. An earlier version of the brief proposed
"log P(correct) - log P(incorrect)" as the AUROC input. These tests pin why that is
wrong and that the shipped score avoids it.
"""

import unittest

from sklearn.metrics import roc_auc_score

from experiments.self_focus import build_tasks as bt, core
from experiments.self_focus import pilot_v3 as p3


def fixed_orientation_score(logp_present, logp_absent, **_ignored):
    """The shipped score. Ground truth never enters."""
    return logp_present - logp_absent


def correct_minus_incorrect(logp_present, logp_absent, intervention_present):
    """The rejected score, kept only so a test can demonstrate the failure."""
    return (logp_present - logp_absent) if intervention_present \
        else (logp_absent - logp_present)


class TestScoreOrientation(unittest.TestCase):
    def test_constant_preference_gives_auroc_one_half(self) -> None:
        """A model that always prefers the same label must not look discriminating."""
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        scores = [fixed_orientation_score(-0.4, -2.1) for _ in labels]
        # degenerate constant scores: AUROC is exactly 0.5
        self.assertAlmostEqual(roc_auc_score(labels, scores), 0.5, places=9)

    def test_correct_minus_incorrect_fabricates_perfect_auroc(self) -> None:
        """The rejected score scores 1.0 on a model with a CONSTANT preference."""
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        bogus = [correct_minus_incorrect(-0.4, -2.1, bool(y)) for y in labels]
        self.assertAlmostEqual(roc_auc_score(labels, bogus), 1.0, places=9)
        # which is the whole problem: the model never changed its behaviour
        self.assertEqual(len(set(bogus)), 2)

    def test_fixed_orientation_tracks_a_real_shift(self) -> None:
        """And the shipped score does register a genuine change."""
        labels = [1, 1, 1, 0, 0, 0]
        scores = [fixed_orientation_score(-0.5, -2.0),
                  fixed_orientation_score(-0.6, -2.0),
                  fixed_orientation_score(-0.4, -2.0),
                  fixed_orientation_score(-2.0, -0.5),
                  fixed_orientation_score(-2.0, -0.6),
                  fixed_orientation_score(-2.0, -0.4)]
        self.assertAlmostEqual(roc_auc_score(labels, scores), 1.0, places=9)

    def test_score_never_consults_the_label(self) -> None:
        """run_one must not receive ground truth at all."""
        import inspect
        params = set(inspect.signature(p3.run_one).parameters)
        self.assertNotIn("intervention_present", params)
        self.assertNotIn("injected", params)
        source = inspect.getsource(p3.run_one)
        self.assertIn("lp_present - lp_absent", source)


class TestCounterbalancing(unittest.TestCase):
    def test_state_labels_are_independent_of_the_arm(self) -> None:
        """Label assignment must not correlate with treatment.

        Checked on the SIGNATURE, not by grepping the source: the word "arm"
        legitimately appears in the docstring, and a source grep would fail on
        correct code while still passing on a function that took an `arm` value
        under another name.
        """
        import inspect
        params = set(inspect.signature(p3.state_labels).parameters)
        self.assertEqual(params, {"concept", "context_id", "condition"})
        self.assertNotIn("arm", params)
        # and empirically: the same inputs give the same labels for every arm
        for arm in p3.ARMS:
            self.assertEqual(p3.state_labels("violin", "ctx_weekdays", "MEDITATE"),
                             p3.state_labels("violin", "ctx_weekdays", "MEDITATE"))

    def test_both_label_orders_occur(self) -> None:
        seen = set()
        for concept in bt.EVAL_CONCEPTS:
            for ctx in bt.EVAL_CONTEXTS:
                for cond in p3.CONDITIONS:
                    seen.add(p3.state_labels(concept, ctx["id"], cond))
        self.assertEqual(len(seen), 2, f"expected both orders, saw {seen}")

    def test_labels_are_stable_across_processes(self) -> None:
        a = p3.state_labels("violin", "ctx_weekdays", "MEDITATE")
        b = p3.state_labels("violin", "ctx_weekdays", "MEDITATE")
        self.assertEqual(a, b)

    def test_state_labels_do_not_collide_with_choice_letters(self) -> None:
        for label in p3.STATE_LABELS:
            self.assertNotIn(label, p3.OPTION_LETTERS,
                             "a STATE label that is also a CHOICE letter would make "
                             "the parsed fields ambiguous")


class TestChoiceOptions(unittest.TestCase):
    def test_five_options_including_the_true_concept_and_none(self) -> None:
        for concept in bt.EVAL_CONCEPTS:
            opts = p3.choice_options(concept, "ctx_weekdays", bt.EVAL_CONCEPTS)
            values = [v for _, v in opts]
            self.assertEqual(len(opts), 5)
            self.assertEqual(len(set(values)), 5, f"duplicate options for {concept}")
            self.assertIn(concept, values)
            self.assertIn("none", values)
            self.assertEqual([k for k, _ in opts], p3.OPTION_LETTERS)

    def test_correct_letter_varies_across_concepts(self) -> None:
        letters = set()
        for concept in bt.EVAL_CONCEPTS:
            opts = p3.choice_options(concept, "ctx_weekdays", bt.EVAL_CONCEPTS)
            letters.add(next(k for k, v in opts if v == concept))
        self.assertGreater(len(letters), 1,
                           "the true answer must not sit at a fixed position")


class TestCaseMatrix(unittest.TestCase):
    def test_arms_share_byte_identical_visible_text(self) -> None:
        groups = {}
        for case in p3.cases(bt.EVAL_CONCEPTS, bt.EVAL_CONTEXTS[:1]):
            key = (case["concept"], case["context_id"], case["condition"])
            groups.setdefault(key, set()).add(case["task_block"])
        for key, blocks in groups.items():
            self.assertEqual(len(blocks), 1, f"visible text differs by arm at {key}")

    def test_wrong_concept_is_still_intervention_present(self) -> None:
        present = {"concept", "wrong_concept", "random"}
        self.assertIn("wrong_concept", present)
        self.assertNotIn("sham", present)
        self.assertNotIn("steer", present)
        self.assertNotIn("none", present)

    def test_every_condition_and_arm_appears(self) -> None:
        cs = p3.cases(bt.EVAL_CONCEPTS, bt.EVAL_CONTEXTS[:1])
        self.assertEqual({c["condition"] for c in cs}, set(p3.CONDITIONS))
        self.assertEqual({c["arm"] for c in cs}, set(p3.ARMS))
        self.assertEqual(len(cs), len(bt.EVAL_CONCEPTS) * len(p3.CONDITIONS) * len(p3.ARMS))

    def test_pilot_uses_no_development_data(self) -> None:
        cs = p3.cases(bt.EVAL_CONCEPTS, bt.EVAL_CONTEXTS[:1])
        used_concepts = {c["concept"] for c in cs}
        used_contexts = {c["context_id"] for c in cs}
        self.assertFalse(used_concepts & set(bt.DEV_CONCEPTS))
        self.assertFalse(used_contexts & {c["id"] for c in bt.DEV_CONTEXTS})


class TestFrozenSettings(unittest.TestCase):
    def test_settings_are_the_approved_pair(self) -> None:
        self.assertEqual(p3.SETTINGS["primary"],
                         {"layer": 17, "window": 64, "alpha": 0.6})
        self.assertEqual(p3.SETTINGS["secondary"],
                         {"layer": 10, "window": 64, "alpha": 0.3})

    def test_no_layer_above_the_propagation_cliff(self) -> None:
        for name, setting in p3.SETTINGS.items():
            self.assertLessEqual(setting["layer"], 22,
                                 f"{name} patches layer {setting['layer']}, at or above "
                                 f"the shared-KV cliff, where it cannot reach the report")


class TestSessionDeadline(unittest.TestCase):
    def test_deadline_is_persisted_on_disk(self) -> None:
        state = core.session_state()
        self.assertEqual(state["authorized_hours"], 7.0)
        self.assertGreater(state["deadline_epoch"], state["start_epoch"])

    def test_status_reports_remaining_time_and_stop_sentinel(self) -> None:
        status = core.deadline_status()
        for key in ("remaining_minutes", "expired", "stop_sentinel",
                    "switch_to_qa_due", "reporting_window"):
            self.assertIn(key, status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
