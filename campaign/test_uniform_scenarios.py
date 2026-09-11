"""Uniform label assignments are not bounds on the Gate C retention ratio.

An earlier version of score_gate_c claimed that imputing every unresolved row as
an error, and then as correct, "brackets every possible adjudication". It does not.
AUROC is not monotone in label flips, and the registered statistic is a RATIO of
two AUROC differences, so a mixed assignment can fall outside both uniform ones
even with every prediction held fixed.

This test pins the counterexample so the guarantee cannot be reintroduced. Four
rows, two of them unresolved:

    baseline = [1, 3, 2, 4]   shared logprob-only baseline
    full     = [1, 2, 3, 4]   full-answer combined detector
    prefix   = [3, 1, 4, 2]   prefix combined detector

    both unresolved correct -> ratio 2.00  HIT
    both unresolved errors  -> ratio 2.00  HIT
    mixed [0, 0, 1, 1]      -> ratio 0.00  MISS

The consequence for the scorer: agreement between uniform scenarios is weak
reassurance, and only adjudication can settle the gate.
"""

import itertools
import unittest

from sklearn.metrics import roc_auc_score

BASELINE = [1, 3, 2, 4]
FULL = [1, 2, 3, 4]
PREFIX = [3, 1, 4, 2]
# Index 0 and 3 are unresolved; index 1 is known correct, index 2 known error.
KNOWN = {1: 0, 2: 1}
THRESHOLD = 0.90


def retention_ratio(labels):
    base = roc_auc_score(labels, BASELINE)
    full = roc_auc_score(labels, FULL)
    prefix = roc_auc_score(labels, PREFIX)
    denominator = full - base
    if denominator <= 0:
        return None
    return (prefix - base) / denominator


def verdict(labels):
    ratio = retention_ratio(labels)
    if ratio is None:
        return "UNDECIDABLE"
    return "HIT" if ratio >= THRESHOLD else "MISS"


def assignment(first, last):
    return [first, KNOWN[1], KNOWN[2], last]


class TestUniformScenariosAreNotBounds(unittest.TestCase):
    def test_both_uniform_assignments_are_hits(self) -> None:
        for value in (0, 1):
            labels = assignment(value, value)
            self.assertAlmostEqual(retention_ratio(labels), 2.0, places=6)
            self.assertEqual(verdict(labels), "HIT")

    def test_a_mixed_assignment_falls_outside_both(self) -> None:
        labels = assignment(0, 1)
        self.assertAlmostEqual(retention_ratio(labels), 0.0, places=6)
        self.assertEqual(verdict(labels), "MISS")

    def test_the_uniform_pair_does_not_enclose_every_assignment(self) -> None:
        uniform = {retention_ratio(assignment(v, v)) for v in (0, 1)}
        low, high = min(uniform), max(uniform)
        outside = [
            (first, last)
            for first, last in itertools.product((0, 1), repeat=2)
            for ratio in [retention_ratio(assignment(first, last))]
            if ratio is not None and not (low - 1e-9 <= ratio <= high + 1e-9)
        ]
        self.assertTrue(
            outside,
            "counterexample no longer reproduces; the bounds claim would be safe "
            "again and this test needs rewriting rather than deleting")

    def test_agreeing_scenarios_do_not_imply_a_stable_verdict(self) -> None:
        uniform_verdicts = {verdict(assignment(v, v)) for v in (0, 1)}
        self.assertEqual(uniform_verdicts, {"HIT"})
        all_verdicts = {
            verdict(assignment(first, last))
            for first, last in itertools.product((0, 1), repeat=2)
        }
        self.assertIn("MISS", all_verdicts,
                      "uniform agreement coexists with a differing mixed assignment")


class TestScorerDoesNotClaimBounds(unittest.TestCase):
    """The wording matters: a field named like a guarantee gets read as one."""

    def test_scorer_has_no_bounds_guarantee_field(self) -> None:
        import io
        from pathlib import Path

        source = io.open(
            Path(__file__).with_name("score_gate_c.py"), encoding="utf-8").read()
        self.assertNotIn("verdict_stable_under_every_assignment", source)
        self.assertNotIn("brackets every possible adjudication", source)
        self.assertIn("uniform_scenarios_agree", source)
        self.assertIn("PROVISIONAL_UNADJUDICATED", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
