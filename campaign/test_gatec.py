"""Gate C online-safety tests.

The original campaign's failure was not arithmetic, it was leakage: whole-answer
logprob aggregates and the final answer length were handed to every checkpoint,
including the first token. These tests pin the property that makes an
early-warning feature legitimate -- a score at token k must be invariant to
everything the model emits after token k.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign.online_features import prefix_lp


class TestPrefixLP(unittest.TestCase):
    def test_future_tokens_cannot_change_an_early_score(self):
        base = [-0.1, -0.5, -0.2, -3.0, -0.4]
        for k in range(len(base)):
            tampered = base[: k + 1] + [-99.0] * (len(base) - k - 1)
            self.assertEqual(
                prefix_lp(base, k),
                prefix_lp(tampered, k),
                f"checkpoint {k} leaked information from a later token",
            )

    def test_length_feature_is_tokens_so_far_not_final_length(self):
        lp = [-0.1] * 40
        self.assertEqual(prefix_lp(lp, 0)["pfxlp_tokens_so_far"], 1.0)
        self.assertEqual(prefix_lp(lp, 7)["pfxlp_tokens_so_far"], 8.0)

    def test_aggregates_cover_only_the_window(self):
        lp = [-1.0, -1.0, -9.0]
        f = prefix_lp(lp, 1)
        self.assertAlmostEqual(f["pfxlp_mean"], -1.0)
        self.assertAlmostEqual(f["pfxlp_min"], -1.0)
        self.assertAlmostEqual(f["pfxlp_last"], -1.0)
        self.assertAlmostEqual(f["pfxlp_first"], -1.0)

    def test_first_token_is_always_the_true_first(self):
        lp = [-2.5, -0.1, -0.1]
        for k in range(3):
            self.assertAlmostEqual(prefix_lp(lp, k)["pfxlp_first"], -2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
