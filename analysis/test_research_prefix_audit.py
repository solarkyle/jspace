import copy
import unittest

from analysis.research_prefix_audit import SCALARS, available_features, describe


class PrefixAuditTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "logprob_features": {"bl_first_token_logprob": -.4, "bl_mean_logprob": -1},
            "onset_workspace_features": dict.fromkeys(SCALARS, 1.),
            "prefix_workspace_features": [
                {"frac": frac, **dict.fromkeys(SCALARS, val)}
                for frac, val in ((0, 1.), (.5, 2.), (1, 3.))]}

    def test_no_future_information(self):
        changed = copy.deepcopy(self.row)
        changed["logprob_features"]["bl_mean_logprob"] = 999
        changed["logprob_features"]["bl_min_logprob"] = 999
        changed["logprob_features"]["bl_answer_len"] = 999
        changed["answer"] = "a changed future answer"
        changed["prefix_workspace_features"][2]["mean_entropy"] = 999
        for f in ("first_lp", "onset", "mid"):
            self.assertEqual(available_features(self.row, f), available_features(changed, f))
        self.assertNotEqual(available_features(self.row, "end"), available_features(changed, "end"))

    def test_mid_deltas_and_dimensions(self):
        self.assertEqual(available_features(self.row, "mid"), [-.4] + [1.] * 12)
        self.assertEqual(len(available_features(self.row, "onset")), 7)

    def test_duplicate_checkpoint_count(self):
        rows = [{"group": str(n), "y": 0, "length": n, "indices": inds}
                for n, inds in ((1, [0, 0, 0]), (2, [0, 0, 1]), (3, [0, 1, 2]))]
        d = describe(rows)
        self.assertEqual(d["mid_equals_onset"], 2)
        self.assertEqual(d["mid_equals_end"], 1)
        self.assertEqual(d["three_distinct_checkpoints"], 1)


if __name__ == "__main__":
    unittest.main()
