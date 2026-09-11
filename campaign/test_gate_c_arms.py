"""End-to-end leakage tests for the Gate C feature-extraction path.

campaign/test_gatec.py pins the prefix-logprob helper in isolation, which is not
the same as proving the pipeline that feeds the classifier is clean. These tests
exercise score_gate_c.arms() directly and assert the structural property the whole
gate rests on: a prefix arm must be a function of the prefix checkpoint alone.

If a future edit reintroduces a whole-answer feature into a prefix arm, the way
the sidecar did, the first test here fails.
"""

import unittest

from campaign.score_gate_c import FULL_LP, PFX_LP, WS_SCALARS, arms

WS = {name: 0.5 for name in WS_SCALARS}


def checkpoint(frac, token_index, base):
    row = dict(WS)
    row.update({
        "frac": frac,
        "token_index": token_index,
        "pfxlp_first": base,
        "pfxlp_last": base - 0.1,
        "pfxlp_mean": base - 0.2,
        "pfxlp_min": base - 0.3,
        "pfxlp_tokens_so_far": float(token_index + 1),
    })
    return row


def synthetic_row():
    return {
        "source_dataset": "synthetic",
        "token_count": 40,
        "logprob_features": {
            "bl_first_token_logprob": -0.1,
            "bl_mean_logprob": -0.4,
            "bl_min_logprob": -2.0,
            "bl_answer_len": 40.0,
        },
        "prefix_workspace_features": [
            checkpoint(0.0, 0, -0.1),
            checkpoint(0.5, 19, -0.5),
            checkpoint(1.0, 39, -0.9),
        ],
        "fixed_checkpoint_features": [
            checkpoint(None, 8, -0.3),
            checkpoint(None, 16, -0.45),
        ],
    }


class TestPrefixArmsDoNotSeeTheFuture(unittest.TestCase):
    def test_whole_answer_features_do_not_reach_prefix_arms(self) -> None:
        base = arms(synthetic_row())
        self.assertTrue(base, "fixture should produce arms")

        tampered_row = synthetic_row()
        # Rewrite every whole-answer quantity, including the final answer length.
        tampered_row["logprob_features"] = {
            "bl_first_token_logprob": -99.0,
            "bl_mean_logprob": -99.0,
            "bl_min_logprob": -99.0,
            "bl_answer_len": 9999.0,
        }
        tampered = arms(tampered_row)

        for arm in ("prefix_lp", "prefix_lp_ws", "prefix_plus_lp",
                    "prefix_plus_lp_ws", "fixed8_lp", "fixed8_lp_ws",
                    "fixed16_lp", "fixed16_lp_ws"):
            self.assertEqual(base[arm], tampered[arm],
                             f"{arm} changed when only whole-answer features moved")

    def test_the_end_checkpoint_does_not_reach_prefix_arms(self) -> None:
        base = arms(synthetic_row())
        tampered_row = synthetic_row()
        for entry in tampered_row["prefix_workspace_features"]:
            if entry["frac"] == 1.0:
                for name in WS_SCALARS:
                    entry[name] = -99.0
        tampered = arms(tampered_row)
        self.assertEqual(base["prefix_lp_ws"], tampered["prefix_lp_ws"])
        self.assertEqual(base["fixed8_lp_ws"], tampered["fixed8_lp_ws"])

    def test_a_later_fixed_checkpoint_does_not_reach_an_earlier_one(self) -> None:
        base = arms(synthetic_row())
        tampered_row = synthetic_row()
        for entry in tampered_row["fixed_checkpoint_features"]:
            if entry["token_index"] == 16:
                for name in WS_SCALARS:
                    entry[name] = -99.0
        tampered = arms(tampered_row)
        self.assertEqual(base["fixed8_lp_ws"], tampered["fixed8_lp_ws"])
        self.assertNotEqual(base["fixed16_lp_ws"], tampered["fixed16_lp_ws"])


class TestArmsAreActuallySensitive(unittest.TestCase):
    """Invariance is only meaningful if the arms respond to their own inputs."""

    def test_prefix_arm_tracks_its_own_checkpoint(self) -> None:
        base = arms(synthetic_row())
        tampered_row = synthetic_row()
        for entry in tampered_row["prefix_workspace_features"]:
            if entry["frac"] == 0.5:
                entry["pfxlp_mean"] = -42.0
        self.assertNotEqual(base["prefix_lp"], arms(tampered_row)["prefix_lp"])

    def test_full_arm_tracks_whole_answer_features(self) -> None:
        base = arms(synthetic_row())
        tampered_row = synthetic_row()
        tampered_row["logprob_features"]["bl_mean_logprob"] = -42.0
        self.assertNotEqual(base["full_lp"], arms(tampered_row)["full_lp"])


class TestBaselinesAreFeatureMatched(unittest.TestCase):
    def test_prefix_and_full_logprob_baselines_have_equal_width(self) -> None:
        self.assertEqual(len(PFX_LP), len(FULL_LP))
        a = arms(synthetic_row())
        self.assertEqual(len(a["prefix_lp"]), len(a["full_lp"]))

    def test_workspace_arms_add_the_same_number_of_features(self) -> None:
        a = arms(synthetic_row())
        self.assertEqual(len(a["prefix_lp_ws"]) - len(a["prefix_lp"]), len(WS_SCALARS))
        self.assertEqual(len(a["full_lp_ws"]) - len(a["full_lp"]), len(WS_SCALARS))

    def test_the_extra_last_token_feature_is_isolated_to_the_plus_arm(self) -> None:
        a = arms(synthetic_row())
        self.assertEqual(len(a["prefix_plus_lp"]), len(a["prefix_lp"]) + 1)


class TestIncompleteRowsAreRefused(unittest.TestCase):
    def test_a_missing_midpoint_yields_no_arms(self) -> None:
        row = synthetic_row()
        row["prefix_workspace_features"] = [
            e for e in row["prefix_workspace_features"] if e["frac"] != 0.5
        ]
        self.assertEqual(arms(row), {})

    def test_missing_whole_answer_features_yield_no_arms(self) -> None:
        row = synthetic_row()
        del row["logprob_features"]["bl_answer_len"]
        self.assertEqual(arms(row), {})

    def test_absent_fixed_checkpoints_are_omitted_not_invented(self) -> None:
        row = synthetic_row()
        row["fixed_checkpoint_features"] = []
        a = arms(row)
        self.assertNotIn("fixed8_lp", a)
        self.assertIn("prefix_lp", a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
