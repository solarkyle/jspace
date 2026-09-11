import unittest

from sidecar.conformance import compare_captures, jensen_shannon
from sidecar.experiments import answer_matches_expected, binding_prompt, normalize_token_family


class ExperimentUtilityTests(unittest.TestCase):
    def test_token_family_normalization_is_conservative(self) -> None:
        self.assertEqual(normalize_token_family(" Euro"), "euro")
        self.assertEqual(normalize_token_family("\u2581EURO"), "euro")
        self.assertNotEqual(normalize_token_family("euro"), normalize_token_family("currency"))

    def test_binding_prompt_is_deterministic(self) -> None:
        prompt, target, expected = binding_prompt(4)
        self.assertIn("Ada = amber", prompt)
        self.assertIn(f"bound to {target}", prompt)
        self.assertEqual((target, expected), ("Dax", "dune"))

    def test_answer_matching_tolerates_exact_repetition_only(self) -> None:
        self.assertTrue(answer_matches_expected("dune", "dune"))
        self.assertTrue(answer_matches_expected("amberamber", "amber"))
        self.assertFalse(answer_matches_expected("ambergris", "amber"))

    def test_sparse_jsd_identity(self) -> None:
        self.assertAlmostEqual(jensen_shannon({1: 0.7, "<tail>": 0.3}, {1: 0.7, "<tail>": 0.3}), 0.0)

    def test_conformance_comparison(self) -> None:
        # input_token_ids is now part of the capture contract: a comparison that
        # cannot confirm both runs saw the same tokens is "incomplete", not a pass.
        capture = {
            "input_token_ids": [1, 2],
            "layers": [
                {"layer": 2, "residual": [1.0, 0.0], "top_ids": [1, 2], "top_probs": [0.7, 0.2]}
            ],
            "final": {"next_token_id": 1},
        }
        result = compare_captures(capture, capture)
        self.assertTrue(result["passed"])
        self.assertEqual(result["aligned_layers"], 1)


if __name__ == "__main__":
    unittest.main()
