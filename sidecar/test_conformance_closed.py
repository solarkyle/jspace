"""The conformance gate must fail closed.

Every pair below is invalid or incomplete, and every one of them used to return
passed=true. Missing evidence is not agreement.
"""

import unittest
from copy import deepcopy

from sidecar.conformance import compare_captures


def reference() -> dict:
    return {
        "input_token_ids": [1, 2],
        "layers": [
            {"layer": i, "residual": [1.0, 0.0], "top_ids": [1, 2], "top_probs": [0.7, 0.2]}
            for i in (2, 3)
        ],
        "final": {"next_token_id": 1},
    }


class TestFailsClosed(unittest.TestCase):
    def test_identical_captures_still_pass(self) -> None:
        result = compare_captures(reference(), reference())
        self.assertTrue(result["passed"], result["problems"])
        self.assertEqual(result["status"], "pass")

    def test_missing_required_layer_is_not_a_pass(self) -> None:
        candidate = reference()
        candidate["layers"].pop()
        result = compare_captures(reference(), candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_candidate_layers"], [3])

    def test_missing_residuals_is_not_a_pass(self) -> None:
        candidate = reference()
        for layer in candidate["layers"]:
            layer.pop("residual")
        result = compare_captures(reference(), candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "incomplete")

    def test_both_missing_final_output_is_not_a_pass(self) -> None:
        ref = reference()
        ref.pop("final")
        result = compare_captures(ref, deepcopy(ref))
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "incomplete")

    def test_rescaled_residuals_are_not_a_pass(self) -> None:
        candidate = reference()
        for layer in candidate["layers"]:
            layer["residual"] = [10.0, 0.0]
        result = compare_captures(reference(), candidate)
        # cosine is 1.0 here: direction is identical, only the scale moved.
        self.assertAlmostEqual(result["layers"][0]["residual_cosine"], 1.0, places=6)
        self.assertFalse(result["passed"])

    def test_missing_input_token_ids_is_not_a_pass(self) -> None:
        candidate = reference()
        candidate.pop("input_token_ids")
        result = compare_captures(reference(), candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "incomplete")

    def test_mismatched_input_token_ids_is_not_a_pass(self) -> None:
        candidate = reference()
        candidate["input_token_ids"] = [1, 3]
        self.assertFalse(compare_captures(reference(), candidate)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
