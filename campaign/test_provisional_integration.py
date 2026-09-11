"""Integration test: provisional status must survive in the SAVED result.

campaign/test_uniform_scenarios.py asserts that the scorer's source no longer
contains a bounds guarantee, which is a grep, not a behaviour check. This runs the
real entry point over a synthetic fixture and reads the JSON it writes.

The case that matters is the one that previously went wrong: the labelled subset
decides cleanly, and a row is still unadjudicated. The subset verdict must survive
as a measurement, and the population conclusion must stay provisional anyway.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib

from campaign.score_gate_c import FULL_LP, PFX_LP, WS_SCALARS

SOURCES = ("alpha", "beta")
PER_SOURCE = 60


def _noise(seed):
    """Deterministic pseudo-uniform in [0, 1). Reproducible across runs."""
    x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    x = (x ^ (x >> 13)) * 2654435761 & 0x7FFFFFFF
    return x / 0x7FFFFFFF


def _checkpoint(value, token_index, frac=None):
    """A checkpoint whose workspace scalars all carry `value`."""
    row = {name: value for name in WS_SCALARS}
    row.update({
        "token_index": token_index,
        "pfxlp_first": -value,
        "pfxlp_last": -value - 0.02,
        "pfxlp_mean": -value - 0.04,
        "pfxlp_min": -value - 0.50,
        "pfxlp_tokens_so_far": float(token_index + 1),
    })
    if frac is not None:
        row["frac"] = frac
    return row


def _row(source, index, is_error, resolved=True):
    """Workspace features are strongly informative, logprob features weakly.

    Both are noisy, so neither arm separates perfectly. That ordering is what
    makes the full-answer increment positive, which the registered ratio needs as
    a denominator, and it mirrors the real data.
    """
    signal = 1.0 if is_error else 0.0
    # zlib.crc32, not hash(): the builtin is salted per process unless
    # PYTHONHASHSEED is pinned, which would make this fixture differ run to run.
    seed = zlib.crc32(source.encode()) % 65536 * 10007 + index
    # strong but imperfect: label moves it 0.70, noise moves it 0.30
    ws = 0.70 * signal + 0.30 * _noise(seed)
    # weak: label moves it 0.20, noise moves it 0.80
    lp = 0.20 * signal + 0.80 * _noise(seed + 7919)
    row = {
        "example_id": f"{source}-{index}",
        "source_dataset": source,
        "split_group": f"{source}-{index % 7}",
        "token_count": 40,
        "answer": "synthetic",
        "logprob_features": {
            "bl_first_token_logprob": -lp,
            "bl_mean_logprob": -lp - 0.04,
            "bl_min_logprob": -lp - 0.50,
            "bl_answer_len": 40.0 + 4.0 * _noise(seed + 104729),
        },
        "prefix_workspace_features": [
            _checkpoint(ws, 0, frac=0.0),
            _checkpoint(ws, 19, frac=0.5),
            _checkpoint(ws, 39, frac=1.0),
        ],
        "fixed_checkpoint_features": [
            _checkpoint(ws, 8),
            _checkpoint(ws, 16),
        ],
    }
    assert all(k in row["logprob_features"] for k in FULL_LP)
    assert all(k in row["prefix_workspace_features"][1] for k in PFX_LP)
    row["deterministic_grade"] = (
        {"method": "alias", "correct": not is_error, "abstained": False}
        if resolved
        else {"method": "needs_judge", "correct": None, "abstained": False})
    return row


def build_fixture(path, unresolved):
    """Informative-but-noisy rows per source, plus `unresolved` unlabelled rows."""
    with io.open(path, "w", encoding="utf-8") as fh:
        for source in SOURCES:
            for index in range(PER_SOURCE):
                # deterministic, roughly 40 percent errors, both classes well above
                # the scorer's minority floor
                is_error = (index % 5) in (0, 1)
                fh.write(json.dumps(_row(source, index, is_error)) + "\n")
        for index in range(unresolved):
            fh.write(json.dumps(
                _row(SOURCES[0], 1000 + index, index % 2 == 0, resolved=False)) + "\n")


def run_scorer(unresolved):
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "graded.jsonl")
        out = os.path.join(tmp, "gate_c.json")
        build_fixture(src, unresolved)
        proc = subprocess.run(
            [sys.executable, "-m", "campaign.score_gate_c", "--input", src, "--out", out],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if proc.returncode != 0:
            raise AssertionError(f"scorer failed:\n{proc.stdout}\n{proc.stderr}")
        with io.open(out, encoding="utf-8") as fh:
            return json.load(fh), proc.stdout


class TestProvisionalStatusInSavedResult(unittest.TestCase):
    def test_one_unresolved_row_makes_the_conclusion_provisional(self) -> None:
        payload, stdout = run_scorer(unresolved=1)

        self.assertEqual(payload["full_population_conclusion"],
                         "PROVISIONAL_UNADJUDICATED")
        self.assertEqual(payload["n_unresolved_kept_for_sensitivity"], 1)
        self.assertEqual(payload["dropped"].get("unlabeled_needs_judge"), 1)
        self.assertEqual(payload["excluded_unresolved_by_source"], {"alpha": 1})

        for model_name, block in payload["models"].items():
            with self.subTest(model=model_name):
                # the subset still decides, and that measurement is preserved
                self.assertIn(block["labelled_subset_verdict"], {"HIT", "MISS"},
                              "fixture should give a decidable subset verdict")
                self.assertEqual(block["registered_primary"]["scope"],
                                 "labelled_subset_only")
                # but the population claim does not inherit it
                self.assertEqual(block["full_population_conclusion"],
                                 "PROVISIONAL_UNADJUDICATED")

        self.assertIn("PROVISIONAL_UNADJUDICATED", stdout)
        self.assertIn("remain unadjudicated", stdout)

    def test_no_unresolved_rows_allows_a_final_conclusion(self) -> None:
        payload, _ = run_scorer(unresolved=0)
        self.assertEqual(payload["full_population_conclusion"], "FINAL")
        self.assertEqual(payload.get("n_unresolved_kept_for_sensitivity"), 0)
        for model_name, block in payload["models"].items():
            with self.subTest(model=model_name):
                self.assertEqual(block["full_population_conclusion"],
                                 block["labelled_subset_verdict"])
                self.assertNotIn("unresolved_sensitivity", block)

    def test_scenarios_are_reported_without_claiming_bounds(self) -> None:
        payload, stdout = run_scorer(unresolved=4)
        for block in payload["models"].values():
            sens = block["unresolved_sensitivity"]
            self.assertIn("uniform_scenarios_agree", sens)
            self.assertNotIn("verdict_stable_under_every_assignment", sens)
            self.assertIn("NOT BOUNDS", sens["note"])
        self.assertIn("NOT bounds", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
