import unittest

import numpy as np

from sidecar.galaxy import trajectory_layout


class TrajectoryLayoutTests(unittest.TestCase):
    def test_layout_is_deterministic_and_bounded(self) -> None:
        token_ids = [11, 22, 33, 44]
        trajectories = np.array(
            [
                [0.8, 0.4, 0.1, 0.0],
                [0.7, 0.5, 0.1, 0.0],
                [0.0, 0.1, 0.6, 0.9],
                [0.0, 0.2, 0.5, 0.8],
            ]
        )

        first = trajectory_layout(token_ids, trajectories)
        second = trajectory_layout(token_ids, trajectories)

        self.assertEqual(first, second)
        nodes, edges = first
        self.assertEqual([node["id"] for node in nodes], token_ids)
        self.assertTrue(all(0.0 <= node["x"] <= 1.0 for node in nodes))
        self.assertTrue(all(0.0 <= node["y"] <= 1.0 for node in nodes))
        self.assertTrue(edges)
        self.assertTrue(all(edge["kind"] == "trajectory_neighbor" for edge in edges))

    def test_degenerate_trace_uses_stable_fallback(self) -> None:
        nodes, edges = trajectory_layout([7, 8, 9], np.zeros((3, 5)))

        self.assertEqual(len(nodes), 3)
        self.assertTrue(edges)
        self.assertEqual(len({(node["x"], node["y"]) for node in nodes}), 3)


if __name__ == "__main__":
    unittest.main()
