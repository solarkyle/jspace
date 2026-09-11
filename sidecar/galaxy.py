from __future__ import annotations

import math
from typing import Any

import numpy as np


def trajectory_layout(
    token_ids: list[int],
    trajectories: np.ndarray,
    *,
    neighbors: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project per-token J-Lens trajectories into a deterministic 2D trace map.

    The coordinates describe similarity between activation trajectories in this
    trace. They are not residual-stream coordinates and the returned edges are
    nearest-neighbor display links, not causal claims.
    """
    count = len(token_ids)
    if count == 0:
        return [], []

    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != count:
        raise ValueError("trajectories must have one row per token id")

    # Log compression keeps one very confident layer from flattening the map.
    values = np.log1p(np.clip(values, 0.0, None) * 10_000.0)
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(row_norms, 1e-12)
    centered = values - values.mean(axis=0, keepdims=True)

    if count == 1 or not np.any(np.abs(centered) > 1e-12):
        coords = _fallback_ring(token_ids)
    else:
        u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
        dims = min(2, u.shape[1], singular.shape[0])
        coords = np.zeros((count, 2), dtype=np.float64)
        coords[:, :dims] = u[:, :dims] * singular[:dims]
        coords = _stabilize_signs(coords)
        if np.ptp(coords[:, 1]) < 1e-9:
            coords[:, 1] = _fallback_ring(token_ids)[:, 1]

    coords = _normalize_coords(coords, token_ids)
    center = coords.mean(axis=0)
    nodes: list[dict[str, Any]] = []
    for index, (token_id, point) in enumerate(zip(token_ids, coords)):
        angle = (math.atan2(point[1] - center[1], point[0] - center[0]) + math.tau) % math.tau
        nodes.append(
            {
                "id": int(token_id),
                "x": round(float(point[0]), 6),
                "y": round(float(point[1]), 6),
                "sector": int(angle / math.tau * 6) % 6,
                "layout_index": index,
            }
        )

    edges = _nearest_neighbor_edges(token_ids, coords, neighbors=max(0, neighbors))
    return nodes, edges


def _stabilize_signs(coords: np.ndarray) -> np.ndarray:
    result = coords.copy()
    for axis in range(result.shape[1]):
        column = result[:, axis]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            result[:, axis] *= -1.0
    return result


def _fallback_ring(token_ids: list[int]) -> np.ndarray:
    count = len(token_ids)
    coords = np.zeros((count, 2), dtype=np.float64)
    for index, token_id in enumerate(token_ids):
        # Token-id phase avoids every degenerate trace getting the same rotation.
        phase = ((int(token_id) * 0.61803398875) % 1.0) * math.tau
        angle = math.tau * index / max(1, count) + phase / max(3, count)
        radius = 0.34 + 0.08 * ((int(token_id) * 2654435761 % 997) / 997.0)
        coords[index] = [0.5 + math.cos(angle) * radius, 0.5 + math.sin(angle) * radius]
    return coords


def _normalize_coords(coords: np.ndarray, token_ids: list[int]) -> np.ndarray:
    result = coords.copy()
    for axis in range(2):
        lo = float(result[:, axis].min())
        hi = float(result[:, axis].max())
        if hi - lo < 1e-9:
            result[:, axis] = 0.5
        else:
            result[:, axis] = 0.1 + 0.8 * (result[:, axis] - lo) / (hi - lo)

    # Separate exact overlaps without introducing run-to-run randomness.
    for index, token_id in enumerate(token_ids):
        angle = ((int(token_id) * 0.754877666) % 1.0) * math.tau
        result[index, 0] = np.clip(result[index, 0] + math.cos(angle) * 0.004, 0.08, 0.92)
        result[index, 1] = np.clip(result[index, 1] + math.sin(angle) * 0.004, 0.08, 0.92)
    return result


def _nearest_neighbor_edges(
    token_ids: list[int], coords: np.ndarray, neighbors: int
) -> list[dict[str, Any]]:
    if len(token_ids) < 2 or neighbors == 0:
        return []
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, Any]] = []
    for source_index, source_id in enumerate(token_ids):
        nearest = np.argsort(distances[source_index])[: min(neighbors, len(token_ids) - 1)]
        for target_index in nearest.tolist():
            target_id = token_ids[target_index]
            key = tuple(sorted((int(source_id), int(target_id))))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source": int(source_id),
                    "target": int(target_id),
                    "distance": round(float(distances[source_index, target_index]), 6),
                    "kind": "trajectory_neighbor",
                }
            )
    return edges
