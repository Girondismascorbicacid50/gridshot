"""Visual pairing and legacy outline-overlap hints for the ZIP flow.

Outline IoU does not establish tool identity and remains review-only. The
visual matcher compares foreground image features, verifies them with robust
epipolar geometry, and only prefills pairs that clear a strict, versioned gate.
Users still review/split those pairs before anything is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contour import to_shapely
from .feature_match import MatchEvidence, load_match_image, prepare_sift, sift_magsac
from .models import Poly


VISUAL_GATE_VERSION = "tools-v0"


def select_global_pairs(
    edges: list[dict],
    image_count: int,
    *,
    min_score: float = 0.0,
) -> dict:
    """Maximum-weight general matching with an explicit unmatched option.

    This is benchmark plumbing, not a production acceptance policy. Edges still
    need corpus-calibrated gates before this assignment may prefill review.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if image_count < 0:
        raise ValueError("image count must be non-negative")
    normalized = []
    rejected = []
    for edge in edges:
        a, b = int(edge["a"]), int(edge["b"])
        score = float(edge["score"])
        if not (0 <= a < image_count and 0 <= b < image_count) or a == b:
            raise ValueError(f"invalid matching edge ({a}, {b})")
        item = {**edge, "a": min(a, b), "b": max(a, b), "score": score}
        if not np.isfinite(score) or score <= min_score:
            rejected.append({**item, "reason": "below_min_score"})
        else:
            normalized.append(item)

    normalized.sort(key=lambda edge: (edge["a"], edge["b"]))
    if not normalized:
        return {
            "pairs": [],
            "singles": list(range(image_count)),
            "total_score": 0.0,
            "edges": sorted(rejected, key=lambda edge: (edge["a"], edge["b"])),
        }

    rows = []
    cols = []
    values = []
    for column, edge in enumerate(normalized):
        for vertex in (edge["a"], edge["b"]):
            rows.append(vertex)
            cols.append(column)
            values.append(1.0)
    incidence = coo_matrix(
        (values, (rows, cols)), shape=(image_count, len(normalized))
    ).tocsr()
    # A tiny stable preference makes equal-score solutions deterministic without
    # changing any meaningful matcher score.
    tie_break = np.linspace(1e-9, 1e-12, len(normalized))
    objective = -(
        np.asarray([edge["score"] for edge in normalized], dtype=np.float64)
        + tie_break
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(normalized), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(incidence, 0.0, 1.0),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"global pair assignment failed: {result.message}")

    selected_columns = {
        index for index, value in enumerate(result.x) if value >= 0.5
    }
    pairs = []
    edge_results = list(rejected)
    used = set()
    for index, edge in enumerate(normalized):
        if index in selected_columns:
            selected = {**edge, "reason": "selected_global"}
            pairs.append(selected)
            edge_results.append(selected)
            used.update((edge["a"], edge["b"]))
        else:
            edge_results.append({**edge, "reason": "global_conflict_or_unmatched"})
    pairs.sort(key=lambda edge: (edge["a"], edge["b"]))
    edge_results.sort(key=lambda edge: (edge["a"], edge["b"]))
    return {
        "pairs": pairs,
        "singles": [index for index in range(image_count) if index not in used],
        "total_score": round(sum(edge["score"] for edge in pairs), 9),
        "edges": edge_results,
    }


@dataclass(frozen=True)
class VisualPairGate:
    """Conservative review-prefill gate derived from the first golden ZIP.

    The five true edges in that corpus had score >= 19.09 and >= 44 inliers;
    every false edge had score <= 4.17 and <= 13 inliers. These deliberately
    conservative thresholds sit well inside that observed gap. They are not a
    population-calibrated probability and must be revalidated as the corpus
    grows.
    """

    min_score: float = 12.0
    min_inliers: int = 30
    min_inlier_ratio: float = 0.55
    min_second_best_ratio: float = 2.5


def _passes_absolute_gate(evidence: MatchEvidence, gate: VisualPairGate) -> bool:
    return (
        evidence.score >= gate.min_score
        and evidence.inliers >= gate.min_inliers
        and evidence.inlier_ratio >= gate.min_inlier_ratio
    )


def select_visual_pairs(
    edges: list[dict],
    image_count: int,
    gate: VisualPairGate = VisualPairGate(),
) -> dict:
    """Select disjoint mutual-best edges with a clear second-best margin."""
    ranked: dict[int, list[dict]] = {idx: [] for idx in range(image_count)}
    for edge in edges:
        ranked[edge["a"]].append(edge)
        ranked[edge["b"]].append(edge)
    for candidates in ranked.values():
        candidates.sort(key=lambda item: item["evidence"].score, reverse=True)

    accepted: list[dict] = []
    for edge in sorted(edges, key=lambda item: item["evidence"].score, reverse=True):
        a, b = edge["a"], edge["b"]
        evidence = edge["evidence"]
        if not _passes_absolute_gate(evidence, gate):
            continue
        if not ranked[a] or ranked[a][0] is not edge:
            continue
        if not ranked[b] or ranked[b][0] is not edge:
            continue

        alternatives_a = [item for item in ranked[a] if item is not edge]
        alternatives_b = [item for item in ranked[b] if item is not edge]
        second_a = alternatives_a[0]["evidence"].score if alternatives_a else 0.0
        second_b = alternatives_b[0]["evidence"].score if alternatives_b else 0.0
        if evidence.score < gate.min_second_best_ratio * max(second_a, second_b):
            continue
        accepted.append(edge)

    used = {idx for edge in accepted for idx in (edge["a"], edge["b"])}
    return {
        "pairs": accepted,
        "singles": [idx for idx in range(image_count) if idx not in used],
        "gate": {
            "version": VISUAL_GATE_VERSION,
            "min_score": gate.min_score,
            "min_inliers": gate.min_inliers,
            "min_inlier_ratio": gate.min_inlier_ratio,
            "min_second_best_ratio": gate.min_second_best_ratio,
        },
    }


def visual_pair_images(
    images_and_masks: list[tuple[np.ndarray, np.ndarray]],
    gate: VisualPairGate = VisualPairGate(),
) -> dict:
    """Verify all image pairs using masked SIFT correspondences + MAGSAC."""
    features = [prepare_sift(image, mask) for image, mask in images_and_masks]
    edges: list[dict] = []
    for a in range(len(features)):
        for b in range(a + 1, len(features)):
            edges.append({
                "a": a,
                "b": b,
                "evidence": sift_magsac(features[a], features[b]),
            })
    result = select_visual_pairs(edges, len(features), gate)
    result["edges"] = edges
    return result


def visual_pair_files(
    image_and_mask_paths: list[tuple[Path, Path]],
    gate: VisualPairGate = VisualPairGate(),
) -> dict:
    """Load replayable matcher artifacts and run the visual pair verifier."""
    return visual_pair_images([load_match_image(*paths) for paths in image_and_mask_paths], gate)


def pair_images(
    outlines: list[Poly], high: float = 0.55, low: float = 0.3
) -> dict:
    """Classify high- and borderline-IoU review candidates.

    Returns {"pairs": [(i, j, iou)], "flagged": [(i, j, iou)], "singles": [i]}.
    `high` separates strong hints from borderline hints; neither is identity evidence.
    `low`..`high` is the flag band;
    below `low` two images are never considered the same tool.
    """
    shapes = [to_shapely(o).buffer(0) for o in outlines]
    cand: list[tuple[float, int, int]] = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            union = shapes[i].union(shapes[j]).area
            iou = shapes[i].intersection(shapes[j]).area / union if union > 0 else 0.0
            if iou >= low:
                cand.append((iou, i, j))
    cand.sort(reverse=True)  # take the strongest matches first

    used: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    flagged: list[tuple[int, int, float]] = []
    for iou, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        (pairs if iou >= high else flagged).append((i, j, round(iou, 3)))
    singles = [i for i in range(len(shapes)) if i not in used]
    return {"pairs": pairs, "flagged": flagged, "singles": singles}
