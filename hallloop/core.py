"""Numerical processing of longitudinal and transverse Hall-loop measurements."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class AnalysisResult:
    """Aligned raw and corrected traces, with branch selection metadata."""

    field: np.ndarray
    longitudinal_up: np.ndarray
    longitudinal_down: np.ndarray
    transverse_up: np.ndarray
    transverse_down: np.ndarray
    longitudinal_even_up: np.ndarray
    longitudinal_even_down: np.ndarray
    transverse_odd_up: np.ndarray
    transverse_odd_down: np.ndarray
    naive_odd_up: np.ndarray
    naive_odd_down: np.ndarray
    branches: list[dict]
    candidates: list[dict]


@dataclass(frozen=True)
class _Branch:
    id: int
    direction: str
    start: int
    end: int
    field: np.ndarray
    longitudinal: np.ndarray
    transverse: np.ndarray

    def metadata(self):
        count = self.end - self.start + 1
        return {
            "id": self.id,
            "direction": self.direction,
            "start_row": self.start + 1,
            "end_row": self.end + 1,
            "sample_count": count,
            "unique_field_count": self.field.size,
            "duplicate_count": count - self.field.size,
            "field_range": [float(self.field[0]), float(self.field[-1])],
        }


def _finite_vector(values, name):
    message = f"{name} must be a one-dimensional array of finite real values"
    try:
        values = np.asarray(values)
        if values.ndim != 1 or np.iscomplexobj(values):
            raise ValueError(message)
        values = values.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not np.isfinite(values).all():
        raise ValueError(message)
    return values


def _sweep_segments(field, tolerance):
    """Yield inclusive acquisition ranges at sustained reversals."""
    low = high = float(field[0])
    low_row = high_row = 0
    direction = 0
    for row in range(1, field.size):
        value = float(field[row])
        if direction == 0:
            if value <= low:
                low, low_row = value, row
            if value >= high:
                high, high_row = value, row
            if value - low > tolerance:
                start, direction = low_row, 1
            elif high - value > tolerance:
                start, direction = high_row, -1
            else:
                continue
            extreme, first, last = value, row, row
            continue

        advance = direction * (value - extreme)
        if advance > 0:
            extreme, first, last = value, row, row
        elif advance == 0:
            last = row
        elif advance < -tolerance:
            # The first and last observations at a turning field belong to
            # the incoming and outgoing sweeps, respectively.
            yield start, first, direction
            start, direction = last, -direction
            extreme, first, last = value, row, row

    if direction:
        yield start, first, direction


def _make_branch(branch_id, segment, field, longitudinal, transverse):
    start, end, direction = segment
    rows = slice(start, end + 1)
    knots, groups, counts = np.unique(
        field[rows], return_inverse=True, return_counts=True
    )
    # Weight each sample before summing so the sum stays at the scale of the mean.
    weights = counts[groups]
    longitudinal = np.bincount(groups, weights=longitudinal[rows] / weights)
    transverse = np.bincount(groups, weights=transverse[rows] / weights)
    return _Branch(
        branch_id, "up" if direction > 0 else "down", start, end,
        knots, longitudinal, transverse,
    )


def _interp_weights(grid, knots):
    """Return bracketing indices and linear weights for an in-range grid."""
    upper = np.clip(np.searchsorted(knots, grid, side="right"), 1, knots.size - 1)
    lower = upper - 1
    left, right = knots[lower], knots[upper]
    with np.errstate(over="ignore"):
        width = right - left
    wide = np.isinf(width)
    weights = np.empty_like(grid)
    weights[~wide] = (grid[~wide] - left[~wide]) / width[~wide]
    weights[wide] = (grid[wide] / 2 - left[wide] / 2) / (
        right[wide] / 2 - left[wide] / 2
    )
    return lower, upper, weights


def _aligned_signals(grid, branch):
    lower, upper, weights = _interp_weights(grid, branch.field)
    return (
        (1 - weights) * branch.longitudinal[lower] + weights * branch.longitudinal[upper],
        (1 - weights) * branch.transverse[lower] + weights * branch.transverse[upper],
    )


def analyze(field, longitudinal, transverse, *, field_tolerance=0.0, pair=None):
    """Process one complete Hall loop without extrapolation.

    Inputs are equal-length real vectors in acquisition order. Preserve their
    field and signal units. A reversal must move more than `field_tolerance`
    from the running extremum. Adjacent opposite sweeps qualify if each reaches
    both extrema of the input within this tolerance and their common support
    spans both signs of field. This conservative check may reject truncated
    loops and cannot validate magnetic history. Select among multiple
    qualifying pairs by passing the reported 1-based `pair` index.

    A turning dwell supplies its first extremum observation to the incoming
    sweep and its last to the outgoing sweep. An isolated extremum is shared.
    Duplicate field values within a sweep are averaged. The output grid
    consists of measured field knots and their reflections within the common
    symmetric range. Even and odd components pair opposite sweeps at opposite
    fields. The same-sweep odd components provide a diagnostic comparison.
    Metadata row ranges are inclusive, 1-based indices into the input arrays.
    """
    field = _finite_vector(field, "field")
    longitudinal = _finite_vector(longitudinal, "longitudinal")
    transverse = _finite_vector(transverse, "transverse")
    if field.size != longitudinal.size or field.size != transverse.size:
        raise ValueError("field, longitudinal, and transverse must have the same length")
    if field.size < 3:
        raise ValueError("At least three input data records are required for a sweep pair")
    try:
        if np.ndim(field_tolerance) != 0:
            raise ValueError
        tolerance = float(field_tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("field_tolerance must be a finite nonnegative scalar") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("field_tolerance must be a finite nonnegative scalar")
    if pair is not None and (
        not isinstance(pair, Integral) or isinstance(pair, bool) or pair < 1
    ):
        raise ValueError("pair must be a positive 1-based integer")

    minimum, maximum = float(field.min()), float(field.max())
    if not minimum < 0 < maximum:
        raise ValueError("A full sweep must include both positive and negative fields")

    branches = [
        _make_branch(branch_id, segment, field, longitudinal, transverse)
        for branch_id, segment in enumerate(_sweep_segments(field, tolerance), start=1)
    ]
    matches = []
    candidates = []
    for first, second in zip(branches, branches[1:]):
        # The segment detector alternates directions. Keeping adjacency here
        # prevents bridging across an intervening minor or incomplete sweep.
        low = max(float(first.field[0]), float(second.field[0]))
        high = min(float(first.field[-1]), float(second.field[-1]))
        limit = min(-low, high)
        if low - minimum > tolerance or maximum - high > tolerance or limit <= 0:
            continue
        matches.append((first, second))
        candidates.append({
            "id": len(matches),
            "selected": False,
            "branch_ids": [first.id, second.id],
            "branches": [first.metadata(), second.metadata()],
            "measured_overlap": [low, high],
            "common_field_range": [-limit, limit],
        })

    if not candidates:
        raise ValueError(
            "No candidate pairs. Expected adjacent opposite sweeps reaching both "
            "input field extrema within field_tolerance, with overlap on both sides of zero."
        )
    if (pair is None and len(candidates) != 1) or (pair is not None and pair > len(candidates)):
        choices = ", ".join(
            f"{candidate['id']} ({first.direction} rows {first.start + 1}-{first.end + 1}, "
            f"{second.direction} rows {second.start + 1}-{second.end + 1})"
            for candidate, (first, second) in zip(candidates, matches)
        )
        reason = "Multiple candidate pairs" if pair is None else f"Pair {pair} is unavailable"
        raise ValueError(f"{reason}. Select pair by its 1-based index. Candidates: {choices}")

    index = 0 if pair is None else int(pair) - 1
    selected = candidates[index]
    selected["selected"] = True
    up, down = sorted(matches[index], key=lambda branch: branch.direction == "down")
    lo, hi = selected["common_field_range"]
    knots = np.concatenate((up.field, down.field))
    knots = knots[(knots >= lo) & (knots <= hi)]
    grid = np.unique(np.concatenate(([lo, hi], knots, -knots)))

    long_up, trans_up = _aligned_signals(grid, up)
    long_down, trans_down = _aligned_signals(grid, down)
    long_even_up = long_up / 2 + long_down[::-1] / 2
    trans_odd_up = trans_up / 2 - trans_down[::-1] / 2
    return AnalysisResult(
        field=grid,
        longitudinal_up=long_up,
        longitudinal_down=long_down,
        transverse_up=trans_up,
        transverse_down=trans_down,
        longitudinal_even_up=long_even_up,
        longitudinal_even_down=long_even_up[::-1].copy(),
        transverse_odd_up=trans_odd_up,
        transverse_odd_down=-trans_odd_up[::-1],
        naive_odd_up=trans_up / 2 - trans_up[::-1] / 2,
        naive_odd_down=trans_down / 2 - trans_down[::-1] / 2,
        branches=selected["branches"],
        candidates=candidates,
    )
