"""Regenerate the synthetic Hall-loop examples next to this script."""

import csv
import math
from pathlib import Path


HALL_SLOPE = 1.8  # ohm per tesla
LONGITUDINAL_PICKUP = 0.12  # ohm per ohm
COERCIVE_FIELD = 0.31  # tesla
PEAK_WIDTH = 0.13  # tesla, Gaussian standard deviation
LOW_FIELD_INDEX = -120
HIGH_FIELD_INDEX = 95
HEADER = ("field_T", "longitudinal_ohm", "transverse_ohm")


def sample(field_index, direction):
    field = field_index / 100
    baseline = 11.0 + 1.9 * field**2
    peak = math.exp(-0.5 * ((field - direction * COERCIVE_FIELD) / PEAK_WIDTH) ** 2)
    # Fade both peaks to zero at saturation so the turn is continuous.
    envelope = max(0.0, 1 - (field / (HIGH_FIELD_INDEX / 100)) ** 2) ** 2
    longitudinal = baseline + 24.0 * envelope * peak
    transverse = HALL_SLOPE * field + LONGITUDINAL_PICKUP * longitudinal
    return tuple(f"{value:.10f}" for value in (field, longitudinal, transverse))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def main():
    directory = Path(__file__).resolve().parent
    up = [sample(index, 1) for index in range(LOW_FIELD_INDEX, HIGH_FIELD_INDEX + 1)]
    down = [sample(index, -1) for index in range(HIGH_FIELD_INDEX - 1, LOW_FIELD_INDEX - 1, -1)]
    write_csv(directory / "mixed_loop.csv", up + down)
    write_csv(directory / "incomplete_sweep.csv", up)


if __name__ == "__main__":
    main()
