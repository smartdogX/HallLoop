"""Read acquisition-order CSV and tab-delimited measurement tables."""

import csv
from dataclasses import dataclass
import io

import numpy as np


@dataclass(frozen=True)
class Table:
    field: np.ndarray
    longitudinal: np.ndarray
    transverse: np.ndarray
    delimiter: str
    encoding: str

    @property
    def signals(self):
        return self.field, self.longitudinal, self.transverse


DELIMITERS = {"comma": ",", "tab": "\t"}


def _decode(data: bytes) -> tuple[str, str]:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return data.decode("cp1252"), "cp1252"


def read_table(data: bytes, columns: tuple[str, str, str], delimiter: str) -> Table:
    if len(set(columns)) != 3 or not all(name.strip() for name in columns):
        raise ValueError("Choose three distinct, nonempty column names.")

    text, encoding = _decode(data)
    lines = text.splitlines(keepends=True)
    offset = next(
        (index + 1 for index, line in enumerate(lines) if line.strip() == "[Data]"),
        0,
    )
    while offset < len(lines) and (
        not lines[offset].strip() or lines[offset].lstrip().startswith("#")
    ):
        offset += 1
    if offset == len(lines):
        raise ValueError("The input has no table header.")

    choices = DELIMITERS if delimiter == "auto" else {delimiter: DELIMITERS[delimiter]}
    matching = []
    for name, separator in choices.items():
        header = next(csv.reader([lines[offset]], delimiter=separator))
        header = [column.strip() for column in header]
        if all(column in header for column in columns):
            matching.append((name, separator, header))

    if len(matching) != 1:
        if matching:
            raise ValueError("The delimiter is ambiguous. Choose --delimiter comma or tab.")
        header_preview = lines[offset].strip()[:240]
        raise ValueError(f"Could not find the selected columns in the header: {header_preview!r}.")

    name, separator, header = matching[0]
    if any(header.count(column) != 1 for column in columns):
        raise ValueError("The header repeats a selected column name.")

    positions = [header.index(column) for column in columns]
    reader = csv.reader(io.StringIO("".join(lines[offset:])), delimiter=separator, strict=True)
    next(reader)
    values = []
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        line_number = offset + reader.line_num
        if len(row) != len(header):
            raise ValueError(f"Line {line_number} has {len(row)} columns. Expected {len(header)}.")
        numeric = []
        for column, index in zip(columns, positions):
            try:
                value = float(row[index])
            except ValueError:
                raise ValueError(f"Line {line_number}, column {column!r} must be numeric.") from None
            if not np.isfinite(value):
                raise ValueError(f"Line {line_number}, column {column!r} must be finite.")
            numeric.append(value)
        values.append(numeric)

    if not values:
        raise ValueError("The table has no measurement rows.")
    return Table(*np.asarray(values, dtype=float).T, delimiter=name, encoding=encoding)
