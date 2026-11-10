"""Command-line interface and replayable processing records."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import matplotlib
import numpy as np

from . import __version__
from .core import analyze
from .io import read_table
from .plot import plot


PROCESSED_COLUMNS = (
    "field",
    "longitudinal_up",
    "longitudinal_down",
    "transverse_up",
    "transverse_down",
    "longitudinal_even_up",
    "longitudinal_even_down",
    "transverse_odd_up",
    "transverse_odd_down",
    "naive_odd_up",
    "naive_odd_down",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def process(
    source: Path,
    destination: Path,
    columns: tuple[str, str, str],
    *,
    delimiter: str,
    field_tolerance: float,
    pair: int | None,
    expected: dict | None = None,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output path already exists: {destination}")

    raw = source.read_bytes()
    raw_hash = sha256(raw)
    if expected is not None and raw_hash != expected["input"]["sha256"]:
        raise ValueError("Input SHA-256 differs from the recipe. Select the original input bytes.")
    table = read_table(raw, columns, delimiter)
    result = analyze(*table.signals, field_tolerance=field_tolerance, pair=pair)
    selected = next(candidate for candidate in result.candidates if candidate["selected"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hallloop-", dir=destination.parent) as staging:
        bundle = Path(staging) / "output"
        bundle.mkdir()
        with (bundle / "processed.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(PROCESSED_COLUMNS)
            writer.writerows(zip(*(getattr(result, column) for column in PROCESSED_COLUMNS)))

        plot(
            result,
            bundle / "diagnostics.png",
            field_label=columns[0],
            longitudinal_label=columns[1],
            transverse_label=columns[2],
            raw=table.signals,
        )
        hashes = {
            name: sha256((bundle / name).read_bytes())
            for name in ("processed.csv", "diagnostics.png")
        }
        recipe = {
            "schema_version": 1,
            "software": {
                "hallloop": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "input": {
                "filename": source.name,
                "sha256": raw_hash,
                "encoding": table.encoding,
                "data_records": len(table.field),
            },
            "options": {
                "columns": list(columns),
                "delimiter": table.delimiter,
                "field_tolerance": field_tolerance,
                "pair": selected["id"],
                "assume_reversal_related": True,
            },
            "analysis": {
                "selected_branches": result.branches,
                "candidates": result.candidates,
                "common_field_range": selected["common_field_range"],
                "output_field_points": len(result.field),
                "interpolation": "linear within measured branch coverage",
            },
            "output_sha256": hashes,
        }
        if expected is not None:
            previous = expected["output_sha256"]
            recipe["replay"] = {
                name: {
                    "original_sha256": previous[name],
                    "matches_original": digest == previous[name],
                }
                for name, digest in hashes.items()
            }

        (bundle / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Output path already exists: {destination}")
        bundle.rename(destination)

    print(f"Wrote {destination / 'processed.csv'}, diagnostics.png, and recipe.json")
    if expected is not None:
        for name, details in recipe["replay"].items():
            status = "matches" if details["matches_original"] else "differs from"
            print(f"{name} {status} the original SHA-256.")


def read_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    try:
        if recipe["schema_version"] != 1 or recipe["software"]["hallloop"] != __version__:
            raise ValueError("Use the HallLoop version that wrote this recipe.")
        options = recipe["options"]
        if (
            not isinstance(options["columns"], list)
            or len(options["columns"]) != 3
            or not all(isinstance(name, str) for name in options["columns"])
            or options["delimiter"] not in ("comma", "tab")
            or not isinstance(options["field_tolerance"], (float, int))
            or type(options["pair"]) is not int
            or options["pair"] < 1
            or options["assume_reversal_related"] is not True
        ):
            raise ValueError("The recipe has invalid processing options.")
        for digest in (
            recipe["input"]["sha256"],
            recipe["output_sha256"]["processed.csv"],
            recipe["output_sha256"]["diagnostics.png"],
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("The recipe has an invalid SHA-256.")
    except (KeyError, TypeError) as error:
        raise ValueError("The file is missing required HallLoop recipe fields.") from error
    return recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and replay Hall-loop processing.")
    parser.add_argument("--version", action="version", version=f"HallLoop {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("process", help="process a field loop and record the choices")
    run.add_argument("input", type=Path)
    run.add_argument("--field", required=True, help="field column name")
    run.add_argument("--longitudinal", required=True, help="longitudinal channel name")
    run.add_argument("--transverse", required=True, help="transverse channel name")
    run.add_argument(
        "--assume-reversal-related",
        action="store_true",
        required=True,
        help="declare that the paired magnetic histories are reversal-related",
    )
    run.add_argument("--delimiter", choices=("auto", "comma", "tab"), default="auto")
    run.add_argument("--field-tolerance", type=float, default=0.0, help="in field column units")
    run.add_argument("--pair", type=int, help="1-based candidate pair when multiple loops are found")
    run.add_argument("--out", required=True, type=Path, help="new output directory")
    replay = commands.add_parser("replay", help="rerun processing from the recipe")
    replay.add_argument("recipe", type=Path)
    replay.add_argument("--input", required=True, type=Path)
    replay.add_argument("--out", required=True, type=Path, help="new output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "process":
            process(
                args.input,
                args.out,
                (args.field, args.longitudinal, args.transverse),
                delimiter=args.delimiter,
                field_tolerance=args.field_tolerance,
                pair=args.pair,
            )
        else:
            recipe = read_recipe(args.recipe)
            options = recipe["options"]
            process(
                args.input,
                args.out,
                tuple(options["columns"]),
                delimiter=options["delimiter"],
                field_tolerance=options["field_tolerance"],
                pair=options["pair"],
                expected=recipe,
            )
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"hallloop: {error}\n")
    return 0
