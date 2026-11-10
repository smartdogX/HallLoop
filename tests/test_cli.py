"""End-to-end checks for the public HallLoop command line."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MIXED_LOOP = ROOT / "examples" / "mixed_loop.csv"
INCOMPLETE_SWEEP = ROOT / "examples" / "incomplete_sweep.csv"
EXAMPLE_COLUMNS = ("field_T", "longitudinal_ohm", "transverse_ohm")


class HallLoopCLITests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.directory = Path(workspace.name)

    def run_hallloop(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "hallloop", *map(str, args)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def process(self, source, output, columns=EXAMPLE_COLUMNS):
        field, longitudinal, transverse = columns
        return self.run_hallloop(
            "process",
            source,
            "--field", field,
            "--longitudinal", longitudinal,
            "--transverse", transverse,
            "--assume-reversal-related",
            "--out", output,
        )

    def test_process_and_replay_csv(self):
        output = self.directory / "original"
        processed = self.process(MIXED_LOOP, output)
        self.assertEqual(processed.returncode, 0, processed.stderr)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"processed.csv", "diagnostics.png", "recipe.json"},
        )
        self.assertTrue((output / "diagnostics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        recipe_path = output / "recipe.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        self.assertEqual(recipe["options"]["delimiter"], "comma")
        self.assertEqual(recipe["input"]["sha256"], hashlib.sha256(MIXED_LOOP.read_bytes()).hexdigest())

        replay_dir = self.directory / "replayed"
        replay = self.run_hallloop(
            "replay", recipe_path, "--input", MIXED_LOOP, "--out", replay_dir
        )
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(
            (replay_dir / "processed.csv").read_bytes(),
            (output / "processed.csv").read_bytes(),
        )
        replay_recipe = json.loads((replay_dir / "recipe.json").read_text(encoding="utf-8"))
        self.assertTrue(replay_recipe["replay"]["processed.csv"]["matches_original"])

    def test_processes_tab_table_after_data_marker(self):
        source = self.directory / "instrument.dat"
        columns = ("Field (Oe)", "Rxx (ohm)", "Ryx (ohm)")
        records = [
            "[Header]",
            "INFO, synthetic instrument export",
            "[Data]",
            "\t".join(columns),
            "-1000\t12\t-3",
            "0\t10\t1",
            "1000\t12\t5",
            "1000\t12\t3",
            "0\t10\t-1",
            "-1000\t12\t-5",
        ]
        source.write_text("\n".join(records) + "\n", encoding="utf-16")
        output = self.directory / "instrument-output"
        result = self.process(source, output, columns)
        self.assertEqual(result.returncode, 0, result.stderr)
        recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["options"]["delimiter"], "tab")
        self.assertEqual(recipe["options"]["columns"], list(columns))
        self.assertEqual(recipe["input"]["data_records"], 6)
        with (output / "processed.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["field"]) for row in rows], [-1000.0, 0.0, 1000.0])
        self.assertEqual([float(row["transverse_odd_up"]) for row in rows], [-3.0, 1.0, 5.0])

    def test_replay_refuses_modified_input(self):
        output = self.directory / "original"
        processed = self.process(MIXED_LOOP, output)
        self.assertEqual(processed.returncode, 0, processed.stderr)
        changed = self.directory / "changed.csv"
        changed.write_bytes(MIXED_LOOP.read_bytes() + b"\n")
        replay_dir = self.directory / "replayed"

        replay = self.run_hallloop(
            "replay", output / "recipe.json", "--input", changed, "--out", replay_dir
        )

        self.assertEqual(replay.returncode, 2, replay.stderr)
        self.assertIn("SHA-256 differs", replay.stderr)
        self.assertFalse(replay_dir.exists())

    def test_incomplete_sweep_refuses_without_creating_output(self):
        output = self.directory / "incomplete-output"

        result = self.process(INCOMPLETE_SWEEP, output)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("No candidate pairs", result.stderr)
        self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_output(self):
        output = self.directory / "existing-output"
        output.mkdir()
        existing = output / "processed.csv"
        existing.write_bytes(b"existing output\n")

        result = self.process(MIXED_LOOP, output)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Output path already exists", result.stderr)
        self.assertEqual(existing.read_bytes(), b"existing output\n")
        self.assertEqual({path.name for path in output.iterdir()}, {"processed.csv"})


if __name__ == "__main__":
    unittest.main()
