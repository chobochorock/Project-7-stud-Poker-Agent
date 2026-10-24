"""Small integration checks, not a performance experiment.

Usage: python -m unittest agents.cpp_mccfr.dynamic_tree_unlimited.test_dynamic_tree -v
Uses 24 continuous vs 12+12 resumed hands, 4 particles, and temporary run dirs.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .run import DATA, build, parser, train


class DynamicTreeTest(unittest.TestCase):
    def test_unlimited_defaults(self):
        options = parser().parse_args(["train", "--run", str(DATA / "unused")])
        self.assertEqual((options.max_splits, options.max_depth), (-1, -1))
        self.assertEqual(options.particles, 32)
        self.assertEqual(options.max_kmeans_states, 500000)
        options = parser().parse_args(["train", "--run", str(DATA / "unused"), "--max-splits", "0"])
        self.assertEqual(options.max_splits, 0)

    def test_cli(self):
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_cli_", dir=DATA) as directory:
            path = Path(directory) / "run"
            command = [
                sys.executable, "-m", "agents.cpp_mccfr.dynamic_tree_unlimited.run",
                "train", "--run", str(path), "--hands", "1", "--particles", "2",
                "--panel-roots", "2", "--reserve-gib", "0",
            ]
            for flag in ("--max-splits", "--max-depth"):
                failed = subprocess.run(command + [flag, "-2"], capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
                self.assertFalse(path.exists())
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("max_splits=unlimited max_depth=unlimited", result.stdout)
            config = json.loads((path / "config.json").read_text())["config"]
            self.assertEqual((config["max_splits"], config["max_depth"]), (-1, -1))
            with (path / "per_hand.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_build_split_and_resume(self):
        build(check=True)
        DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_", dir=DATA) as directory:
            root = Path(directory)

            def run(path, hands, resume=False):
                arguments = [
                    "train",
                    "--run",
                    str(path),
                    "--hands",
                    str(hands),
                    "--particles",
                    "4",
                    "--panel-roots",
                    "4",
                    "--panel-every",
                    "8",
                    "--checkpoint-every",
                    "8",
                    "--window",
                    "8",
                    "--min-child",
                    "2",
                    "--gain",
                    "0.01",
                    "--min-rmse",
                    "0",
                    "--reserve-gib",
                    "0",
                ]
                if resume:
                    arguments.append("--resume")
                self.assertEqual(train(parser().parse_args(arguments)), 0)

            run(root / "continuous", 24)
            run(root / "resumed", 12)
            # A flushed suffix beyond the last checkpoint must not be replayed twice.
            with (root / "resumed/per_hand.csv").open("a") as handle:
                handle.write("uncommitted suffix\n")
            run(root / "resumed", 24, True)
            for filename in ("per_hand.csv", "panel.csv"):
                with (root / "continuous" / filename).open() as first, (
                    root / "resumed" / filename
                ).open() as second:
                    left, right = list(csv.DictReader(first)), list(
                        csv.DictReader(second)
                    )
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    a.pop("peak_rss_bytes")
                    b.pop("peak_rss_bytes")
                    self.assertEqual(a, b)
                self.assertGreater(int(left[-1]["splits"]), 0)
            self.assertEqual(
                (root / "continuous/splits.csv").read_bytes(),
                (root / "resumed/splits.csv").read_bytes(),
            )
            with self.assertRaises(ValueError):
                run(root / "continuous", 24)
            from ..dynamic_tree.plot import plot

            plot(root / "continuous", 4)
            self.assertTrue((root / "continuous/local_br_gap_log.png").is_file())


if __name__ == "__main__":
    unittest.main()
