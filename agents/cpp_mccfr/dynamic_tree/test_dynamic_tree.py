"""Small integration checks, not a performance experiment.

Usage: python -m unittest agents.cpp_mccfr.dynamic_tree.test_dynamic_tree -v
Uses 24 continuous vs 12+12 resumed hands, 4 particles, and temporary run dirs.
"""

import csv
from pathlib import Path
import tempfile
import unittest

from .run import DATA, build, parser, train


class DynamicTreeTest(unittest.TestCase):
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
                    "--max-splits",
                    "8",
                    "--max-depth",
                    "3",
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
            from .plot import plot

            plot(root / "continuous", 4)
            self.assertTrue((root / "continuous/local_br_gap_log.png").is_file())


if __name__ == "__main__":
    unittest.main()
