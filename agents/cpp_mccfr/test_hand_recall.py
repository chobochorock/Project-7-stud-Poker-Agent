"""Non-training checks: python -m unittest agents.cpp_mccfr.test_hand_recall -v."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .hand_recall import ATLAS, ATLAS_SHA, FAMILY, PROJECT, SOURCES, sha256
from .hand_recall_plot import plot, rolling_bands


class HandRecallTest(unittest.TestCase):
    def test_frozen_inputs_and_paths(self):
        self.assertEqual(sha256(ATLAS), ATLAS_SHA)
        self.assertTrue((PROJECT / "AGENTS.md").exists())
        self.assertTrue(all(path.is_file() for path in SOURCES))

    def test_bands_include_zeros_and_use_full_trailing_windows(self):
        values = np.array([0, 1, 9, 2, 8, 3], dtype=float)
        actual = rolling_bands(values, 3)
        expected = []
        for index in range(2, len(values)):
            block = values[index - 2 : index + 1]
            expected.append(
                [
                    min(block),
                    np.quantile(block, 0.05),
                    block.mean(),
                    np.quantile(block, 0.95),
                    max(block),
                ]
            )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual[0, 0], 0)
        for values, window in [([1], 2), ([1], 0), ([float("nan")], 1)]:
            with self.assertRaises(ValueError):
                rolling_bands(values, window)

    def test_render_has_no_scatter_and_keeps_zero_statistics(self):
        import matplotlib.axes

        # Synthetic plotting fixture, NOT a game/training/performance result.
        with tempfile.TemporaryDirectory(
            prefix="hand_recall_check_", dir=FAMILY / "data"
        ) as name:
            run = Path(name).resolve()
            self.assertTrue(run.is_relative_to((FAMILY / "data").resolve()))
            (run / "config.json").write_text(
                json.dumps(dict(config=dict(particles=32, panel_roots=2))),
                encoding="utf-8",
            )
            for file, hands in (("per_hand.csv", range(1, 121)), ("panel.csv", [0, 0])):
                with (run / file).open("w", newline="", encoding="utf-8") as out:
                    writer = csv.writer(out)
                    writer.writerow(
                        [
                            "hands",
                            "recall_current",
                            "baseline_current",
                            "recall_average",
                            "baseline_average",
                        ]
                    )
                    writer.writerows([hand, hand % 7, 2, hand % 5, 1] for hand in hands)
            with patch.object(
                matplotlib.axes.Axes,
                "scatter",
                side_effect=AssertionError("No scatter"),
            ):
                plot(run, 100)
            summary = json.loads((run / "plot_summary.json").read_text())
            self.assertEqual(summary["hands"], 120)
            self.assertEqual(summary["raw_zero_counts"], [17, 0, 24, 0])
            self.assertGreater((run / "local_br_gap_log.png").stat().st_size, 1000)
            self.assertGreater((run / "local_br_gap_linear.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
