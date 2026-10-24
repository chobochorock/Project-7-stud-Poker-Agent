"""Tiny storage checks, not a training benchmark.

Usage: python -m unittest agents.cpp_mccfr.test_hand_recall_disk -v
Builds the disk driver, compares uninterrupted/resumed updates, and removes
only its temporary runs. The C++ check covers RAM parity and legacy import.
"""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from . import hand_recall_disk as disk


class DiskStorageTest(unittest.TestCase):
    def test_storage_and_resume(self):
        exe = disk.build(check=True)
        with tempfile.TemporaryDirectory(
            prefix="disk_check_", dir=disk.FAMILY / "data"
        ) as temporary:
            root = Path(temporary)
            uninterrupted, resumed = root / "full", root / "resumed"

            def run(directory, hands):
                directory.mkdir(exist_ok=True)
                shutil.copy2(disk.legacy.ATLAS, directory / "hand_atlas.bin")
                subprocess.run(
                    [
                        str(exe),
                        "train",
                        str(directory),
                        "test_identity",
                        str(hands),
                        "111",
                        "307",
                        "4",
                        "2",
                        "2",
                        "2",
                        "1",
                        "3",
                        "0",
                        "",
                        "",
                        "1",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

            run(uninterrupted, 4)
            run(resumed, 2)
            run(resumed, 4)
            with disk.connect(uninterrupted) as first, disk.connect(resumed) as second:
                self.assertEqual(
                    first.execute("SELECT k,v FROM nodes ORDER BY k").fetchall(),
                    second.execute("SELECT k,v FROM nodes ORDER BY k").fetchall(),
                )
                # Timings differ. All counters/RNG after the 24-byte progress block do not.
                a = first.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0]
                b = second.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0]
                header_size = 4 + len(b"RECALLDISK1")
                self.assertEqual(a[: header_size + 8], b[: header_size + 8])
                self.assertEqual(a[header_size + 24 :], b[header_size + 24 :])
                self.assertEqual(disk.metadata(resumed)["hands"], "4")
                lines_a = first.execute(
                    "SELECT line FROM metrics ORDER BY kind,hand,root"
                ).fetchall()
                lines_b = second.execute(
                    "SELECT line FROM metrics ORDER BY kind,hand,root"
                ).fetchall()
                self.assertEqual(len(lines_a), 10)
                for (a,), (b,) in zip(lines_a, lines_b):
                    a, b = next(csv.reader([a])), next(csv.reader([b]))
                    self.assertEqual(a[:20] + a[21:], b[:20] + b[21:])
                self.assertEqual(
                    second.execute("PRAGMA integrity_check").fetchone()[0], "ok"
                )
                self.assertEqual(
                    second.execute("SELECT count(*) FROM action_values").fetchone()[0],
                    24,
                )
            disk.export_metrics(resumed)
            with (resumed / "per_hand.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["hands"] for row in rows], ["1", "2", "3", "4"])
            self.assertTrue(all(row["baseline_average_mean_action_se"] for row in rows))
            with disk.run_lock(resumed):
                with self.assertRaises(ValueError):
                    with disk.run_lock(resumed):
                        pass
            # The existing red/blue plotter must also accept the extended CSV.
            (resumed / "config.json").write_text(
                json.dumps({"config": {"particles": 4, "panel_roots": 2}}),
                encoding="utf-8",
            )
            from .hand_recall_plot import plot

            plot(resumed, 2)
            self.assertTrue((resumed / "local_br_gap_log.png").is_file())


if __name__ == "__main__":
    unittest.main()
