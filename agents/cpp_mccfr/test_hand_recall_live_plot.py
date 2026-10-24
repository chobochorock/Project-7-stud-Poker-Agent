"""Checks for read-only report parsing; python -m unittest this module."""

import unittest
from pathlib import Path
import sqlite3
import tempfile

from .hand_recall_disk import HEADER, POLICIES
from .hand_recall_live_plot import means, parse_rows, read_committed_metrics, validate_trace


class LiveReportTest(unittest.TestCase):
    def test_batched_committed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with sqlite3.connect(run / "model.sqlite") as db:
                db.executescript(
                    "CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB);"
                    "CREATE TABLE metrics(kind INTEGER,hand INTEGER,root INTEGER,line TEXT,"
                    "PRIMARY KEY(kind,hand,root)) WITHOUT ROWID;"
                )
                db.executemany("INSERT INTO meta VALUES(?,?)", [
                    ("hands", b"3"), ("identity", b"test"), ("status", b"ready")
                ])
                db.executemany("INSERT INTO metrics VALUES(?,?,?,?)", [
                    (0, h, 0, str(h)) for h in range(1, 5)
                ] + [(1, 0, 0, "panel0"), (1, 4, 0, "future_panel")])
            db.close()
            meta, rows = read_committed_metrics(run, batch_hands=2)
            self.assertEqual(meta["hands"], "3")
            self.assertEqual(rows, [["1", "2", "3"], ["panel0"]])
            with sqlite3.connect(run / "model.sqlite") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM metrics").fetchone()[0], 6)
            db.close()
            with self.assertRaises(ValueError):
                read_committed_metrics(run, batch_hands=0)

    def test_legacy_missing_diagnostics_and_exact_seed(self):
        fields = ["1"] * 21
        fields[5] = "18446744073709551615"
        rows = parse_rows([",".join(fields)])
        validate_trace(rows, 1)
        self.assertEqual(rows[0]["root_seed"], fields[5])
        self.assertIsNone(means(rows)[POLICIES[0]]["crossfit"])
        with self.assertRaises(ValueError):
            validate_trace(rows, 2)

    def test_signed_crossfit_and_invalid_gap(self):
        fields = ["1"] * len(HEADER)
        fields[HEADER.index(f"{POLICIES[0]}_crossfit")] = "-2"
        self.assertEqual(means(parse_rows([",".join(fields)]))[POLICIES[0]]["crossfit"], -2)
        fields[1] = "nan"
        with self.assertRaises(ValueError):
            parse_rows([",".join(fields)])


if __name__ == "__main__":
    unittest.main()
