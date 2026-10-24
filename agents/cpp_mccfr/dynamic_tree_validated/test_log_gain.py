"""Small log-gain integration checks, never a 1M experiment.

Usage: python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_log_gain -v
Temporary runs are confined to this model family's data directory.
"""

from pathlib import Path
import tempfile
import unittest

from . import log_gain, log_gain_report, run
from .test_dynamic_tree import rows
from .test_schedule import solver_payload


class LogGainTest(unittest.TestCase):
    def options(self, path, hands=128, extra=()):
        return log_gain.parser().parse_args(
            [
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
                "16",
                "--checkpoint-every",
                "16",
                "--window",
                "16",
                "--min-child",
                "2",
                "--min-rmse",
                "0",
                "--reserve-gib",
                "0",
                *extra,
            ]
        )

    def test_defaults(self):
        args = log_gain.parser().parse_args(["train"])
        log_gain.validate(args)
        self.assertEqual(
            (args.hands, args.seed, args.panel_every), (1000000, 222, 10000)
        )
        self.assertEqual(args.checkpoint_every, 10000)
        self.assertIsNone(args.fixed_gain)
        for flag, value in (
            ("--fixed-gain", "0"),
            ("--fixed-gain", "nan"),
            ("--reserve-gib", "nan"),
            ("--window", "0"),
            ("--seed", str(2**64)),
        ):
            with self.assertRaises(ValueError):
                log_gain.validate(log_gain.parser().parse_args(["train", flag, value]))

    def test_resume_parity_and_existing_solver(self):
        run.DATA.mkdir(parents=True, exist_ok=True)
        log_gain.build(check=True)
        with tempfile.TemporaryDirectory(
            dir=run.DATA, prefix="test_log_gain_"
        ) as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(run.DATA.resolve()))
            full, resumed, fixed, original = (
                root / name for name in ("full", "resumed", "fixed", "original")
            )
            self.assertEqual(log_gain.train(self.options(full)), 0)
            self.assertEqual(log_gain.train(self.options(resumed, 64)), 0)
            for name in ("per_hand.csv", "panel.csv", "splits.csv"):
                with (resumed / name).open("a") as handle:
                    handle.write("uncommitted suffix\n")
            self.assertEqual(
                log_gain.train(self.options(resumed, extra=("--resume",))), 0
            )
            self.assertEqual(solver_payload(full), solver_payload(resumed))
            self.assertEqual(rows(full / "panel.csv"), rows(resumed / "panel.csv"))
            self.assertEqual(rows(full / "splits.csv"), rows(resumed / "splits.csv"))
            decisions = rows(full / "splits.csv")
            self.assertTrue(decisions)
            self.assertTrue(any(int(row["accepted"]) for row in decisions))
            summary = log_gain_report.report(full, root / "report")
            self.assertEqual(summary["hands"], 128)
            self.assertTrue((root / "report/local_br_gap_log.png").exists())
            before = run.legacy.sha256(full / "checkpoint.bin")
            with self.assertRaises(ValueError):
                log_gain.train(self.options(full))
            with self.assertRaises(ValueError):
                log_gain.train(
                    self.options(full, extra=("--resume", "--fixed-gain", "0.15"))
                )
            self.assertEqual(before, run.legacy.sha256(full / "checkpoint.bin"))
            self.assertEqual(
                log_gain.train(self.options(fixed, 64, ("--fixed-gain", "0.15"))), 0
            )
            args = run.parser().parse_args(
                [
                    "train",
                    "--run",
                    str(original),
                    "--hands",
                    "64",
                    "--seed",
                    "222",
                    "--particles",
                    "4",
                    "--panel-roots",
                    "4",
                    "--panel-every",
                    "16",
                    "--checkpoint-every",
                    "16",
                    "--window",
                    "16",
                    "--min-child",
                    "2",
                    "--min-rmse",
                    "0",
                    "--gain",
                    "0.15",
                    "--split-rule",
                    "centered",
                    "--reserve-gib",
                    "0",
                ]
            )
            self.assertEqual(run.train(args), 0)
            self.assertEqual(solver_payload(fixed), solver_payload(original))
            a = [r for r in rows(full / "panel.csv") if int(r["hands"]) <= 64]
            b = rows(fixed / "panel.csv")
            self.assertEqual(
                [
                    {
                        k: v
                        for k, v in row.items()
                        if k.startswith("kmeans_") or k == "root_seed"
                    }
                    for row in a
                ],
                [
                    {
                        k: v
                        for k, v in row.items()
                        if k.startswith("kmeans_") or k == "root_seed"
                    }
                    for row in b
                ],
            )
            lbr = log_gain.parser().parse_args(
                [
                    "lbr",
                    "--run",
                    str(full),
                    "--output",
                    str(root / "lbr"),
                    "--pairs",
                    "2",
                    "--particles",
                    "4",
                    "--seeds",
                    "3456",
                ]
            )
            log_gain.manifest_at(full)
            run.evaluate_lbr(lbr)
            self.assertEqual(before, run.legacy.sha256(full / "checkpoint.bin"))


if __name__ == "__main__":
    unittest.main()
