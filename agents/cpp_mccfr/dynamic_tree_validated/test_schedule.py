"""Tiny parity/fork/resume/freeze checks; never runs a large experiment.

Usage: python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_schedule -v
Temporary test runs stay in the family's data folder and are removed afterwards.
"""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from . import run, schedule, schedule_report
from .report import leaf_support
from .test_dynamic_tree import rows


def solver_payload(path):
    """Compare logical state, not unordered_map serialization order."""
    with (path / "checkpoint.bin").open("rb") as handle:

        def read(fmt):
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))

        def text():
            return handle.read(read("I")[0])

        text()
        text()
        handle.seek(32, 1)
        solvers = []
        for _ in range(2):
            mode, visits, rng = read("B"), read("Q"), text()
            table = {text(): read("16dQ") for _ in range(read("Q")[0])}
            counts, monitor_rng = read("2Q"), text()
            nodes = []
            for _ in range(read("I")[0]):
                node = read("IIiiifQQQI")
                nodes.append((node, handle.read(node[-1] * 208)))
            roots = [read("Ii") for _ in range(read("I")[0])]
            solvers.append(
                (mode, visits, rng, table, counts, monitor_rng, nodes, roots)
            )
        if handle.read():
            raise ValueError("Trailing checkpoint bytes")
        return solvers


class ScheduleTest(unittest.TestCase):
    def original(self, path, hands, resume=False):
        args = run.parser().parse_args(
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
                "8",
                "--checkpoint-every",
                "8",
                "--window",
                "16",
                "--min-child",
                "2",
                "--gain",
                "0.01",
                "--min-rmse",
                "0",
                "--split-rule",
                "centered",
                "--reserve-gib",
                "0",
            ]
            + (["--resume"] if resume else [])
        )
        self.assertEqual(run.train(args), 0)

    def fork(self, parent, path, mode, extra=()):
        args = schedule.parser().parse_args(
            [
                "fork",
                "--source",
                str(parent),
                "--run",
                str(path),
                "--mode",
                mode,
                "--eval-every",
                "8",
                "--reserve-gib",
                "0",
                *extra,
            ]
        )
        schedule.fork(args)

    def extend(self, path, hands):
        args = schedule.parser().parse_args(
            [
                "train",
                "--run",
                str(path),
                "--hands",
                str(hands),
                "--checkpoint-every",
                "8",
                "--reserve-gib",
                "0",
            ]
        )
        self.assertEqual(schedule.train(args), 0)

    def test_branches(self):
        run.DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=run.DATA, prefix="test_schedule_"
        ) as directory:
            root = Path(directory)
            parent = root / "parent"
            self.original(parent, 32)
            before = run.legacy.sha256(parent / "checkpoint.bin")
            branches = {
                name: root / name
                for name in (
                    "continue",
                    "resumed",
                    "freeze",
                    "freeze_resume",
                    "slow",
                    "slow_resume",
                )
            }
            for name, path in branches.items():
                extra = (
                    ["--min-split-visits", "64", "--split-check-visits", "64"]
                    if name.startswith("slow")
                    else []
                )
                self.fork(
                    parent,
                    path,
                    "freeze" if name.startswith("freeze") else "continue",
                    extra,
                )
                self.assertEqual(
                    rows(path / "panel.csv"), rows(parent / "panel.csv")[-4:]
                )
            self.assertEqual(before, run.legacy.sha256(parent / "checkpoint.bin"))
            frozen_before = leaf_support(branches["freeze"])
            self.assertEqual(frozen_before["monitor_samples"], 0)
            self.extend(branches["continue"], 64)
            self.extend(branches["freeze"], 64)
            self.extend(branches["slow"], 64)
            self.extend(branches["resumed"], 48)
            with (branches["resumed"] / "panel.csv").open("a") as handle:
                handle.write("uncommitted suffix\n")
            self.extend(branches["resumed"], 64)
            for name in ("panel.csv", "splits.csv"):
                self.assertEqual(
                    rows(branches["continue"] / name), rows(branches["resumed"] / name)
                )
            for mode in ("freeze", "slow"):
                self.extend(branches[mode + "_resume"], 48)
                self.extend(branches[mode + "_resume"], 64)
                self.assertEqual(
                    solver_payload(branches[mode]),
                    solver_payload(branches[mode + "_resume"]),
                )
            after = leaf_support(branches["freeze"])
            self.assertEqual(after["splits"], frozen_before["splits"])
            self.assertEqual(after["leaves"], frozen_before["leaves"])
            self.assertEqual(after["monitor_samples"], 0)
            self.assertGreater(
                float(rows(branches["freeze"] / "panel.csv")[-1]["tree_nodes"]),
                float(rows(branches["freeze"] / "panel.csv")[0]["tree_nodes"]),
            )
            schedule_report.compare(
                {k: branches[k] for k in ("freeze", "continue", "slow")},
                root / "review",
            )
            # Existing training and sparse evaluation must produce identical models/policies.
            self.original(parent, 64, True)
            expected = [r for r in rows(parent / "panel.csv") if int(r["hands"]) >= 32]
            self.assertEqual(expected, rows(branches["continue"] / "panel.csv"))
            self.assertEqual(
                solver_payload(parent), solver_payload(branches["continue"])
            )
            options = run.parser().parse_args(
                [
                    "lbr",
                    "--run",
                    str(branches["freeze"]),
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
            run.evaluate_lbr(options)
            self.assertTrue(
                json.loads((root / "lbr/summary.json").read_text())["input_unchanged"]
            )


if __name__ == "__main__":
    unittest.main()
