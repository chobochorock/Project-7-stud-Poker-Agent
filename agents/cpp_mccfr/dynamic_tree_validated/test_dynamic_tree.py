"""Small deterministic checks. Run: python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_dynamic_tree -v.

Checks original-policy parity, save/resume, heldout splits, and frozen LBR privacy.
Temporary outputs stay in this model family's data folder.
"""

import csv
from pathlib import Path
import tempfile
import unittest

from . import run
from .report import leaf_support, read_lbr
from ..dynamic_tree_unlimited import run as original


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        row.pop("peak_rss_bytes", None)
    return result


class DynamicTreeTest(unittest.TestCase):
    def train(self, module, path, hands, mode=None, resume=False):
        args = ["train", "--run", str(path), "--hands", str(hands),
                "--particles", "4", "--panel-roots", "4", "--panel-every", "8",
                "--checkpoint-every", "8", "--window", "16", "--min-child", "2",
                "--gain", "0.01", "--min-rmse", "0", "--reserve-gib", "0"]
        if mode:
            args += ["--split-rule", mode]
        if resume:
            args += ["--resume"]
        self.assertEqual(module.train(module.parser().parse_args(args)), 0)

    def test_policies_resume_and_lbr(self):
        run.build(check=True)
        run.DATA.mkdir(parents=True, exist_ok=True)
        original.DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run.DATA, prefix="test_") as new_dir, \
                tempfile.TemporaryDirectory(dir=original.DATA, prefix="test_parity_") as old_dir:
            root = Path(new_dir)
            reference = Path(old_dir) / "run"
            self.train(original, reference, 32)
            for mode in ("raw", "centered", "validated"):
                full, resumed = root / (mode + "_full"), root / (mode + "_resume")
                self.train(run, full, 32, mode)
                self.train(run, resumed, 16, mode)
                with (resumed / "per_hand.csv").open("a") as stream:
                    stream.write("uncommitted suffix\n")
                self.train(run, resumed, 32, mode, True)
                support = leaf_support(full)
                self.assertEqual(support["hands"], 32)
                self.assertEqual(support["leaves"], int(rows(full / "per_hand.csv")[-1]["tree_leaves"]))
                self.assertGreaterEqual(support["monitor_samples"], 0)
                for filename in ("per_hand.csv", "panel.csv", "splits.csv"):
                    self.assertEqual(rows(full / filename), rows(resumed / filename))
                if mode == "raw":
                    for filename in ("per_hand.csv", "panel.csv"):
                        self.assertEqual(rows(full / filename), rows(reference / filename))
            options = run.parser().parse_args([
                "lbr", "--run", str(root / "validated_full"), "--output", str(root / "lbr"),
                "--pairs", "2", "--particles", "4", "--seeds", "12345"])
            run.evaluate_lbr(options)
            self.assertTrue((root / "lbr/summary.json").exists())
            read_lbr(root / "validated_full", root / "lbr", 32)
            with self.assertRaisesRegex(ValueError, "Stale LBR"):
                read_lbr(root / "validated_full", root / "lbr", 33)


if __name__ == "__main__":
    unittest.main()
