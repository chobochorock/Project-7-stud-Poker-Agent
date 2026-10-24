"""Small deterministic checks. Run: python -m unittest agents.cpp_mccfr.shadow_split.test_shadow_split -v.

Covers the C++ self-test (edges, keys, cancellation split, inheritance B, stuck leaves,
one traverser visit per leaf and traversal for own/exact keys, resume, gap evaluator),
k-means arm parity with dynamic_tree_validated on the same seed, exact resume of the CSV
logs, and a tiny frozen Policy-LBR run. Temporary outputs stay in this family's data folder.
"""

import csv
from pathlib import Path
import tempfile
import unittest

from . import run
from ..dynamic_tree_validated import run as validated


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        row.pop("peak_rss_bytes", None)
    return result


class ShadowSplitTest(unittest.TestCase):
    def train(self, module, path, hands, extra=(), resume=False):
        args = ["train", "--run", str(path), "--hands", str(hands), "--particles", "4",
                "--panel-roots", "4", "--panel-every", "8", "--checkpoint-every", "8",
                "--reserve-gib", "0", *extra]
        if resume:
            args += ["--resume"]
        self.assertEqual(module.train(module.parser().parse_args(args)), 0)

    def test_parity_resume_and_lbr(self):
        run.build(check=True)
        run.DATA.mkdir(parents=True, exist_ok=True)
        validated.DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run.DATA, prefix="test_") as new_dir, \
                tempfile.TemporaryDirectory(dir=validated.DATA, prefix="test_shadow_parity_") as old_dir:
            root = Path(new_dir)
            reference = Path(old_dir) / "run"
            validated.DATA.mkdir(parents=True, exist_ok=True)
            self.train(validated, reference, 24, ("--seed", "222", "--window", "16", "--min-child", "2"))
            options = ("--seed", "222", "--grace", "4", "--shadow-after", "4", "--z", "0.3")
            full, resumed = root / "full", root / "resumed"
            self.train(run, full, 24, options)
            self.train(run, resumed, 16, options)
            with (resumed / "per_hand.csv").open("a") as stream:
                stream.write("uncommitted suffix\n")
            self.train(run, resumed, 24, options, resume=True)
            for filename in ("per_hand.csv", "panel.csv", "splits.csv"):
                self.assertEqual(rows(full / filename), rows(resumed / filename))
            mine, theirs = rows(full / "per_hand.csv"), rows(reference / "per_hand.csv")
            for column in ("kmeans_current", "kmeans_average", "kmeans_states", "root_seed"):
                self.assertEqual([r[column] for r in mine], [r[column] for r in theirs])
            self.assertEqual(int(mine[-1]["revisits"]) >= 0, True)
            args = run.parser().parse_args([
                "lbr", "--run", str(full), "--output", str(root / "lbr"),
                "--pairs", "2", "--particles", "4", "--seeds", "12345"])
            run.evaluate_lbr(args)
            self.assertTrue((root / "lbr" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
