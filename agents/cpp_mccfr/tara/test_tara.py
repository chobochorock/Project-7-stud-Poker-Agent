"""Small deterministic checks. Run: python -m unittest agents.cpp_mccfr.tara.test_tara -v.

Covers the C++ self-test (features, keys, card-blind tree 1, pow2 = halving bit-exact, weights,
latest-only, reservoir, resume, gap evaluator, B0 kind-2 forest = k-means partition), forest fitting (sklearn routing reproduced with
float32 thresholds, D'Hondt allocation, random-partition control, legality of buckets), a short
interval-by-interval run whose k-means arm matches dynamic_tree_validated on the same seed,
exact resume/extension, bit-identical probe policies at tree addition, a tiny Policy-LBR run, and
short B0-sched / B0-single runs (snapshots, kind-2 forests, k-means parity).
Temporary outputs stay in this family's data folder.
"""

import csv
import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from . import fit, run
from ..dynamic_tree_validated import run as validated


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        row.pop("peak_rss_bytes", None)
    return result


def route(nodes, x):
    i = 0
    while nodes[i][0] >= 0:
        feature, threshold, left, right = nodes[i]
        i = left if np.float32(x[feature]) <= np.float32(threshold) else right
    return i


def synthetic_slot(seed, n=600, mask=0b1010011):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, fit.FEATURES)).astype(np.float32)
    r = np.zeros((n, fit.ACTIONS), np.float32)
    for a in fit.legal(mask):
        r[:, a] = (x[:, a % fit.FEATURES] > 0) * 3 - 1 + rng.normal(scale=0.1, size=n)
    return dict(key=b"k%d" % seed, mask=mask, street=5, visits=n, mass=float(np.abs(r).sum()), x=x, r=r)


class FitTest(unittest.TestCase):
    def test_dhondt(self):
        self.assertEqual(fit.dhondt([10.0, 1.0, 0.0], [5, 5, 5], 6), [4, 1, 1])
        self.assertEqual(fit.dhondt([10.0, 1.0], [2, 9], 7), [2, 5])
        self.assertEqual(sum(fit.dhondt([0.0, 0.0], [1, 1], 9)), 2)

    def test_floor32_routing_matches_sklearn(self):
        from sklearn.tree import DecisionTreeRegressor

        slot = synthetic_slot(1)
        records, stats = fit.fit_forest([slot], 8, 20, 7)
        nodes = records[0][2]
        actions = fit.legal(slot["mask"])
        model = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=20, random_state=7)
        model.fit(slot["x"], slot["r"][:, actions].astype(np.float64))
        ours = [route(nodes, x) for x in slot["x"]]
        self.assertEqual(ours, list(model.apply(slot["x"])))
        self.assertEqual(stats["leaves"], 8)

    def test_random_partition_and_legality(self):
        slots = [synthetic_slot(s, mask=m) for s, m in ((2, 0b11), (3, 0b1010011))]
        records, stats = fit.fit_forest(slots, 10, 25, 3, "random")
        self.assertEqual(stats["partition"], "random")
        for (key, mask, nodes), slot in zip(records, slots):
            self.assertEqual(mask, slot["mask"])  # every bucket of a public node shares its legal mask
            leaves = [route(nodes, x) for x in slot["x"]]
            counts = np.unique(leaves, return_counts=True)[1]
            self.assertGreaterEqual(counts.min(), 25)
        self.assertEqual(stats["leaves"], 10)

    def test_forest_file_layout(self):
        slot = synthetic_slot(4)
        records, _ = fit.fit_forest([slot], 4, 30, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest_2.bin"
            fit.write_forest(path, records)
            data = io.BytesIO(path.read_bytes())
        self.assertEqual(fit._text(data), b"TARAFOREST1")
        self.assertEqual(struct.unpack("<i", data.read(4))[0], 0)
        self.assertEqual(struct.unpack("<I", data.read(4))[0], 1)
        self.assertEqual(fit._text(data), slot["key"])


class TaraRunTest(unittest.TestCase):
    def train(self, path, hands, extra=(), resume=False):
        args = ["train", "--run", str(path), "--hands", str(hands), "--n0", "40", "--particles", "4",
                "--panel-roots", "4", "--panel-every", "40", "--checkpoint-every", "16", "--probe-delta", "8",
                "--probe-episodes", "6", "--leaf-budget", "600", "--min-samples-leaf", "20", "--reserve-gib", "0",
                *extra]
        if resume:
            args = ["train", "--run", str(path), "--hands", str(hands), "--resume"]
        self.assertEqual(run.train(run.parser().parse_args(args)), 0)

    def test_parity_resume_probes_and_lbr(self):
        run.build(check=True)
        run.DATA.mkdir(parents=True, exist_ok=True)
        validated.DATA.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run.DATA, prefix="test_") as directory, \
                tempfile.TemporaryDirectory(dir=validated.DATA, prefix="test_tara_parity_") as old_dir:
            directory = Path(directory)
            whole, split = directory / "whole", directory / "split"
            self.train(whole, 200)
            self.train(split, 120)  # 120 = 40 + 80 is a boundary: forest 3 is fitted and waits
            self.train(split, 200, resume=True)
            for name in ("per_hand.csv", "panel.csv", "forests.csv", "probes.csv"):
                self.assertEqual(rows(whole / name), rows(split / name), name)
            self.assertEqual((whole / "models" / "model_200.bin").read_bytes(),
                             (split / "models" / "model_200.bin").read_bytes())
            fits = (whole / "fits.jsonl").read_text().splitlines()
            self.assertEqual(len(fits), 2)  # boundaries 40 and 120 (280 > 200)

            # zero-initialized new tree: current, sum-average and newest-mature-tree average right after the
            # addition equal those at the boundary
            probes = rows(whole / "probes.csv")
            for boundary in ("40", "120"):
                at = [r for r in probes if r["boundary"] == boundary and r["tag"] == "at"]
                added = [r for r in probes if r["boundary"] == boundary and r["tag"] == "added"]
                self.assertTrue(at)
                strip = lambda r: {k: v for k, v in r.items() if k != "tag"}
                self.assertEqual([strip(r) for r in at], [strip(r) for r in added])

            old = Path(old_dir) / "run"
            args = validated.parser().parse_args([
                "train", "--run", str(old), "--hands", "200", "--seed", "222", "--particles", "4", "--panel-roots", "4",
                "--panel-every", "40", "--checkpoint-every", "16", "--reserve-gib", "0"])
            self.assertEqual(validated.train(args), 0)
            tara_rows, dt_rows = rows(whole / "per_hand.csv"), rows(old / "per_hand.csv")
            for key in ("root_seed", "kmeans_current", "kmeans_average", "kmeans_states"):
                self.assertEqual([r[key] for r in tara_rows], [r[key] for r in dt_rows], key)

            output = whole / "lbr_test"
            run.evaluate_lbr(run.parser().parse_args([
                "lbr", "--run", str(whole), "--output", str(output), "--seeds", "5", "--pairs", "2",
                "--particles", "4"]))
            self.assertTrue((output / "summary.json").exists())

            # B0-sched (k-means partition with TARA's schedule) and B0-single: snapshots at every schedule
            # hand, the same k-means arm, forest kind 2 throughout.
            for mode, forests in (("kmeans_sched", 3), ("kmeans_partition", 1)):
                path = directory / mode
                self.train(path, 200, ("--mode", mode))
                models = sorted(int(m.stem.split("_")[1]) for m in (path / "models").glob("model_*.bin"))
                self.assertEqual(models, [40, 120, 200])
                b0_rows = rows(path / "per_hand.csv")
                for key in ("root_seed", "kmeans_current", "kmeans_average", "kmeans_states"):
                    self.assertEqual([r[key] for r in b0_rows], [r[key] for r in dt_rows], key)
                self.assertEqual(int(b0_rows[-1]["forests"]), forests)
                kinds = {r["kind"] for r in rows(path / "forests.csv")}
                self.assertEqual(kinds, {"2"})
                self.assertFalse(list(path.glob("buffer_*.bin")))


if __name__ == "__main__":
    unittest.main()
