"""Run: python -m unittest agents.autoencoder_abstraction.test_experiment -v."""

from io import BytesIO
import unittest

import numpy as np
import torch

from .experiment import AutoEncoder, ROW, VQVAE, assign, bucket_keys, inputs
from .comparison import gap_ratios, log_trend


class AbstractionChecks(unittest.TestCase):
    def test_paired_lbr_summary(self):
        from .reevaluate_lbr import summarize

        stats = summarize(np.ones((2, 4)), bootstrap=100)
        self.assertEqual(stats["mean"], 1)
        self.assertEqual(stats["bootstrap_ci95"], [1, 1])
        self.assertEqual(stats["samples"], 8)
        self.assertEqual(stats["seed_means"], [1, 1])
        with self.assertRaises(ValueError):
            summarize(np.array([[np.nan, 1]]))

    def test_log_trend_and_exact_baseline_budgets(self):
        metrics = [dict(hands=h, mean=(h / 100000) ** -.5)
                   for h in (100000, 200000, 400000, 800000)]
        trend = log_trend(metrics)
        self.assertAlmostEqual(trend["slope"], -.5)
        self.assertAlmostEqual(trend["factor_per_doubling"], 2 ** -.5)
        self.assertIsNone(log_trend(metrics[:2])["slope"])
        later = [dict(hands=g["hands"] * 10, mean=g["mean"]) for g in metrics]
        self.assertAlmostEqual(log_trend(metrics + later, 1000000, 10000000)["slope"], -.5)
        baseline = [dict(hands=100000, mean=2), dict(hands=200000, mean=0),
                    dict(hands=300000, mean=1)]
        self.assertEqual(gap_ratios(metrics, baseline), [dict(hands=100000, ratio=.5)])
        with self.assertRaises(ValueError):
            log_trend(metrics + metrics[:1])

    def test_compact_scope_and_history(self):
        self.assertEqual(ROW.itemsize, 148)
        rows = np.zeros(3, ROW)
        rows["street"] = 5
        rows["mask"] = 129
        full = inputs(rows, "infoset")
        self.assertEqual(full.shape, (3, 1832))
        self.assertEqual(inputs(rows, "power").shape, (3, 18))
        rows["action"] = 7
        rows["outcome"] = 1000
        np.testing.assert_array_equal(inputs(rows, "infoset"), full)
        rows["history"][1, 0] = 1
        self.assertFalse(np.array_equal(inputs(rows, "infoset")[1], full[1]))
        self.assertTrue(np.array_equal(inputs(rows, "power")[0], inputs(rows, "power")[1]))

    def test_vq_gradients_and_nearest_codes(self):
        torch.manual_seed(11)
        model = VQVAE(18, 16, 4, 8)
        x = torch.randn(12, 18)
        z = model.encoder(x)
        _, ids = model.quantize(z)
        self.assertTrue(torch.equal(ids, torch.cdist(z, model.codebook.weight).argmin(1)))
        loss, reconstruction, _ = model.loss(x)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in (model.encoder[0].weight, model.decoder[0].weight, model.codebook.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertGreaterEqual(float(loss), float(reconstruction))

    def test_checkpoint_and_hard_action_partition(self):
        rows = np.zeros(3, ROW)
        rows["street"] = 5
        rows["mask"] = [129, 192, 129]
        rows["actor"] = [0, 0, 1]
        keys = bucket_keys(rows, [2, 2, 2], [b"history"] * 3)
        self.assertEqual(len(set(keys)), 3)
        with self.assertRaises(ValueError):
            bucket_keys(rows, [2, 2, 2])
        x = np.random.default_rng(3).normal(size=(12, 18)).astype(np.float32)
        for method in ("raw_kmeans", "ae_kmeans", "vqvae"):
            architecture = dict(features=18, hidden=16, latent=4)
            if method == "vqvae":
                architecture["codes"] = 8
            model = (VQVAE if method == "vqvae" else AutoEncoder)(**architecture)
            checkpoint = dict(method=method, features=18, architecture=architecture,
                model=model.state_dict(), centers=torch.randn(8, 18 if method == "raw_kmeans" else 4))
            with BytesIO() as buffer:
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, weights_only=True)
            np.testing.assert_array_equal(assign(checkpoint, x), assign(loaded, x))
            with self.assertRaises(ValueError):
                assign(loaded, np.zeros((1, 19), np.float32))


if __name__ == "__main__":
    unittest.main()
