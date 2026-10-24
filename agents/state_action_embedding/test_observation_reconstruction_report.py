"""Run with sklearn Python: python -m unittest agents.state_action_embedding.test_observation_reconstruction_report -v."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .observation_reconstruction_report import cluster_probe


class ClusterProbeTests(unittest.TestCase):
    def test_same_partition_budget_and_reload(self):
        root = (Path(__file__).parent / "data").resolve()
        temporary = tempfile.TemporaryDirectory(dir=root)
        try:
            run = Path(temporary.name)
            directory = run / "transformer" / "seed11"
            directory.mkdir(parents=True)
            truth = np.zeros((20, 235), np.float32)
            truth[np.arange(20), np.arange(20)] = 1
            truth[:, [208, 212, 213, 220]] = 1
            truth[:, 221:230] = np.arange(20)[:, None] / 20
            payload = dict(
                truth=truth,
                predicted=truth * 0.95,
                latent=truth[:, :32],
                hand=np.arange(20),
                rows=np.arange(20),
                lengths=np.full(20, 25),
            )
            for split in ("train", "test_lowfold"):
                np.savez_compressed(directory / f"{split}_latents.npz", **payload)
            results = cluster_probe(run, [4])
            self.assertEqual(len(results), 3)
            for row in results:
                self.assertEqual(row["allocated_buckets"], 4)
                self.assertEqual(row["partition_groups"], 1)
                self.assertEqual(row["row_coverage"], 1)
                with np.load(
                    directory / f"kmeans_{row['condition']}_4.npz", allow_pickle=False
                ) as saved:
                    self.assertTrue(np.isfinite(saved["centers"]).all())
                    self.assertTrue((saved["assignments"] >= 0).all())
                    self.assertEqual(saved["prototypes"].shape, (4, 235))
        finally:
            assert Path(temporary.name).resolve().is_relative_to(root)
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
