"""Run: python -m unittest agents.state_action_embedding.test_ppo_self_play -v."""

import contextlib
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import bc_ppo as ppo
from . import ppo_self_play as experiment


class SelfPlayChecks(unittest.TestCase):
    def setUp(self):
        ppo.torch.set_num_threads(1)
        ppo.torch.manual_seed(11)

    def test_frozen_opponents_and_on_policy_data(self):
        model = ppo.ActorCritic(8, 1).eval()
        pool = {0: experiment.frozen_copy(model)}
        original = pool[0].action_head.weight.detach().clone()
        for update in (1, 2, 3):
            experiment.add_snapshot(pool, update, model, 2)
        self.assertEqual(list(pool), [0, 3])
        self.assertFalse(any(p.requires_grad for p in pool[3].parameters()))
        with patch.object(
            ppo, "HeuristicPokerAgent", side_effect=AssertionError("Not self-play")
        ):
            data, profits = ppo.rollout(model, 4, 8123, 7123, opponent=pool[3])
            mirrored = ppo.evaluate_policy(model, 4, 711, 811, opponent=pool[3])
        ppo.np.testing.assert_array_equal(mirrored.sum(1), 0)
        self.assertEqual(len(profits), 4)
        inputs = ppo.batch(data, ppo.np.arange(len(data["action"])), "cpu")
        with ppo.torch.no_grad():
            distribution, _ = model.distribution_value(*inputs)
            ppo.torch.testing.assert_close(
                distribution.log_prob(ppo.torch.as_tensor(data["action"])),
                ppo.torch.as_tensor(data["old_log_prob"]),
                rtol=1e-5,
                atol=1e-5,
            )
        ppo.update_ppo(
            model,
            ppo.torch.optim.Adam(model.parameters(), lr=1e-4),
            data,
            ppo.np.random.default_rng(4),
            SimpleNamespace(ppo_epochs=1, batch=16),
            "cpu",
        )
        self.assertFalse(ppo.torch.equal(model.action_head.weight, original))
        for opponent in pool.values():
            ppo.torch.testing.assert_close(opponent.action_head.weight, original)
            self.assertTrue(all(p.grad is None for p in opponent.parameters()))

    def test_run_database_and_held_out_evaluation(self):
        with TemporaryDirectory(
            dir=Path(__file__).resolve().parent / "data"
        ) as temporary, contextlib.redirect_stdout(StringIO()):
            run = Path(temporary) / "run"
            args = [
                "train",
                "--out-dir",
                str(run),
                "--updates",
                "2",
                "--rollout-hands",
                "4",
                "--eval-every",
                "1",
                "--eval-pairs",
                "2",
                "--snapshot-every",
                "1",
                "--pool-size",
                "2",
                "--threads",
                "1",
                "--dim",
                "8",
                "--layers",
                "1",
                "--ppo-epochs",
                "1",
                "--batch",
                "16",
            ]
            experiment.main(args)
            with self.assertRaises(FileExistsError):
                experiment.main(args)
            initial, _ = ppo.load_policy(run / "initial.pt")
            final, metadata = ppo.load_policy(run / "last.pt")
            self.assertEqual(metadata["selected_update"], 2)
            self.assertFalse(
                ppo.torch.equal(initial.action_head.weight, final.action_head.weight)
            )
            original_benchmark = experiment.benchmark

            def interrupt_final(*values):
                if values[3] == "test" and values[4] > 0:
                    raise KeyboardInterrupt
                return original_benchmark(*values)

            test_args = ["test", "--out-dir", str(run), "--test-pairs", "2"]
            with patch.object(experiment, "benchmark", side_effect=interrupt_final):
                with self.assertRaises(KeyboardInterrupt):
                    experiment.main(test_args)
            experiment.main(test_args)
            with self.assertRaises(FileExistsError):
                experiment.main(["test", "--out-dir", str(run), "--test-pairs", "2"])
            with contextlib.closing(sqlite3.connect(run / "metrics.sqlite")) as db:
                rows = db.execute(
                    "SELECT update_index, train_hands, opponent_update FROM updates"
                ).fetchall()
                self.assertEqual([r[1] for r in rows], [4, 8])
                self.assertTrue(all(opponent < update for update, _, opponent in rows))
                self.assertEqual(
                    db.execute("SELECT count(*) FROM evaluations").fetchone()[0], 15
                )
                self.assertEqual(
                    db.execute("SELECT count(*) FROM pair_returns").fetchone()[0], 30
                )
                for split in ("monitor", "test"):
                    values = experiment.stored_returns(db, split, 0, "initial")
                    ppo.np.testing.assert_array_equal(values.sum(1), 0)
                for update, opponent, delta in db.execute(
                    "SELECT update_index, opponent, delta_chips FROM evaluations WHERE split='monitor'"
                ):
                    raw = experiment.stored_returns(db, "monitor", update, opponent)
                    base = experiment.stored_returns(db, "monitor", 0, opponent)
                    self.assertAlmostEqual(delta, float((raw - base).mean()))
                self.assertEqual(
                    db.execute("PRAGMA integrity_check").fetchone()[0], "ok"
                )


if __name__ == "__main__":
    unittest.main()
