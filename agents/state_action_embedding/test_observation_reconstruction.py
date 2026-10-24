"""Run: python -m unittest agents.state_action_embedding.test_observation_reconstruction -v."""

import io
import json
from pathlib import Path
import tempfile
import unittest

from . import observation_reconstruction as r


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(dir=(r.FAMILY / "data").resolve())
        path = Path(cls.directory.name) / "small.npz"
        r.collect_fresh(path, 8, 815200000)
        with r.np.load(path) as source:
            raw = dict(source)
        raw["split"][:] = 0
        raw["split"][-4:] = 1
        r.np.savez_compressed(path, **raw)
        cls.path, cls.raw = path, raw
        cls.data = r.load_data(path)

    @classmethod
    def tearDownClass(cls):
        assert (
            Path(cls.directory.name)
            .resolve()
            .is_relative_to((r.FAMILY / "data").resolve())
        )
        cls.directory.cleanup()

    def test_audit_alignment_and_split(self):
        self.assertEqual(self.data["audit_error"], 0)
        self.assertTrue(r.np.isin(self.data["tokens"][:, 0], r.ACTION_KINDS).all())
        for hand in r.np.unique(self.data["hand"]):
            self.assertEqual(
                len(r.np.unique(self.data["split"][self.data["hand"] == hand])), 1
            )
        for i in range(len(self.raw["hand"])):
            lo, hi = self.raw["action_offsets"][i : i + 2]
            stream = self.raw["action"][lo:hi]
            bets = r.np.flatnonzero(stream[:, 0] == r.events.BET)
            for row, bet in enumerate(bets, start=self.raw["state_offsets"][i]):
                expected = stream[:bet][r.np.isin(stream[:bet, 0], r.ACTION_KINDS)]
                start = self.data["starts"][i]
                actual = self.data["tokens"][start : start + self.data["ends"][row]]
                r.np.testing.assert_array_equal(actual, expected)

    def test_future_and_padding_invariance_and_reload(self):
        model = r.ReconstructionModel().eval()
        x, lengths, _ = r.batch(self.data, r.np.array([0, 1, 2]))
        with r.torch.no_grad():
            expected, z = model(x, lengths)
            altered = r.torch.cat((x.clone(), r.torch.randn(len(x), 3, 85)), 1)
            for i, length in enumerate(lengths):
                altered[i, length:] = r.torch.randn_like(altered[i, length:]) * 100
            actual = model(altered, lengths)[0]
        r.torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
        self.assertEqual(z.shape[1], 64)
        buffer = io.BytesIO()
        r.torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = r.ReconstructionModel().eval()
        restored.load_state_dict(r.torch.load(buffer, weights_only=True))
        r.torch.testing.assert_close(
            model(x, lengths)[0], restored(x, lengths)[0], atol=0, rtol=0
        )

    def test_mask_and_backward(self):
        for kind in (r.events.PRIVATE, r.events.DISCARD):
            one = r.events.viewed_event(kind, 1, 1, 0, 4, 0, 0)
            two = r.events.viewed_event(kind, 1, 52, 0, 4, 0, 0)
            self.assertEqual(one, two)
        model = r.ReconstructionModel()
        x, lengths, target = r.batch(self.data, r.np.arange(8))
        stats = r.statistics(self.data["y"])
        output, _ = model(x, lengths)
        loss = r.losses(output, target, stats).mean()
        loss.backward()
        self.assertTrue(r.torch.isfinite(loss))
        self.assertGreater(float(model.backbone.project.weight.grad.norm()), 0)
        self.assertTrue(
            all(
                r.torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_reconstruction_loss_uses_logits(self):
        target = self.data["y"][:8]
        stats = r.statistics(self.data["y"])
        output = r.torch.where(target > 0.5, 15.0, -15.0)
        output[:, 221:233] = (target[:, 221:233] - stats["mean"]) / stats["scale"]
        self.assertLess(float(r.losses(output, target, stats).max()), 1e-4)

    def test_resume_optimizer_and_sampler(self):
        model = r.ReconstructionModel()
        optimizer = r.torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = r.np.random.default_rng(111)
        stats = r.statistics(self.data["y"])
        hands = r.np.unique(self.data["hand"][self.data["split"] == 0])

        def step(network, opt, sampler):
            rows = r.sampled_rows(self.data, hands, sampler, 8)
            x, lengths, target = r.batch(self.data, rows)
            loss = r.losses(network(x, lengths)[0], target, stats).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            r.nn.utils.clip_grad_norm_(network.parameters(), 1)
            opt.step()
            return rows

        step(model, optimizer, rng)
        state = r.checkpoint(model, stats, self.data, 1)
        state.update(
            optimizer=optimizer.state_dict(),
            sampler_rng=json.dumps(rng.bit_generator.state),
            torch_rng=r.torch.get_rng_state(),
        )
        buffer = io.BytesIO()
        r.torch.save(state, buffer)
        expected_rows = step(model, optimizer, rng)
        buffer.seek(0)
        saved = r.torch.load(buffer, weights_only=True)
        resumed = r.ReconstructionModel(saved["kind"], saved["width"])
        resumed.load_state_dict(saved["model"])
        opt = r.torch.optim.Adam(resumed.parameters(), lr=3e-4)
        opt.load_state_dict(saved["optimizer"])
        sampler = r.np.random.default_rng()
        sampler.bit_generator.state = json.loads(saved["sampler_rng"])
        r.torch.set_rng_state(saved["torch_rng"])
        r.np.testing.assert_array_equal(step(resumed, opt, sampler), expected_rows)
        for left, right in zip(model.parameters(), resumed.parameters()):
            r.torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
