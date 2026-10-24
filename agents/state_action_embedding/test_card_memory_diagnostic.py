"""Run: python -m unittest agents.state_action_embedding.test_card_memory_diagnostic -v."""

import io
import unittest

from . import card_memory_diagnostic as d
from . import observation_reconstruction as r


class CardMemoryTests(unittest.TestCase):
    def test_readout_threshold_is_distinct_from_ranking(self):
        target = d.np.zeros((1, 52), d.np.float32)
        target[0, 3] = 1
        logits = d.np.ones((1, 52), d.np.float32)
        logits[0, 3] = 5
        score = d.card_scores(logits, target, d.np.full(52, 51.0))
        self.assertEqual(score["exact"][0], 0)
        self.assertEqual(score["topk_exact"][0], 1)
        self.assertEqual(score["false_positive"][0], 51)
        self.assertEqual(score["deweighted_exact"][0], 1)

    def test_prefix_alignment_card_movement_and_causal_pass(self):
        r.torch.set_num_threads(1)
        e = r.events
        stream = [(e.ANTE, p, 0, 0, 1, 1) for p in (0, 1)]
        for card in range(1, 5):
            stream.extend([(e.PRIVATE, 0, card, 0, 1, 0), (e.PRIVATE, 1, 0, 0, 1, 0)])
        stream.extend(
            [
                (e.DISCARD, 0, 1, 0, 2, 0),
                (e.REVEAL, 0, 2, 0, 2, 0),
                (e.DISCARD, 1, 0, 0, 2, 0),
                (e.REVEAL, 1, 5, 0, 2, 0),
                (e.PUBLIC, 0, 6, 0, 3, 0),
                (e.PUBLIC, 1, 7, 0, 3, 0),
                (e.PUBLIC, 0, 8, 0, 4, 0),
                (e.PUBLIC, 1, 9, 0, 4, 0),
            ]
        )
        before = d.np.asarray(stream, d.np.int16)
        stream.append((e.BET, 0, 0, 1, 4, 0))
        buffer = io.BytesIO()
        d.np.savez(
            buffer,
            action=d.np.asarray(stream, d.np.int16),
            action_offsets=[0, len(stream)],
            hand=[0],
            split=[0],
            state=e.reconstruct_prefix(before)[0].reshape(1, -1),
            state_offsets=[0, 1],
        )
        buffer.seek(0)
        normalizer = dict(
            paid=dict(mean=0.0, std=1.0), mean=d.np.zeros(85), scale=d.np.ones(85)
        )
        data = d.streams(buffer, normalizer)
        initial = data["markers"]["initial_deal"][0]
        discard = data["markers"]["own_discard"][0]
        setup = data["markers"]["setup_done"][0]
        self.assertEqual(data["y"][initial, :52].sum(), 4)
        self.assertEqual(data["y"][discard, :52].sum(), 3)
        self.assertEqual(data["y"][setup, :52].sum(), 2)
        self.assertEqual(data["y"][discard, 104], 1)
        self.assertEqual(data["age"][discard], 0)
        self.assertEqual(data["ends"][data["markers"]["5_first"]][0], len(before))
        self.assertEqual(data["audited"], 1)
        d.frozen_logits(r.ReconstructionModel("transformer", 32).eval(), data)


if __name__ == "__main__":
    unittest.main()
