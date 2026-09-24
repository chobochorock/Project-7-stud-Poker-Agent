"""Run: python -m unittest agents.state_action_embedding.test_action_path_embedding -v."""

import contextlib
import copy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from .action2vec import PRIVATE, viewed_event
from .action_path_embedding import (
    PathEncoder,
    action_events,
    assert_relations,
    branches,
    collect,
    load_path_encoder,
    make_root,
    np,
    observations,
    relation_choices,
    relation_panels,
    sample_relations,
    template_key,
    torch,
)


@contextlib.contextmanager
def temporary_output():
    # Windows sandbox cannot reopen Python 3.12's mode-0700 mkdtemp directories.
    parent = (Path(__file__).parent / "data").resolve()
    path = parent / ("test_action_path_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        assert path.resolve().is_relative_to(parent)
        for file in path.iterdir():
            file.unlink()
        path.rmdir()


class ActionPathChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_atomic_private_tokens_and_exact_unequal_paths(self):
        self.assertEqual(
            viewed_event(PRIVATE, 1, 1, 0, 1, 0, 0),
            viewed_event(PRIVATE, 1, 52, 0, 1, 0, 0),
        )
        with contextlib.redirect_stdout(StringIO()):
            root = make_root(50260924)
            paths = list(branches(root))
        self.assertGreater(len(paths), 2)
        initial = observations(root)
        changed = copy.deepcopy(root)
        changed.players[1].hidden_cards[0], changed.deck.cards[0] = (
            changed.deck.cards[0],
            changed.players[1].hidden_cards[0],
        )
        np.testing.assert_array_equal(initial[0], observations(changed)[0])
        found = False
        for game_a, script_a, _ in paths:
            for game_b, script_b, _ in paths:
                if len(script_a) == len(script_b):
                    continue
                if np.array_equal(observations(game_a), observations(game_b)):
                    self.assertNotEqual(game_a.betting_history, game_b.betting_history)
                    self.assertEqual(observations(game_a).shape, (2, 235))
                    found = True
                    break
            if found:
                break
        self.assertTrue(found, "No genuine different-length exact-observation pair")

    def test_dataset_grouping_negatives_and_template_exclusion(self):
        with temporary_output() as directory:
            args = SimpleNamespace(
                out_dir=Path(directory), roots=10, data_seed=50260924
            )
            with contextlib.redirect_stdout(StringIO()):
                collect(args)
            with np.load(args.out_dir / "paths.npz", allow_pickle=False) as source:
                data = dict(source)
            for hand in np.unique(data["hand"]):
                self.assertEqual(len(np.unique(data["split"][data["hand"] == hand])), 1)
            choices = relation_choices(data, 0, train=True)
            panel = sample_relations(choices, np.random.default_rng(5), 32)
            self.assertFalse(data["heldout"][panel].any())
            assert_relations(data, panel)
            for panel in relation_panels(data, args.data_seed).values():
                assert_relations(data, panel[:20])
            templates = {
                template_key(t[:n])
                for t, n, h in zip(data["tokens"], data["length"], data["heldout"])
                if not h
            }
            self.assertTrue(
                all(
                    template_key(t[:n]) not in templates
                    for t, n, h in zip(data["tokens"], data["length"], data["heldout"])
                    if h
                )
            )

    def test_order_padding_gradient_and_reload(self):
        torch.manual_seed(11)
        model = PathEncoder(20, 8)
        tokens = torch.tensor([[2, 3, 4, 0], [4, 3, 2, 0], [2, 3, 4, 10]])
        lengths = torch.tensor([3, 3, 3])
        encoded = model(tokens, lengths)
        torch.testing.assert_close(encoded[0], encoded[2])
        self.assertFalse(torch.allclose(encoded[0], encoded[1]))
        loss = (encoded[0] - encoded[1]).square().sum()
        loss.backward()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0)
        with temporary_output() as directory:
            path = Path(directory) / "model.pt"
            torch.save(dict(model=model.state_dict(), vocabulary=20, dim=8), path)
            loaded = load_path_encoder(path)
            torch.testing.assert_close(encoded, loaded(tokens, lengths))


if __name__ == "__main__":
    unittest.main()
