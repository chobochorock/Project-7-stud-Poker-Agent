"""Atomic action SGNS and exact-current-observation endpoint path embedding.

Usage (project root; Python 3.12 runtime, existing PyTorch dependencies):
    python -m agents.state_action_embedding.action_path_embedding collect --out-dir RUN
    python -m agents.state_action_embedding.action_path_embedding train --out-dir RUN
Input: existing low-fold 10k corpus; legal branches of custom seven-poker v3.
Output: packed paths, train-only vocabularies, checkpoints and retrieval metrics.
Limits: equal 235D observations are NOT equal perfect-recall information sets.
No hidden-card targets, reward labels, state encoder, policy or CFR training.
See ACTION_PATH_EMBEDDING.md for definitions, budgets and held-out templates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

from .action2vec import ANTE, BET, DISCARD, PRIVATE, PUBLIC, REVEAL, EventGame
from .experiment import (
    BETTING_ACTIONS,
    CARD_IDS,
    ROOT,
    UniformAgent,
    encode_event,
    np,
    torch,
    nn,
    F,
    write_json,
)
from .separate_skipgram import (
    SkipGramTeacher,
    sample_contexts,
    score_metrics,
    sgns_loss,
    state_features,
    unique_rows,
)


ACTION_KINDS = (ANTE, PRIVATE, PUBLIC, DISCARD, REVEAL, BET)
DEFAULT_CORPUS = ROOT / "agents/state_action_embedding/data/separate_20260922"


class NeedAction(Exception):
    def __init__(self, legal):
        self.legal = tuple(legal)


class Script:
    """Let the existing betting loop enumerate branches without copying its rules."""

    def __init__(self, actions=()):
        self.actions = actions
        self.position = 0

    def choose(self, legal):
        if self.position == len(self.actions):
            raise NeedAction(legal)
        action = self.actions[self.position]
        if action not in legal:
            raise AssertionError(f"Illegal scripted action: {action}, {legal}")
        self.position += 1
        return action


class ScriptAgent(UniformAgent):
    def __init__(self, name, seed, script):
        super().__init__(name, seed)
        self.script = script

    def choose_action(self, state, valid_actions):
        return self.script.choose(valid_actions)


def agents_for(script, seed=0):
    return {f"p{i}": ScriptAgent(f"p{i}", seed + i, script) for i in range(2)}


def make_root(seed: int) -> EventGame:
    """Stop the real hand loop at its first fifth-street decision."""
    random.seed(seed)
    game = EventGame()
    try:
        game.play_hand(agents_for(Script(), seed))
    except NeedAction:
        assert game.street == "5th" and not game.betting_history
        return game
    raise AssertionError("Expected a fifth-street decision")


def observations(game) -> np.ndarray:
    """Only valid at a betting round's first decision, after its reset."""
    actor = game.players[game._first_bettor_index(set(game.players))]
    legal = game.get_valid_actions(actor)
    return np.stack(
        [
            state_features(
                encode_event(game.get_ai_state(viewer, legal), "CHECK", viewer is actor)
            )
            for viewer in game.players
        ]
    )


def action_events(events) -> np.ndarray:
    events = np.asarray(events, dtype=np.int16).reshape(-1, 6)
    return events[np.isin(events[:, 0], ACTION_KINDS)]


def branches(root):
    """All non-fold fifth-street paths, ending at the real sixth-street reset."""
    pending = [()]
    visited = 0
    while pending:
        prefix = pending.pop()
        game = copy.deepcopy(root)
        script = Script(prefix)
        visited += 1
        if visited > 2000:
            raise RuntimeError("Unexpected branch growth; do not silently truncate")
        try:
            game.play_betting_round(agents_for(script))
        except NeedAction as needed:
            pending.extend(prefix + (a,) for a in needed.legal if a != "FOLD")
            continue
        assert script.position == len(prefix)
        assert all(p.can_act() for p in game.players)
        assert sum(p.chips for p in game.players) + game.pot == 2000
        game.street = "6th"
        game.deal_cards_to_active(is_public=True)
        try:
            game.play_betting_round(agents_for(Script()))
        except NeedAction:
            pass
        else:
            raise AssertionError("Expected sixth-street boundary")
        yield game, prefix, visited


def template_key(tokens) -> bytes:
    """Hold out whole betting patterns globally, excluding accidental card IDs."""
    bets = tokens[tokens[:, 0] == BET]
    return np.ascontiguousarray(bets[:, [1, 3, 4, 5]]).tobytes()


def token_text(row) -> str:
    kind, subject, card, action, street, paid = map(int, row)
    actor = ("self", "opponent", "chance")[subject]
    cards = tuple(CARD_IDS)
    target = cards[card - 1] if card else "MASK"
    if kind in (PRIVATE, PUBLIC):
        name = "DealPrivate" if kind == PRIVATE else "DealPublic"
        return f"{name}({actor},{target})"
    if kind in (DISCARD, REVEAL):
        return f"{('Discard' if kind == DISCARD else 'Open')}({actor},{target})"
    if kind == BET:
        return f"{BETTING_ACTIONS[action - 1]}({actor},paid={paid})"
    return f"Ante({actor},paid={paid})"


def collect(args) -> None:
    if args.roots < 10:
        raise ValueError("At least 10 roots for grouped 80/10/10 splitting")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "paths.npz"
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    split = np.zeros(args.roots, dtype=np.int8)
    order = np.random.default_rng(args.data_seed).permutation(args.roots)
    split[order[int(args.roots * 0.8) : int(args.roots * 0.9)]] = 1
    split[order[int(args.roots * 0.9) :]] = 2
    sequences, left, right, roots, hands, partitions, heldout, group_ids = (
        [] for _ in range(8)
    )
    group_map, examples, queries = {}, [], 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in range(args.roots):
            root = make_root(args.data_seed + hand)
            start = observations(root)
            last_queries = 0
            for game, prefix, last_queries in branches(root):
                end = observations(game)
                for viewer in range(2):
                    tokens = action_events(
                        game.events[viewer][len(root.events[viewer]) :]
                    )
                    key = (hand, viewer, start[viewer].tobytes(), end[viewer].tobytes())
                    group = group_map.setdefault(key, len(group_map))
                    sequences.append(tokens)
                    left.append(start[viewer])
                    right.append(end[viewer])
                    roots.append(hand * 2 + viewer)
                    hands.append(hand)
                    partitions.append(split[hand])
                    heldout.append(
                        int.from_bytes(
                            hashlib.sha256(template_key(tokens)).digest()[:4], "little"
                        )
                        % 5
                        == 0
                    )
                    group_ids.append(group)
                    if hand == 0:
                        examples.append(
                            {
                                "index": len(sequences) - 1,
                                "viewer": viewer,
                                "group": group,
                                "events": [token_text(t) for t in tokens],
                                "script": prefix,
                            }
                        )
            queries += last_queries
    lengths = np.asarray([len(s) for s in sequences], np.int16)
    tokens = np.zeros((len(sequences), int(lengths.max()), 6), np.int16)
    for i, sequence in enumerate(sequences):
        tokens[i, : len(sequence)] = sequence
    data = dict(
        tokens=tokens,
        length=lengths,
        left=np.asarray(left),
        right=np.asarray(right),
        root=np.asarray(roots),
        hand=np.asarray(hands),
        split=np.asarray(partitions, np.int8),
        heldout=np.asarray(heldout),
        group=np.asarray(group_ids),
    )
    np.savez_compressed(output, **data)
    mismatched_length_groups = sum(
        len(set(lengths[np.asarray(group_ids) == g])) > 1 for g in range(len(group_map))
    )
    write_json(
        args.out_dir / "path_dataset.json",
        {
            "game": "custom seven-poker v3 heads-up cash; ante1; stack1000",
            "seed": args.data_seed,
            "roots": args.roots,
            "paths": len(sequences),
            "groups": len(group_map),
            "cross_length_groups": mismatched_length_groups,
            "length_histogram": {
                str(n): int((lengths == n).sum()) for n in np.unique(lengths)
            },
            "split_roots": [int((split == k).sum()) for k in range(3)],
            "heldout_template_fraction": float(np.mean(heldout)),
            "branch_replay_calls": queries,
            "seconds": time.perf_counter() - started,
            "rules_version": 3,
            "left": "all 235 current-observation fields at first fifth-street decision",
            "right": "all 235 current-observation fields at first sixth-street decision",
            "equality": "bit-exact float32 arrays; no field removal, distance threshold or bucketing",
            "limitations": [
                "not full history/information-set equivalence",
                "no folds or terminal paths",
                "fixed per-root deck; only fifth-to-sixth-street branches",
                "cards shared across same-root candidates; endpoint difference is chip payment",
                "templates exclude cards and are held out globally from path training",
            ],
            "source_sha256": source_hashes(),
        },
    )
    write_json(args.out_dir / "examples.json", examples)
    print(
        f"paths={len(sequences)}, groups={len(group_map)}, cross-length groups={mismatched_length_groups}",
        flush=True,
    )


def source_hashes():
    paths = [
        Path(__file__),
        ROOT / "agents/state_action_embedding/action2vec.py",
        ROOT / "agents/state_action_embedding/separate_skipgram.py",
        ROOT / "environments/seven_stud/poker_env.py",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def corpus_pairs(corpus, window):
    """Symmetric neighboring atomic actions, not next-two BET-only contexts."""
    with np.load(corpus / "separate_raw.npz", allow_pickle=False) as loaded:
        data = dict(loaded)
    raw, partitions, offsets = [], [], [0]
    for i, (start, end) in enumerate(
        zip(data["action_offsets"][:-1], data["action_offsets"][1:])
    ):
        sequence = action_events(data["action"][start:end])
        raw.append(sequence)
        partitions.extend([data["split"][i]] * len(sequence))
        offsets.append(offsets[-1] + len(sequence))
    raw, partitions = np.concatenate(raw), np.asarray(partitions)
    vocab, _ = unique_rows(raw[partitions == 0])
    mapping = {tuple(row): i + 2 for i, row in enumerate(vocab)}
    ids = np.asarray([mapping.get(tuple(row), 1) for row in raw], np.int64)
    pairs, splits = [], []
    for start, end in zip(offsets[:-1], offsets[1:]):
        for distance in range(1, window + 1):
            a = np.arange(start, end - distance)
            if not len(a):
                continue
            pairs.extend(
                [
                    np.column_stack([ids[a], ids[a + distance]]),
                    np.column_stack([ids[a + distance], ids[a]]),
                ]
            )
            splits.extend([np.full(len(a), partitions[start])] * 2)
    pairs, splits = np.concatenate(pairs), np.concatenate(splits)
    known = (pairs >= 2).all(1)
    coverage = {str(k): float(known[splits == k].mean()) for k in (0, 1)}
    return vocab, pairs[known], splits[known], coverage


def context_panels(vocab, pairs, split, seed):
    rng = np.random.default_rng(seed + 12345)
    training = pairs[split == 0]
    noise, counts = np.unique(training[:, 1], return_counts=True)
    prob = counts.astype(float) ** 0.75
    prob /= prob.sum()
    by_type = defaultdict(list)
    for identity in noise:
        by_type[tuple(vocab[identity - 2, [0, 1, 4]])].append(identity)
    panels = {}
    for partition, label in ((0, "train"), (1, "val")):
        available = pairs[split == partition]
        selected = available[
            rng.choice(len(available), min(4096, len(available)), replace=False)
        ]
        panels[label] = (
            selected[:, 0],
            sample_contexts(rng, selected[:, 1], noise, prob, 5),
        )
        centers, candidates = [], []
        for center, positive in selected:
            pool = [
                i
                for i in by_type[tuple(vocab[positive - 2, [0, 1, 4]])]
                if i != positive
            ]
            if pool:
                centers.append(center)
                candidates.append(np.r_[positive, rng.choice(pool, 5)])
        panels[label + "_matched"] = (np.asarray(centers), np.asarray(candidates))
    return panels, noise, prob


@torch.no_grad()
def evaluate_sgns(model, panels):
    result = {}
    for name, (centers, candidates) in panels.items():
        if not len(centers):
            result[name] = None
            continue
        metrics = score_metrics(
            model.scores(torch.as_tensor(centers), torch.as_tensor(candidates))
        )
        result[name] = dict(n=len(centers), **metrics)
    return result


def train_sgns(args, out, prepared):
    vocab, pairs, split, coverage = prepared
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SkipGramTeacher(len(vocab) + 2, args.dim)
    with torch.no_grad():
        model.center.weight[:2].zero_()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.02)
    panels, noise, probability = context_panels(vocab, pairs, split, args.data_seed)
    training = pairs[split == 0]
    curve = []
    started = time.perf_counter()
    for step in range(args.sgns_steps + 1):
        if step:
            batch = training[rng.integers(len(training), size=args.batch)]
            contexts = sample_contexts(rng, batch[:, 1], noise, probability, 5)
            loss = sgns_loss(
                model.scores(torch.as_tensor(batch[:, 0]), torch.as_tensor(contexts))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % args.log_every == 0 or step == args.sgns_steps:
            row = dict(step=step, metrics=evaluate_sgns(model, panels))
            curve.append(row)
            print(
                f"seed={args.seed} sgns {step}: val={row['metrics']['val']['retrieval_top1']:.4f}",
                flush=True,
            )
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            numpy_rng=rng.bit_generator.state,
            dim=args.dim,
            vocab=torch.from_numpy(vocab),
            step=args.sgns_steps,
        ),
        out / "sgns.pt",
    )
    write_json(
        out / "sgns_metrics.json",
        dict(
            seed=args.seed,
            coverage=coverage,
            vocabulary=len(vocab),
            pairs={str(k): int((split == k).sum()) for k in (0, 1)},
            window=args.window,
            batch=args.batch,
            seconds=time.perf_counter() - started,
            curve=curve,
            note="SGNS context retrieval; not path equivalence or poker strength",
        ),
    )
    return model.center.weight.detach().clone(), vocab


class PathEncoder(nn.Module):
    """Order-preserving f(a1,...,an); no endpoint or initial-observation input."""

    def __init__(self, vocabulary, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.05)
        with torch.no_grad():
            self.embedding.weight[:2].zero_()
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, tokens, lengths):
        encoded = F.normalize(self.embedding(tokens), dim=-1)
        hidden, _ = self.gru(encoded)
        return F.normalize(hidden[torch.arange(len(tokens)), lengths - 1], dim=-1)


def path_inputs(data, vocab):
    mapping = {tuple(row): i + 2 for i, row in enumerate(vocab)}
    ids = np.zeros(data["tokens"].shape[:2], np.int64)
    unknown, total = 0, 0
    for i, length in enumerate(data["length"]):
        ids[i, :length] = [
            mapping.get(tuple(row), 1) for row in data["tokens"][i, :length]
        ]
        unknown += int((ids[i, :length] == 1).sum())
        total += int(length)
    return (
        torch.from_numpy(ids),
        torch.from_numpy(data["length"].astype(np.int64)),
        unknown / total,
    )


def relation_choices(data, partition, train=False, heldout=False):
    allowed = data["split"] == partition
    if train:
        allowed &= ~data["heldout"]
    roots, groups = defaultdict(list), defaultdict(list)
    for i in np.flatnonzero(allowed):
        roots[int(data["root"][i])].append(i)
        groups[int(data["group"][i])].append(i)
    choices = []
    for i in np.flatnonzero(allowed):
        if heldout and not data["heldout"][i]:
            continue
        positive = [
            j
            for j in groups[int(data["group"][i])]
            if j != i and data["length"][j] != data["length"][i]
        ]
        negatives = [
            j
            for j in roots[int(data["root"][i])]
            if data["group"][j] != data["group"][i]
        ]
        if positive and negatives:
            choices.append((int(i), np.asarray(positive), np.asarray(negatives)))
    return choices


def sample_relations(choices, rng, count):
    if not choices:
        raise ValueError("No exact-endpoint cross-length relations in this split")
    rows = []
    for index in rng.integers(len(choices), size=count):
        anchor, positives, negatives = choices[index]
        rows.append(
            [anchor, int(rng.choice(positives)), *rng.choice(negatives, 5).tolist()]
        )
    return np.asarray(rows, np.int64)


def relation_panels(data, seed):
    rng = np.random.default_rng(seed + 76543)
    output = {}
    for name, split, unseen in (
        ("train", 0, False),
        ("val", 1, False),
        ("test", 2, False),
        ("test_unseen_template", 2, True),
    ):
        choices = relation_choices(data, split, train=split == 0, heldout=unseen)
        output[name] = sample_relations(choices, rng, 2048)
    return output


def assert_relations(data, panel):
    for row in panel:
        i, j = row[:2]
        if not np.array_equal(data["left"][i], data["left"][j]):
            raise AssertionError("Positive starts differ")
        if not np.array_equal(data["right"][i], data["right"][j]):
            raise AssertionError("Positive endpoints differ")
        if data["length"][i] == data["length"][j]:
            raise AssertionError("Expected unequal action-sequence lengths")
        for k in row[2:]:
            assert np.array_equal(data["left"][i], data["left"][k])
            assert not np.array_equal(data["right"][i], data["right"][k])


@torch.no_grad()
def encoded_paths(model, ids, lengths):
    return torch.cat(
        [
            model(ids[start : start + 512], lengths[start : start + 512])
            for start in range(0, len(ids), 512)
        ]
    )


def retrieval(scores):
    positive = scores[:, :1]
    greater = (scores[:, 1:] > positive + 1e-6).sum(1)
    ties = torch.isclose(scores[:, 1:], positive, atol=1e-6, rtol=0).sum(1)
    return float(torch.where(greater == 0, 1.0 / (ties + 1), 0.0).mean())


def path_metrics(vectors, panels):
    output = {}
    for name, panel in panels.items():
        scores = (vectors[panel[:, :1]] * vectors[panel[:, 1:]]).sum(-1)
        output[name] = dict(
            top1=retrieval(scores),
            n=len(panel),
            positive_cosine=float(scores[:, 0].mean()),
            negative_cosine=float(scores[:, 1:].mean()),
            nll=float(
                F.cross_entropy(
                    scores / 0.1, torch.zeros(len(scores), dtype=torch.long)
                )
            ),
        )
    return output


def payment_metrics(data, panels):
    """A deliberate shortcut control: exact public chip payments, no learned model."""
    paid = np.zeros((len(data["length"]), 2), np.float32)
    for i, length in enumerate(data["length"]):
        for kind, subject, card, action, street, amount in data["tokens"][i, :length]:
            if kind == BET:
                paid[i, subject] += amount
    paid = torch.from_numpy(paid)
    return {
        name: {
            "top1": retrieval(
                -(paid[panel[:, :1]] - paid[panel[:, 1:]]).square().sum(-1)
            )
        }
        for name, panel in panels.items()
    }


def train_paths(args, out, data, vectors, vocab, mode, panels):
    started = time.perf_counter()
    torch.manual_seed(args.seed + 100)
    rng = np.random.default_rng(args.seed + 100)
    ids, lengths, unknown = path_inputs(data, vocab)
    if unknown:
        raise ValueError(
            f"Path tokens outside train-only SGNS vocabulary: {unknown:.3%}"
        )
    model = PathEncoder(len(vocab) + 2, args.dim)
    if mode == "sgns_endpoint":
        with torch.no_grad():
            model.embedding.weight.copy_(vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    choices = relation_choices(data, 0, train=True)
    curve = []
    for step in range(args.path_steps + 1):
        if step:
            panel = sample_relations(choices, rng, min(args.batch, 128))
            flat = panel.ravel()
            embedded = model(ids[flat], lengths[flat]).reshape(len(panel), 7, -1)
            scores = (embedded[:, :1] * embedded[:, 1:]).sum(-1) / 0.1
            loss = F.cross_entropy(scores, torch.zeros(len(panel), dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % args.log_every == 0 or step == args.path_steps:
            metrics = path_metrics(
                encoded_paths(model, ids, lengths),
                {k: v for k, v in panels.items() if not k.startswith("test")},
            )
            curve.append(dict(step=step, metrics=metrics))
            write_json(out / f"{mode}_progress.json", curve)
            print(
                f"seed={args.seed} {mode} {step}: val={metrics['val']['top1']:.4f}",
                flush=True,
            )
    checkpoint = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        numpy_rng=rng.bit_generator.state,
        vocabulary=len(vocab) + 2,
        vocab=torch.from_numpy(vocab),
        dim=args.dim,
        step=args.path_steps,
        mode=mode,
        source_sha256=source_hashes(),
        paths_sha256=hashlib.sha256(
            (args.out_dir / "paths.npz").read_bytes()
        ).hexdigest(),
    )
    torch.save(checkpoint, out / f"{mode}.pt")
    restored = load_path_encoder(out / f"{mode}.pt")
    with torch.no_grad():
        torch.testing.assert_close(
            model(ids[:8], lengths[:8]), restored(ids[:8], lengths[:8])
        )
    final = path_metrics(encoded_paths(restored, ids, lengths), panels)
    result = dict(
        mode=mode,
        seed=args.seed,
        curve=curve,
        final=final,
        unknown_fraction=unknown,
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started,
        selection="fixed final step; test evaluated only at final",
        steps=args.path_steps,
        batch=min(args.batch, 128),
        training_anchors=len(choices),
    )
    write_json(out / f"{mode}_metrics.json", result)


def load_path_encoder(path):
    """Load a trusted checkpoint for f(token_ids, lengths); vocabulary is in payload."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = PathEncoder(payload["vocabulary"], payload["dim"])
    model.load_state_dict(payload["model"])
    return model.eval()


def train(args):
    out = args.out_dir / f"seed{args.seed}"
    out.mkdir(exist_ok=False)
    write_json(
        out / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256": source_hashes(),
        },
    )
    prepared = corpus_pairs(args.corpus, args.window)
    vectors, vocab = train_sgns(args, out, prepared)
    with np.load(args.out_dir / "paths.npz", allow_pickle=False) as loaded:
        data = dict(loaded)
    panels = relation_panels(data, args.data_seed)
    for panel in panels.values():
        assert_relations(data, panel)
    np.savez_compressed(out / "evaluation_panels.npz", **panels)
    ids, lengths, unknown = path_inputs(data, vocab)
    mean = F.normalize(vectors[ids].sum(1) / lengths[:, None], dim=-1)
    baselines = dict(
        sgns_mean=path_metrics(mean, panels),
        payment=payment_metrics(data, panels),
        constant_top1=1 / 6,
        unknown_fraction=unknown,
    )
    write_json(out / "baselines.json", baselines)
    for mode in ("endpoint", "sgns_endpoint"):
        train_paths(args, out, data, vectors, vocab, mode, panels)


def audit(args):
    """Context-frequency shortcut baseline on exactly the saved-run eval panels."""
    vocab, pairs, split, coverage = corpus_pairs(args.corpus, args.window)
    panels, _, _ = context_panels(vocab, pairs, split, args.data_seed)
    counts = np.bincount(pairs[split == 0, 1], minlength=len(vocab) + 2)
    prior = torch.from_numpy(np.log(np.maximum(counts, 1)).astype(np.float32))
    output = {
        name: dict(n=len(centers), top1=retrieval(prior[candidates]))
        for name, (centers, candidates) in panels.items()
    }
    with np.load(args.out_dir / "paths.npz", allow_pickle=False) as loaded:
        data = dict(loaded)
    choices = relation_choices(data, 2, heldout=True)
    output["path_scope"] = {
        "heldout_queries": len(choices),
        "heldout_query_templates": len(
            {
                template_key(data["tokens"][i, : data["length"][i]])
                for i, _, _ in choices
            }
        ),
        "all_templates": len(
            {template_key(t[:n]) for t, n in zip(data["tokens"], data["length"])}
        ),
    }
    output[
        "definition"
    ] = "Training context frequency only; no center/action input. Same candidate panels."
    write_json(args.out_dir / "frequency_audit.json", output)
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "audit"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=50260924)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--sgns-steps", type=int, default=2500)
    parser.add_argument("--path-steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for name in (
        "dim",
        "window",
        "sgns_steps",
        "path_steps",
        "batch",
        "log_every",
        "threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "collect":
        collect(args)
    elif args.command == "train":
        train(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
