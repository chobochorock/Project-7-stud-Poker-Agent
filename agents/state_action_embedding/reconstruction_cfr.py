"""Frozen observation-reconstruction buckets with batched outcome-sampling CFR.

Usage (project root, existing Torch Python):
    python -m agents.state_action_embedding.reconstruction_cfr audit --out-dir RUN
    python -m agents.state_action_embedding.reconstruction_cfr train --out-dir RUN --hands 10000
    python -m agents.state_action_embedding.reconstruction_cfr train --out-dir RUN --hands 1000000 --resume
Output: family data/RUN, resumable NPZ tables, JSONL progress, paired evaluations.
Metrics: known-count top-k reconstruction and chips/hand, NOT BR-gap/exploitability.
Limit: imperfect-recall buckets; frozen policy within each 128-hand sample batch.
See RECONSTRUCTION_CFR.md for sampling, averaging, partitions and comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import time

from . import observation_reconstruction as r
from . import raw_cfr_ppo as sampling
from .separate_skipgram import state_features

np, torch = r.np, r.torch
FAMILY = Path(__file__).parent
REFERENCE = FAMILY / "data/reconstruction_20261002"
CARD_GROUPS = ("hidden", "public", "discard", "opp_public")
MAX_DECISIONS = 32
WORKER = None


def known_card_counts(tokens):
    """Counts from visible event types/recipients, never from card identities."""
    counts = np.zeros(4, np.int64)
    for kind, subject, *_ in tokens:
        if kind == r.events.PRIVATE and subject == 0:
            counts[0] += 1
        elif kind == r.events.PUBLIC:
            counts[1 if subject == 0 else 3] += 1
        elif kind == r.events.DISCARD and subject == 0:
            counts[0] -= 1
            counts[2] += 1
        elif kind == r.events.REVEAL:
            if subject == 0:
                counts[0] -= 1
            counts[1 if subject == 0 else 3] += 1
    if (counts < 0).any() or (counts > 52).any():
        raise ValueError("Invalid observable card counts")
    return counts


def state_scores(logits, prediction, truth, counts):
    """Rank each card group independently; keep impossible cross-group overlaps."""
    truth_cards = truth[:208].reshape(4, 52) > 0.5
    np.testing.assert_array_equal(counts, truth_cards.sum(1))
    order = np.argsort(-logits[:208].reshape(4, 52), axis=1, kind="stable")
    selected = np.zeros((4, 52), bool)
    scores = {}
    for group, name in enumerate(CARD_GROUPS):
        k = int(counts[group])
        selected[group, order[group, :k]] = True
        hits = int((selected[group] & truth_cards[group]).sum())
        scores[f"{name}_topk_exact"] = float(hits == k)
        scores[f"{name}_topk_recall"] = hits / k if k else 1.0
    scores["cards_topk_exact"] = float((selected == truth_cards).all())
    scores["topk_card_collision"] = float((selected.sum(0) > 1).any())
    scores["legal_exact"] = float(
        np.array_equal(prediction[213:221] >= 0.5, truth[213:221] > 0.5)
    )
    scores["chip_mae"] = float(
        np.abs(prediction[221:230] - truth[221:230]).mean() * 1000
    )
    scores["count_mae"] = float(
        (np.abs(prediction[230:233] - truth[230:233]) * r.UNITS[9:]).mean()
    )
    scores["street_accuracy"] = float(
        prediction[208:211].argmax() == truth[208:211].argmax()
    )
    scores["actor_accuracy"] = float(
        prediction[211:213].argmax() == truth[211:213].argmax()
    )
    return scores


def mean_records(records):
    if not records:
        return {}
    return {key: float(np.mean([row[key] for row in records])) for key in records[0]}


class FrozenBuckets:
    def __init__(self, reference, encoder_seed, budget):
        self.directory = Path(reference) / "transformer" / f"seed{encoder_seed}"
        self.saved = torch.load(
            self.directory / "best.pt", weights_only=True, map_location="cpu"
        )
        self.model = r.ReconstructionModel(
            self.saved["kind"], self.saved["width"]
        ).eval()
        self.model.load_state_dict(self.saved["model"])
        self.normal = self.saved["normalization"]
        self.atlases = {}
        for method in ("raw", "latent"):
            with np.load(
                self.directory / f"kmeans_{method}_{budget}.npz", allow_pickle=False
            ) as file:
                atlas = {k: file[k] for k in ("centers", "groups", "mean", "scale")}
            if (
                len(atlas["centers"]) != budget
                or not np.isfinite(atlas["centers"]).all()
            ):
                raise ValueError("Invalid saved centroid table")
            atlas["partitions"] = {
                int(g): np.flatnonzero(atlas["groups"] == g)
                for g in np.unique(atlas["groups"])
            }
            self.atlases[method] = atlas

    @torch.inference_mode()
    def encode(self, tokens):
        if not 0 < len(tokens) <= 128:
            raise ValueError("Context overflow; no silent truncation")
        features = r.field_arrays(tokens, self.normal["paid"])[0]
        features = (features - self.normal["mean"].numpy()) / self.normal[
            "scale"
        ].numpy()
        output, latent = self.model(
            torch.from_numpy(features[None]), torch.tensor([len(tokens)])
        )
        prediction = r.decode(output, self.saved["statistics"])[0].numpy()
        return output[0].numpy(), prediction, latent[0].numpy()

    def assign(self, method, features, truth):
        mask = int(np.dot(truth[213:221].astype(int), 2 ** np.arange(8)))
        signature = int(truth[208:211].argmax()) * 256 + mask
        atlas = self.atlases[method]
        candidates = atlas["partitions"].get(signature)
        if candidates is None:
            return -1
        vector = (features - atlas["mean"]) / atlas["scale"]
        distance = ((atlas["centers"][candidates] - vector) ** 2).sum(1)
        return int(candidates[distance.argmin()])


def probabilities(values, legal):
    positive = np.maximum(values, 0) * legal
    if not np.isfinite(positive).all():
        raise FloatingPointError("Nonfinite policy values")
    if positive.sum() == 0:
        positive = legal.astype(float)
    return positive / positive.sum()


class BucketAgent(sampling.ppo.PokerAgent):
    def __init__(
        self, game, seat, buckets, method, table, seed, exploration=0.0, diagnose=False
    ):
        super().__init__(f"p{seat}")
        self.game, self.seat, self.buckets = game, seat, buckets
        self.method, self.table, self.exploration = method, table, exploration
        self.rng = np.random.default_rng(seed)
        self.rows, self.scores, self.missing = [], [], 0
        self.diagnose = diagnose
        self.heuristic = (
            r.events.HeuristicPokerAgent(self.name) if method == "heuristic" else None
        )

    def choose_discard_and_reveal(self, hidden_cards):
        return tuple(self.rng.choice(len(hidden_cards), 2, replace=False).tolist())

    def choose_action(self, state, valid_actions):
        legal = np.array([a in valid_actions for a in r.BETTING_ACTIONS])
        if not legal.any():
            raise ValueError("Empty action set")
        truth = state_features(r.events.encode_event(state, "CHECK", True))
        tokens = sampling.prefix(self.game, self.seat)
        encoded = None
        if self.method == "latent" or self.diagnose:
            encoded = self.buckets.encode(tokens)
        if self.diagnose:
            self.scores.append(
                state_scores(encoded[0], encoded[1], truth, known_card_counts(tokens))
            )
        depth = len(self.rows)
        if depth >= MAX_DECISIONS:
            raise ValueError("Decision-depth capacity exceeded; do not merge/truncate")
        bucket = -1
        if self.method in ("raw", "latent"):
            features = truth if self.method == "raw" else encoded[2]
            bucket = self.buckets.assign(self.method, features, truth)
            self.missing += int(bucket < 0)
            p = (
                probabilities(self.table[self.seat, depth, bucket], legal)
                if bucket >= 0
                else legal / legal.sum()
            )
        elif self.method == "uniform":
            p = legal / legal.sum()
        elif self.method == "heuristic":
            action = self.heuristic.choose_action(state, valid_actions)
            if action not in valid_actions:
                raise AssertionError("Heuristic returned an illegal action")
            p = np.eye(8)[r.BETTING_ACTIONS.index(action)]
        else:
            raise ValueError(self.method)
        q = (1 - self.exploration) * p + self.exploration * legal / legal.sum()
        action = int(self.rng.choice(8, p=q))
        self.rows.append(
            dict(
                key=(self.seat, depth, bucket),
                legal=legal,
                probabilities=p,
                sampling=q,
                action=action,
                tokens=None,
                global_order=len(self.game.decisions),
            )
        )
        return r.BETTING_ACTIONS[action]


def average_increments(rows, traverser):
    """Own reach / behavior reach; common street-specific chance factor omitted.

    Both seats have uniform fixed H4, and every betting node on a street has the
    same number of chance deals. This cancellation must not be generalized to
    unequal/dependent chance probabilities or mixed-street buckets.
    """
    own_reach, behavior_reach = 1.0, 1.0
    result = []
    for row in sorted(rows, key=lambda v: v["global_order"]):
        if row["key"][0] == traverser:
            weight = own_reach / behavior_reach
            result.append(weight * row["probabilities"])
            own_reach *= row["probabilities"][row["action"]]
        behavior_reach *= row["sampling"][row["action"]]
    return result


def init_worker(
    reference, encoder_seed, budget, shared_name, shape, method, seed, exploration
):
    global WORKER
    torch.set_num_threads(1)
    memory = shared_memory.SharedMemory(name=shared_name)
    table = np.ndarray(shape, np.float64, buffer=memory.buf)
    table.flags.writeable = False
    WORKER = (
        memory,
        table,
        FrozenBuckets(reference, encoder_seed, budget),
        method,
        seed,
        exploration,
    )


def sample_hands(indices):
    _, table, buckets, method, seed, exploration = WORKER
    keys, regrets, averages = [], [], []
    decisions = missing = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in indices:
            traverser = hand % 2
            game = r.events.EventGame()
            agents = [
                BucketAgent(
                    game,
                    seat,
                    buckets,
                    method,
                    table,
                    seed + 2 * hand + seat,
                    exploration if seat == traverser else 0.0,
                )
                for seat in (0, 1)
            ]
            result = sampling.ppo.play(
                game, {a.name: a for a in agents}, 940_000_000 + seed + hand
            )
            utility = (result["final_chips"][f"p{traverser}"] - r.STACK) / 100.0
            rows = agents[traverser].rows
            increments = list(reversed(sampling.outcome_targets(rows, utility)))
            avg = average_increments(agents[0].rows + agents[1].rows, traverser)
            for row, increment, av in zip(rows, increments, avg):
                if row["key"][2] >= 0:
                    keys.append(row["key"])
                    regrets.append(increment["target"])
                    averages.append(av)
            decisions += sum(len(a.rows) for a in agents)
            missing += sum(a.missing for a in agents)
    return (
        np.asarray(keys, np.int64).reshape(-1, 3),
        np.asarray(regrets, np.float64).reshape(-1, 8),
        np.asarray(averages, np.float64).reshape(-1, 8),
        decisions,
        missing,
    )


def checkpoint(
    path, table, average, visits, hand, decisions, missing, seconds, config_hash
):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        regret=table,
        average=average,
        visits=visits,
        hands=hand,
        decisions=decisions,
        missing=missing,
        seconds=seconds,
        config_hash=config_hash,
    )
    os.replace(temporary, path)


def load_tables(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def evaluate(args, method, step, buckets):
    path = args.out_dir / method / f"checkpoint_{step:07d}.npz"
    saved = load_tables(path)
    output = args.out_dir / method / f"evaluation_{step:07d}.json"
    if output.exists():
        return
    records = []
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for policy in ("current", "average"):
            table = saved["regret" if policy == "current" else "average"]
            for opponent in ("uniform", "heuristic"):
                payoffs, scores = [], []
                missing = decisions = 0
                for pair in range(args.eval_pairs):
                    seat_values = []
                    for seat in (0, 1):
                        game = r.events.EventGame()
                        agents = [
                            BucketAgent(
                                game,
                                s,
                                buckets,
                                method if s == seat else opponent,
                                table,
                                970_000_000 + pair * 2 + s,
                                diagnose=s == seat,
                            )
                            for s in (0, 1)
                        ]
                        result = sampling.ppo.play(
                            game, {a.name: a for a in agents}, 960_000_000 + pair
                        )
                        seat_values.append(result["final_chips"][f"p{seat}"] - r.STACK)
                        if agents[seat].scores:
                            scores.append(mean_records(agents[seat].scores))
                        missing += agents[seat].missing
                        decisions += len(agents[seat].rows)
                    payoffs.append(float(np.mean(seat_values)))
                values = np.asarray(payoffs)
                mean = float(values.mean())
                se = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                )
                records.append(
                    dict(
                        method=method,
                        hands=step,
                        policy=policy,
                        opponent=opponent,
                        pairs=args.eval_pairs,
                        mean_chips=mean,
                        se=se,
                        normal95=[mean - 1.96 * se, mean + 1.96 * se],
                        pair_payoffs=payoffs,
                        reconstruction=mean_records(scores),
                        decisions=decisions,
                        missing=missing,
                    )
                )
    r.write_json(output, records)
    print(
        f"EVALUATED method={method} hands={step} pairs_per_opponent={args.eval_pairs}",
        flush=True,
    )


def audit(args):
    torch.set_num_threads(1)
    buckets = FrozenBuckets(args.reference, args.encoder_seed, args.buckets)
    normal = {
        key: value.numpy() if isinstance(value, torch.Tensor) else value
        for key, value in buckets.saved["normalization"].items()
    }
    data = r.load_data(args.reference / "fresh_lowfold.npz", normal)
    groups = {name: [] for name in ("all", "street_5", "street_6", "street_7")}
    owners = {name: [] for name in groups}
    max_latent_error = 0.0
    with np.load(
        buckets.directory / "test_lowfold_latents.npz", allow_pickle=False
    ) as file:
        exported = {k: file[k] for k in ("latent", "truth")}
    np.testing.assert_array_equal(exported["truth"], data["y"].numpy())
    with torch.inference_mode():
        for start in range(0, len(data["x"]), 64):
            end = start + 64
            hidden = buckets.model.backbone(data["x"][start:end])[1]
            logits = buckets.model.head(hidden)
            rows = np.flatnonzero(
                (data["sequence"] >= start) & (data["sequence"] < end)
            )
            output = logits[data["sequence"][rows] - start, data["ends"][rows] - 1]
            predictions = r.decode(output, buckets.saved["statistics"]).numpy()
            latents = hidden[
                data["sequence"][rows] - start, data["ends"][rows] - 1
            ].numpy()
            max_latent_error = max(
                max_latent_error,
                float(np.max(np.abs(latents - exported["latent"][rows]))),
            )
            for local, row in enumerate(rows):
                truth = data["y"][row].numpy()
                if truth[212] != 1:
                    continue
                lo = data["starts"][data["sequence"][row]]
                tokens = data["tokens"][lo : lo + data["ends"][row]]
                scores = state_scores(
                    output[local].numpy(),
                    predictions[local],
                    truth,
                    known_card_counts(tokens),
                )
                for scope in ("all", f"street_{truth[208:211].argmax() + 5}"):
                    groups[scope].append(scores)
                    owners[scope].append(int(data["hand"][row]))
    if max_latent_error > 1e-4:
        raise AssertionError(f"Latent reload mismatch: {max_latent_error}")
    summary = {}
    for scope, records in groups.items():
        hands = np.asarray(owners[scope])
        keys = list(records[0])
        values = np.array([[v[k] for k in keys] for v in records])
        summary[scope] = dict(
            rows=len(records),
            hands=len(np.unique(hands)),
            hand_macro=dict(zip(keys, r.hand_macro(values, hands).tolist())),
        )
    r.write_json(
        args.out_dir / "topk_static.json",
        dict(
            reference=str(buckets.directory.resolve()),
            scopes=summary,
            latent_reload_max_error=max_latent_error,
            note="Existing held-out low-fold panel, own betting decisions only; no new model fitting.",
        ),
    )
    print(json.dumps(summary["all"], ensure_ascii=False), flush=True)


def run(args):
    torch.set_num_threads(1)
    shape = (2, MAX_DECISIONS, args.buckets, 8)
    directory = args.out_dir / args.method
    directory.mkdir(exist_ok=True)
    checkpoint_path = directory / "last.npz"
    config = dict(
        reference=str(args.reference.resolve()),
        encoder_seed=args.encoder_seed,
        buckets=args.buckets,
        seed=args.seed,
        batch=args.batch,
        exploration=args.exploration,
        method=args.method,
        max_decisions=MAX_DECISIONS,
        eval_pairs=args.eval_pairs,
        protocol="batched signed outcome sampling; uniform H4; own-depth key; constant chance cancellation",
    )
    bucket_dir = args.reference / "transformer" / f"seed{args.encoder_seed}"
    config["hashes"] = {
        str(p.resolve()): r.sha(p)
        for p in (
            Path(__file__),
            Path(r.__file__),
            Path(sampling.__file__),
            FAMILY / "action2vec.py",
            FAMILY / "action_window.py",
            FAMILY / "experiment.py",
            FAMILY / "separate_skipgram.py",
            FAMILY / "transformer_legality.py",
            FAMILY / "bc_ppo.py",
            r.ROOT / "environments/seven_stud/poker_env.py",
            bucket_dir / "best.pt",
            bucket_dir / f"kmeans_{args.method}_{args.buckets}.npz",
        )
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError("Use --resume; old results will not be overwritten")
    memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
    try:
        table = np.ndarray(shape, np.float64, buffer=memory.buf)
        table.fill(0)
        average, visits = np.zeros(shape), np.zeros(shape[:-1], np.uint64)
        hand = decisions = missing = 0
        elapsed = 0.0
        if checkpoint_path.exists():
            old = load_tables(checkpoint_path)
            if old["config_hash"].item() != config_hash:
                raise ValueError("Resume configuration/source/model mismatch")
            table[:] = old["regret"]
            average[:], visits[:] = old["average"], old["visits"]
            hand, decisions, missing = (
                int(old[k]) for k in ("hands", "decisions", "missing")
            )
            elapsed = float(old["seconds"])
        else:
            r.write_json(directory / "config.json", config)
        if hand > args.hands:
            raise ValueError("Requested budget is behind the saved checkpoint")
        milestones = sorted(
            set([v for v in args.milestones if v <= args.hands] + [args.hands])
        )
        buckets = FrozenBuckets(args.reference, args.encoder_seed, args.buckets)
        # Recover a completed checkpoint whose evaluation was interrupted.
        for step in milestones:
            if (directory / f"checkpoint_{step:07d}.npz").exists():
                evaluate(args, args.method, step, buckets)
        started = time.perf_counter()
        last_log = started
        with mp.get_context("spawn").Pool(
            args.workers,
            init_worker,
            (
                str(args.reference),
                args.encoder_seed,
                args.buckets,
                memory.name,
                shape,
                args.method,
                args.seed,
                args.exploration,
            ),
        ) as pool:
            while hand < args.hands:
                next_milestone = min(v for v in milestones if v > hand)
                stop = min(hand + args.batch, next_milestone)
                chunks = [
                    a.tolist()
                    for a in np.array_split(np.arange(hand, stop), args.workers)
                    if len(a)
                ]
                results = pool.map(sample_hands, chunks)
                for keys, regret, av, count, missed in results:
                    if not np.isfinite(regret).all() or not np.isfinite(av).all():
                        raise FloatingPointError("Nonfinite sampled update")
                    index = tuple(keys.T)
                    np.add.at(table, index, regret)
                    np.add.at(average, index, av)
                    np.add.at(visits, index, 1)
                    decisions += count
                    missing += missed
                hand = stop
                now = time.perf_counter()
                seconds = elapsed + now - started
                if now - last_log >= 30 or hand in milestones:
                    checkpoint(
                        checkpoint_path,
                        table,
                        average,
                        visits,
                        hand,
                        decisions,
                        missing,
                        seconds,
                        config_hash,
                    )
                    progress = dict(
                        method=args.method,
                        hands=hand,
                        decisions=decisions,
                        occupied_keys=int(np.count_nonzero(visits)),
                        missing=missing,
                        seconds=seconds,
                        hands_per_second=hand / max(seconds, 1e-9),
                    )
                    with (directory / "progress.jsonl").open(
                        "a", encoding="utf-8"
                    ) as file:
                        file.write(json.dumps(progress) + "\n")
                    print(
                        "PROGRESS " + " ".join(f"{k}={v}" for k, v in progress.items()),
                        flush=True,
                    )
                    last_log = now
                if hand in milestones:
                    checkpoint(
                        directory / f"checkpoint_{hand:07d}.npz",
                        table,
                        average,
                        visits,
                        hand,
                        decisions,
                        missing,
                        seconds,
                        config_hash,
                    )
                    before = time.perf_counter()
                    evaluate(args, args.method, hand, buckets)
                    started += time.perf_counter() - before
    finally:
        memory.close()
        memory.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--encoder-seed", type=int, default=11)
    parser.add_argument("--buckets", type=int, default=512)
    parser.add_argument("--method", choices=("raw", "latent"), default="latent")
    parser.add_argument("--hands", type=int, default=10000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=[10000, 100000, 1000000]
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--exploration", type=float, default=0.6)
    parser.add_argument("--eval-pairs", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(args.hands, args.workers, args.batch, args.eval_pairs) < 1
        or not 0 < args.exploration <= 1
    ):
        parser.error("Positive budgets and exploration in (0,1] required")
    if not args.out_dir.resolve().is_relative_to((FAMILY / "data").resolve()):
        parser.error("Keep generated artifacts inside this family's data/")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit(args) if args.command == "audit" else run(args)


if __name__ == "__main__":
    main()
