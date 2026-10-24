"""Root information-set Local BR-gap for frozen reconstruction bucket policies.

Usage (project root, existing Torch Python):
    python -m agents.state_action_embedding.reconstruction_br_gap --run RUN --out-dir OUT
    python -m agents.state_action_embedding.reconstruction_br_gap --run RUN --out-dir OUT --resume
Input: saved raw/latent 10k/100k/1M tables; encoder and atlas stay frozen.
Output: OUT/config.json and per_root.jsonl; no training or replay dataset writes.
Metric: mean-hidden-world Q first, then max(Q) - sigma dot Q, in ante units.
Scope: first betting decision on fifth street, conditional uniform H4 posterior.
Not exact exploitability, whole-tree BR, or the C++ fixed-H4 evaluation panel.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

from . import reconstruction_cfr as c
from environments.seven_stud.poker_env import ALL_CARDS

np = c.np
WORKER = None


class RootReached(Exception):
    pass


class CaptureRoot(c.BucketAgent):
    def choose_action(self, state, valid_actions):
        raise RootReached(self.seat)


def make_root(seed: int):
    """Stop the original engine before its first fifth-street betting action."""
    game = c.r.events.EventGame()
    agents = {
        f"p{seat}": CaptureRoot(game, seat, None, "uniform", None, seed + seat)
        for seat in (0, 1)
    }
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        try:
            c.sampling.ppo.play(game, agents, seed)
        except RootReached as reached:
            viewer = int(reached.args[0])
        else:
            raise AssertionError("No fifth-street root")
    if game.street != "5th" or game.betting_history or game.decisions:
        raise ValueError("Only pre-betting fifth-street roots are supported")
    return game, viewer


def particle_root(root, viewer: int, seed: int):
    """Sample hidden H4 cards/order/choices and future deck given the viewer's I.

    Uniform H4 choices are independent of card values. Given the revealed card,
    its initial slot and the discard slot are uniform; the other three cards are
    an ordered sample without replacement. No realized opponent hidden card or
    original future deck enters the conditional sample.
    """
    if root.street != "5th" or root.betting_history:
        raise ValueError("Later histories need a policy-conditioned posterior")
    game = copy.deepcopy(root)
    own, opponent = game.players[viewer], game.players[1 - viewer]
    known = set(own.get_all_cards() + [own.discarded_card] + opponent.public_cards)
    if len(known) != 9 or None in known:
        raise ValueError("Unexpected fifth-street card counts")
    unseen = [card for card in ALL_CARDS if card not in known]
    rng = np.random.default_rng(seed)
    rng.shuffle(unseen)
    reveal_slot, discard_slot = rng.choice(4, size=2, replace=False)
    initial = [None] * 4
    initial[reveal_slot] = opponent.public_cards[0]
    initial[discard_slot] = unseen[2]
    hidden = iter(unseen[:2])
    for slot in range(4):
        if initial[slot] is None:
            initial[slot] = next(hidden)
    opponent.hidden_cards = [
        card
        for slot, card in enumerate(initial)
        if slot not in (reveal_slot, discard_slot)
    ]
    opponent.discarded_card = unseen[2]
    game.deck.cards = unseen[3:]

    private_index = 0
    replaced = []
    e = c.r.events
    for token in game.events[1 - viewer]:
        kind, subject, card, action, street, paid = token
        if kind == e.PRIVATE and subject == 0:
            card = e.CARD_IDS[str(initial[private_index])] + 1
            private_index += 1
        elif kind == e.DISCARD and subject == 0:
            card = e.CARD_IDS[str(opponent.discarded_card)] + 1
        replaced.append((kind, subject, card, action, street, paid))
    if private_index != 4:
        raise AssertionError("Incomplete opponent initial-card history")
    game.events[1 - viewer] = replaced
    return game


class ForcedRootAgent(c.BucketAgent):
    """Keep the original policy/depth bookkeeping, overriding only the root move."""

    forced_action = None

    def choose_action(self, state, valid_actions):
        action = super().choose_action(state, valid_actions)
        if self.forced_action is not None:
            action, self.forced_action = self.forced_action, None
            if action not in valid_actions:
                raise ValueError("Illegal root intervention")
            self.rows[-1]["action"] = c.r.BETTING_ACTIONS.index(action)
        return action


def finish_from_fifth(game, agents: dict) -> dict:
    """Continue a pre-betting root with original engine round/payout functions.

    This only calls agents, never bucket IDs. A refining policy can use the same
    evaluator once its saved policy exposes the same observation-history API.
    """
    if game.street != "5th" or game.betting_history:
        raise ValueError("Continuation must start before fifth-street betting")
    game.play_betting_round(agents)
    for street, public in (("6th", True), ("7th_hidden", False)):
        survivors = [p for p in game.players if not p.is_folded and not p.is_eliminated]
        if len(survivors) <= 1:
            break
        game.street = street
        game.deal_cards_to_active(public)
        if sum(p.can_act() for p in survivors) >= 2:
            game.play_betting_round(agents)
    result = game.resolve_showdown()
    if sum(result["final_chips"].values()) != 2 * c.r.STACK:
        raise AssertionError("Chip conservation failed")
    return result


def gap_statistics(values, strategy, legal) -> dict:
    """Plugin gap and two-fold held-out deviation, without hidden-world max."""
    values = np.asarray(values, float)
    strategy, legal = np.asarray(strategy, float), np.asarray(legal, bool)
    if (
        values.ndim != 2
        or values.shape[1] != len(strategy)
        or len(values) < 4
        or len(values) % 2
        or not np.isfinite(values).all()
        or not np.isfinite(strategy).all()
        or (strategy < 0).any()
        or not np.isclose(strategy.sum(), 1)
        or not legal.any()
        or (strategy[~legal] != 0).any()
    ):
        raise ValueError("Invalid rollout values or legal policy")
    actions = np.flatnonzero(legal)
    q = values.mean(0)
    half = len(values) // 2
    left, right = values[:half].mean(0), values[half:].mean(0)

    def best(q_values):
        return int(actions[np.argmax(q_values[actions])])

    def gap(q_values):
        return max(0.0, float(q_values[best(q_values)] - strategy @ q_values))

    # Selection and evaluation use independent particles; unlike plugin gap,
    # this noisy estimate may be negative and must not be clipped/log-plotted.
    crossfit = 0.5 * (
        right[best(left)] - strategy @ right + left[best(right)] - strategy @ left
    )
    return dict(
        local_br_gap=gap(q),
        half_particle_gap=0.5 * (gap(left) + gap(right)),
        crossfit_deviation=float(crossfit),
        q_mean=q.tolist(),
        q_se=(values.std(0, ddof=1) / np.sqrt(len(values))).tolist(),
        strategy=strategy.tolist(),
        legal=legal.tolist(),
    )


@lru_cache(maxsize=8)
def checkpoint_tables(path: str):
    saved = c.load_tables(Path(path))
    for name in ("regret", "average"):
        if not np.isfinite(saved[name]).all():
            raise ValueError("Nonfinite checkpoint")
        saved[name].flags.writeable = False
    return saved


def init_worker(
    run: str, reference: str, encoder_seed: int, buckets: int, particles: int, seed: int
):
    global WORKER
    c.torch.set_num_threads(1)
    WORKER = (
        Path(run),
        c.FrozenBuckets(reference, encoder_seed, buckets),
        particles,
        seed,
    )


def evaluate_root(task) -> dict:
    method, hands, policy, index = task
    run, buckets, particles, seed = WORKER
    started = time.perf_counter()
    root_seed = seed + index * 104729
    root, viewer = make_root(root_seed)
    table = checkpoint_tables(str(run / method / f"checkpoint_{hands:07d}.npz"))[
        "regret" if policy == "current" else "average"
    ]
    root_agent = c.BucketAgent(root, viewer, buckets, method, table, 0)
    legal_actions = root.get_valid_actions(root.players[viewer])
    root_agent.choose_action(
        root.get_ai_state(root.players[viewer], legal_actions), legal_actions
    )
    record = root_agent.rows[-1]
    strategy, legal = record["probabilities"], record["legal"]
    values = np.zeros((particles, len(strategy)))
    queries = missing = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for particle in range(particles):
            particle_seed = root_seed + 1_000_000_007 + particle * 65537
            hidden_root = particle_root(root, viewer, particle_seed)
            for action in np.flatnonzero(legal):
                game = copy.deepcopy(hidden_root)
                agents = {
                    f"p{seat}": ForcedRootAgent(
                        game,
                        seat,
                        buckets,
                        method,
                        table,
                        particle_seed + 2_000_000_011 + seat,
                    )
                    for seat in (0, 1)
                }
                agents[f"p{viewer}"].forced_action = c.r.BETTING_ACTIONS[action]
                result = finish_from_fifth(game, agents)
                values[particle, action] = (
                    result["final_chips"][f"p{viewer}"] - c.r.STACK
                ) / game.ante
                missing += sum(a.missing for a in agents.values())
                queries += sum(len(a.rows) for a in agents.values())
    signature = hashlib.sha256(c.sampling.prefix(root, viewer).tobytes()).hexdigest()
    return dict(
        method=method,
        hands=hands,
        policy=policy,
        root=index,
        root_seed=root_seed,
        viewer=viewer,
        observation_sha256=signature,
        root_bucket=int(record["key"][2]),
        particles=particles,
        rollouts=particles * int(legal.sum()),
        policy_queries=queries,
        missing=missing,
        root_missing=root_agent.missing,
        seconds=time.perf_counter() - started,
        **gap_statistics(values, strategy, legal),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=128)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=261003)
    parser.add_argument(
        "--hands", type=int, nargs="+", default=[10000, 100000, 1000000]
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("raw", "latent"), default=["raw", "latent"]
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(args.roots, args.workers, args.seed, *args.hands) < 1
        or args.particles < 4
        or args.particles % 2
    ):
        parser.error(
            "Positive budgets/seed and an even particle count >= 4 are required"
        )
    args.run, args.out_dir = args.run.resolve(), args.out_dir.resolve()
    if (
        not args.out_dir.is_relative_to((c.FAMILY / "data").resolve())
        or args.out_dir == args.run
    ):
        parser.error("Use a separate output directory in this family's data/")
    sources = {str(Path(__file__).resolve()): c.r.sha(Path(__file__))}
    configs = {}
    for method in args.methods:
        path = args.run / method / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        for source, digest in config["hashes"].items():
            if c.r.sha(Path(source)) != digest:
                raise ValueError(f"Training source/model changed: {source}")
            sources[source] = digest
        if "uniform H4" not in config["protocol"]:
            raise ValueError("This posterior requires uniform H4")
        configs[method] = config
        for hands in args.hands:
            checkpoint = args.run / method / f"checkpoint_{hands:07d}.npz"
            saved = c.load_tables(checkpoint)
            expected = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            if int(saved["hands"]) != hands or saved["config_hash"].item() != expected:
                raise ValueError("Checkpoint metadata mismatch")
            sources[str(checkpoint)] = c.r.sha(checkpoint)
    source = next(iter(configs.values()))
    for config in configs.values():
        for key in (
            "reference",
            "encoder_seed",
            "buckets",
            "protocol",
            "seed",
            "batch",
            "exploration",
        ):
            if config[key] != source[key]:
                raise ValueError(f"Unmatched comparison condition: {key}")
    metadata = dict(
        run=str(args.run),
        roots=args.roots,
        particles=args.particles,
        seed=args.seed,
        hands=args.hands,
        methods=args.methods,
        policies=["current", "average"],
        protocol="fifth-street first-actor conditional uniform-H4 posterior; mean Q then max; ante units; paired action RNG",
        hashes=sources,
        training_configs=configs,
    )
    config_path = args.out_dir / "config.json"
    output = args.out_dir / "per_root.jsonl"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a fresh evaluation directory")
        if json.loads(config_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Evaluation configuration/source mismatch")
    else:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise FileExistsError("Output directory is not empty")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        c.r.write_json(config_path, metadata)
    finished = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ("method", "hands", "policy", "root"))
            if key in finished:
                raise ValueError("Duplicate evaluation row")
            finished.add(key)
    tasks = [
        (method, hands, policy, root)
        for hands in args.hands
        for policy in ("current", "average")
        for root in range(args.roots)
        for method in args.methods
    ]
    pending = [task for task in tasks if task not in finished]
    started, last_log = time.perf_counter(), 0.0
    arguments = (
        str(args.run),
        source["reference"],
        source["encoder_seed"],
        source["buckets"],
        args.particles,
        args.seed,
    )
    with output.open("a", encoding="utf-8") as file:
        if args.workers == 1:
            init_worker(*arguments)
            pool = contextlib.nullcontext(None)
        else:
            pool = mp.get_context("spawn").Pool(args.workers, init_worker, arguments)
        with pool as workers:
            rows = (
                map(evaluate_root, pending)
                if workers is None
                else workers.imap(evaluate_root, pending, chunksize=1)
            )
            for count, row in enumerate(rows, start=1):
                file.write(json.dumps(row, allow_nan=False) + "\n")
                file.flush()
                now = time.perf_counter()
                if now - last_log >= 30 or count == len(pending):
                    print(
                        f"GAP_EVAL completed={len(finished) + count} total={len(tasks)} method={row['method']} hands={row['hands']} policy={row['policy']} elapsed_seconds={now - started:.1f}",
                        flush=True,
                    )
                    last_log = now
    print(
        f"COMPLETE roots_per_policy={args.roots} particles={args.particles} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
