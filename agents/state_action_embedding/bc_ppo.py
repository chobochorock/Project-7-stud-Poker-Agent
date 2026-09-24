"""Imitate the complete heuristic policy, then improve it with on-policy PPO.

Usage (project root; Python 3.12 with the existing PyTorch runtime):
    python -m agents.state_action_embedding.bc_ppo collect --out-dir RUN
    python -m agents.state_action_embedding.bc_ppo train --out-dir RUN --seed 11
    python -m agents.state_action_embedding.bc_ppo evaluate --out-dir RUN --seed 11

Input: player-visible chance/action prefixes; original seven-poker v3 rules.
Output: heuristic demonstrations, BC/PPO checkpoints, paired cash-game returns.
Metric: net chips/hand against a fixed heuristic, NOT exploitability or equilibrium.
Both discard/reveal and betting are learned; no heuristic fallback for the learner.
See BC_PPO.md for budgets, source versions, evaluation and checkpoint limitations.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from .action2vec import (
    BETTING_ACTIONS,
    BETTING_RULES_VERSION,
    EventGame,
    EventTransformer,
    HeuristicPokerAgent,
    MAX_LENGTH,
    MemoryMeter,
    ROOT,
    STACK,
    STREETS,
    TURN,
    UniformAgent,
    F,
    nn,
    np,
    torch,
    write_json,
)
from agents.common.base import PokerAgent


DISCARD_PAIRS = tuple((d, r) for d in range(4) for r in range(4) if d != r)
ACTION_COUNT = len(BETTING_ACTIONS) + len(DISCARD_PAIRS)
REWARD_SCALE = 100.0
BC_DEAL_BASE = 310_000_000
PPO_DEAL_BASE = 410_000_000
VALID_DEAL_BASE = 510_000_000
TEST_DEAL_BASE = 610_000_000


def prefix(game: EventGame, seat: int) -> np.ndarray:
    """Append a decision query, with neither the chosen action nor its outcome."""
    turn = (TURN, 0, 0, 0, STREETS.index(game.street), 0)
    tokens = np.asarray([*game.events[seat], turn], dtype=np.int16)
    if len(tokens) > MAX_LENGTH:
        raise ValueError("Full history exceeds the model's context; no truncation")
    return tokens


def legal_mask(valid_actions=None) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    if valid_actions is None:
        mask[8:] = True
    else:
        mask[:8] = [a in valid_actions for a in BETTING_ACTIONS]
    if not mask.any():
        raise ValueError("No legal action")
    return mask


def pack(records: list[dict]) -> dict[str, np.ndarray]:
    lengths = np.asarray([len(r["tokens"]) for r in records], np.int16)
    tokens = np.zeros((len(records), int(lengths.max()), 6), np.int16)
    for row, record in enumerate(records):
        tokens[row, : lengths[row]] = record["tokens"]
    return {
        "tokens": tokens,
        "lengths": lengths,
        "legal": np.asarray([r["legal"] for r in records], bool),
        "action": np.asarray([r["action"] for r in records], np.int64),
    }


def batch(data: dict, rows: np.ndarray, device: str):
    lengths = data["lengths"][rows]
    tokens = torch.as_tensor(
        data["tokens"][rows, : lengths.max()], dtype=torch.long, device=device
    )
    return (
        tokens,
        torch.as_tensor(lengths, dtype=torch.long, device=device),
        torch.as_tensor(data["legal"][rows], device=device),
    )


class ActorCritic(EventTransformer):
    """Same small event backbone; 8 betting + 12 discard/reveal policy outputs."""

    def __init__(self, dim: int = 64, layers: int = 2):
        super().__init__(dim, layers)
        self.action_head = nn.Linear(dim, ACTION_COUNT)
        del self.chance_head

    def distribution_value(self, tokens, lengths, legal):
        if not legal.any(-1).all():
            raise ValueError("Empty legal-action mask")
        hidden = self(tokens)[
            torch.arange(len(tokens), device=tokens.device), lengths - 1
        ]
        logits = self.action_head(hidden).masked_fill(~legal, -1e9)
        return torch.distributions.Categorical(logits=logits), self.value_head(
            hidden
        ).squeeze(-1)


class DecisionAgent(PokerAgent):
    """Engine adapter: model reads only its event view; teacher uses public API."""

    def __init__(self, game, seat, model=None, rng=None, record=True, greedy=False):
        super().__init__(f"p{seat}")
        self.game, self.seat, self.model = game, seat, model
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.record, self.greedy = record, greedy
        self.teacher = HeuristicPokerAgent(self.name) if model is None else None
        self.records = []

    def decide(self, valid_actions=None, state=None, hidden_cards=None):
        tokens = prefix(self.game, self.seat)
        legal = legal_mask(valid_actions)
        if self.model is None:
            if valid_actions is None:
                action = 8 + DISCARD_PAIRS.index(
                    self.teacher.choose_discard_and_reveal(hidden_cards)
                )
            else:
                action = BETTING_ACTIONS.index(
                    self.teacher.choose_action(state, valid_actions)
                )
            log_prob = value = 0.0
        else:
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                distribution, values = self.model.distribution_value(
                    torch.as_tensor(tokens[None], dtype=torch.long, device=device),
                    torch.tensor([len(tokens)], device=device),
                    torch.as_tensor(legal[None], device=device),
                )
                probability = distribution.probs[0].cpu().numpy().astype(np.float64)
                probability /= probability.sum()
                action = int(
                    probability.argmax()
                    if self.greedy
                    else self.rng.choice(ACTION_COUNT, p=probability)
                )
                log_prob = float(
                    distribution.log_prob(torch.tensor([action], device=device)).item()
                )
                value = float(values.item())
        if not legal[action]:
            raise AssertionError(
                "Agent selected an illegal action; no fallback allowed"
            )
        if self.record:
            self.records.append(
                {
                    "tokens": tokens,
                    "legal": legal,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                }
            )
        return action

    def choose_action(self, state, valid_actions):
        return BETTING_ACTIONS[self.decide(valid_actions, state)]

    def choose_discard_and_reveal(self, hidden_cards):
        return DISCARD_PAIRS[self.decide(hidden_cards=hidden_cards) - 8]


def play(game: EventGame, agents: dict, deal_seed: int) -> dict:
    # Policy RNGs are separate: actions cannot consume/reorder the chance stream.
    random.seed(deal_seed)
    result = game.play_hand(agents)
    if sum(result["final_chips"].values()) != 2 * STACK:
        raise AssertionError("Chip conservation failed")
    return result


def source_hashes():
    paths = [
        Path(__file__),
        Path(__file__).with_name("action2vec.py"),
        ROOT / "environments/seven_stud/poker_env.py",
        ROOT / "agents/baselines/heuristic_agent.py",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def collect(args):
    path = args.out_dir / "demonstrations.npz"
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    rng = np.random.default_rng(args.data_seed)
    order = rng.permutation(args.bc_hands)
    hand_split = np.zeros(args.bc_hands, np.int8)
    hand_split[order[int(args.bc_hands * 0.8) : int(args.bc_hands * 0.9)]] = 1
    hand_split[order[int(args.bc_hands * 0.9) :]] = 2
    records, hands, splits = [], [], []
    game = EventGame()
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in range(args.bc_hands):
            agents = {f"p{i}": DecisionAgent(game, i) for i in range(2)}
            play(game, agents, BC_DEAL_BASE + hand)
            for agent in agents.values():
                records.extend(agent.records)
                hands.extend([hand] * len(agent.records))
                splits.extend([hand_split[hand]] * len(agent.records))
    data = pack(records)
    data.update(hand=np.asarray(hands, np.int32), split=np.asarray(splits, np.int8))
    np.savez_compressed(path, **data)
    write_json(
        args.out_dir / "demonstrations.json",
        {
            "hands": args.bc_hands,
            "decisions": len(records),
            "split_hands": [int((hand_split == i).sum()) for i in range(3)],
            "max_prefix": int(data["lengths"].max()),
            "deal_seed_base": BC_DEAL_BASE,
            "split_seed": args.data_seed,
            "rules_version": BETTING_RULES_VERSION,
            "policy": "heuristic vs heuristic; all actions incl. discard/reveal; actor-view prefixes",
            "source_sha256": source_hashes(),
            "seconds": time.perf_counter() - started,
        },
    )
    print(f"Collected {len(records)} decisions from {args.bc_hands} hands", flush=True)


def imitation_metrics(model, data, rows, device):
    model.eval()
    totals = {"all": [0.0, 0, 0], "bet": [0.0, 0, 0], "discard_reveal": [0.0, 0, 0]}
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            chosen = rows[start : start + 256]
            distribution, _ = model.distribution_value(*batch(data, chosen, device))
            target = torch.as_tensor(data["action"][chosen], device=device)
            nll = -distribution.log_prob(target)
            correct = distribution.probs.argmax(-1) == target
            for name, mask in (
                ("all", target >= 0),
                ("bet", target < 8),
                ("discard_reveal", target >= 8),
            ):
                totals[name][0] += float(nll[mask].sum())
                totals[name][1] += int(correct[mask].sum())
                totals[name][2] += int(mask.sum())
    return {
        name: {"nll": v[0] / v[2], "accuracy": v[1] / v[2], "decisions": v[2]}
        for name, v in totals.items()
    }


def save_policy(path, model, **metadata):
    torch.save(
        {
            "weights": model.state_dict(),
            "dim": model.dim,
            "layers": len(model.layers),
            "betting_actions": BETTING_ACTIONS,
            "discard_pairs": DISCARD_PAIRS,
            "reward_scale": REWARD_SCALE,
            "rules_version": BETTING_RULES_VERSION,
            **metadata,
        },
        path,
    )


def load_policy(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    if (
        saved["betting_actions"] != BETTING_ACTIONS
        or saved["discard_pairs"] != DISCARD_PAIRS
        or saved["rules_version"] != BETTING_RULES_VERSION
    ):
        raise ValueError("Checkpoint rules/action schema mismatch")
    model = ActorCritic(saved["dim"], saved.get("layers", 2)).to(device)
    model.load_state_dict(saved["weights"])
    return model.eval(), saved


def gae(values: np.ndarray, terminal_reward: float, gamma=1.0, lam=0.95):
    """Complete ONE learner trajectory: chance/opponent steps have no policy loss."""
    advantages = np.zeros(len(values), np.float32)
    running = next_value = 0.0
    for step in range(len(values) - 1, -1, -1):
        reward = terminal_reward if step == len(values) - 1 else 0.0
        delta = reward + gamma * next_value - values[step]
        running = delta + gamma * lam * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + values


def clipped_surrogate(log_prob, old_log_prob, advantage, clip=0.2):
    ratio = (log_prob - old_log_prob).exp()
    loss = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()
    approx_kl = ((ratio - 1) - (log_prob - old_log_prob)).mean()
    return loss, approx_kl, ((ratio - 1).abs() > clip).float().mean()


def opponent_agent(
    game: EventGame,
    seat: int,
    opponent: ActorCritic | str | None,
    action_seed: int,
) -> PokerAgent:
    """None retains the original heuristic; models use player-visible prefixes."""
    if opponent is None:
        return HeuristicPokerAgent(f"p{seat}")
    if isinstance(opponent, str):
        if opponent != "random":
            raise ValueError(f"Unknown opponent: {opponent}")
        return UniformAgent(f"p{seat}", action_seed)
    opponent.eval()
    return DecisionAgent(
        game, seat, opponent, np.random.default_rng(action_seed), record=False
    )


def rollout(model, hands, seed_base, action_seed, opponent=None):
    """Fresh hands vs one frozen opponent; alternate the learner's physical seat."""
    model.eval()
    game = EventGame()
    rng = np.random.default_rng(action_seed)
    records, advantages, returns, profits = [], [], [], []
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in range(hands):
            seat = hand % 2
            learner = DecisionAgent(game, seat, model, rng)
            agents = {
                learner.name: learner,
                f"p{1-seat}": opponent_agent(
                    game, 1 - seat, opponent, action_seed + 1_000_000_007 + hand
                ),
            }
            result = play(game, agents, seed_base + hand)
            profit = result["final_chips"][learner.name] - STACK
            values = np.asarray([r["value"] for r in learner.records], np.float32)
            adv, ret = gae(values, profit / REWARD_SCALE)
            records.extend(learner.records)
            advantages.extend(adv)
            returns.extend(ret)
            profits.append(profit)
    data = pack(records)
    data.update(
        old_log_prob=np.asarray([r["log_prob"] for r in records], np.float32),
        advantage=np.asarray(advantages, np.float32),
        returns=np.asarray(returns, np.float32),
    )
    return data, np.asarray(profits)


def evaluate_policy(
    model,
    pairs: int,
    seed_base: int,
    action_seed: int,
    baseline="heuristic",
    greedy=False,
    opponent=None,
):
    """Each shuffled deck runs twice with learner seats exchanged; no test tuning."""
    if model is not None:
        model.eval()
    game = EventGame()
    profits = np.zeros((pairs, 2), np.int32)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for pair in range(pairs):
            for seat in range(2):
                seed = action_seed + pair * 2 + seat
                if model is not None:
                    hero = DecisionAgent(
                        game,
                        seat,
                        model,
                        np.random.default_rng(seed),
                        record=False,
                        greedy=greedy,
                    )
                elif baseline == "random":
                    hero = UniformAgent(f"p{seat}", seed)
                else:
                    hero = HeuristicPokerAgent(f"p{seat}")
                agents = {
                    hero.name: hero,
                    f"p{1-seat}": opponent_agent(
                        game, 1 - seat, opponent, action_seed + pair * 2 + (1 - seat)
                    ),
                }
                result = play(game, agents, seed_base + pair)
                profits[pair, seat] = result["final_chips"][hero.name] - STACK
    return profits


def profit_metrics(profits):
    pair_mean = np.asarray(profits, np.float64).mean(1)
    se = pair_mean.std(ddof=1) / np.sqrt(len(pair_mean))
    return {
        "chips_per_hand": float(pair_mean.mean()),
        "pair_normal_95pct": [
            float(pair_mean.mean() - 1.96 * se),
            float(pair_mean.mean() + 1.96 * se),
        ],
        "win_rate": float((profits > 0).mean()),
        "loss_rate": float((profits < 0).mean()),
        "tie_rate": float((profits == 0).mean()),
        "total_net_chips": int(profits.sum()),
        "hands": int(profits.size),
        "deal_pairs": len(profits),
    }


def update_ppo(model, optimizer, data, rng, args, device):
    advantage = data["advantage"]
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
    totals, stop = [], False
    model.train()
    for epoch in range(args.ppo_epochs):
        order = rng.permutation(len(advantage))
        for start in range(0, len(order), args.batch):
            rows = order[start : start + args.batch]
            distribution, value = model.distribution_value(*batch(data, rows, device))
            action = torch.as_tensor(data["action"][rows], device=device)
            old_log_prob = torch.as_tensor(data["old_log_prob"][rows], device=device)
            adv = torch.as_tensor(advantage[rows], device=device)
            policy_loss, kl, clip_fraction = clipped_surrogate(
                distribution.log_prob(action), old_log_prob, adv
            )
            if float(kl.detach()) > 0.03:
                stop = True
                break
            target = torch.as_tensor(data["returns"][rows], device=device)
            value_loss = 0.5 * F.mse_loss(value, target)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            totals.append(
                [
                    float(v.detach())
                    for v in (policy_loss, value_loss, entropy, kl, clip_fraction)
                ]
            )
        if stop:
            break
    if not totals:
        raise AssertionError("Fresh on-policy batch could not take a single PPO step")
    return dict(
        zip(
            ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"),
            np.mean(totals, axis=0).tolist(),
        ),
        optimizer_steps=len(totals),
        kl_stopped=stop,
    )


def train(args):
    directory = args.out_dir / f"seed{args.seed}"
    directory.mkdir(exist_ok=False)
    demonstrations = args.demonstrations or args.out_dir / "demonstrations.npz"
    with np.load(demonstrations, allow_pickle=False) as archive:
        data = dict(archive)
    train_rows, valid_rows, test_rows = (
        np.flatnonzero(data["split"] == i) for i in range(3)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic(args.dim, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_nll, bc_step, best_weights = float("inf"), 0, None
    curve = []
    train_probe = np.random.default_rng(901).choice(
        train_rows, min(2048, len(train_rows)), replace=False
    )
    write_json(
        directory / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "source_sha256": source_hashes(),
            "demonstrations_sha256": hashlib.sha256(
                demonstrations.read_bytes()
            ).hexdigest(),
            "ppo_clip": 0.2,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "reward_scale": REWARD_SCALE,
            "ppo_lr": 1e-4,
            "entropy_coefficient": 0.01,
            "value_loss": "0.5 * (0.5 * MSE)",
            "max_grad_norm": 0.5,
            "opponent": "fixed HeuristicPokerAgent",
            "not_self_play": True,
        },
    )
    with MemoryMeter() as bc_memory:
        for step in range(args.bc_steps + 1):
            if step:
                model.train()
                rows = rng.choice(train_rows, args.batch)
                distribution, _ = model.distribution_value(*batch(data, rows, device))
                loss = -distribution.log_prob(
                    torch.as_tensor(data["action"][rows], device=device)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if step % args.bc_eval_every == 0 or step == args.bc_steps:
                metric = imitation_metrics(model, data, valid_rows, device)
                curve.append(
                    {
                        "stage": "bc",
                        "step": step,
                        **metric,
                        "train_probe": imitation_metrics(
                            model, data, train_probe, device
                        ),
                    }
                )
                write_json(directory / "progress.json", {"curve": curve})
                print(f"seed={args.seed} BC step={step} {metric}", flush=True)
                if metric["all"]["nll"] < best_nll:
                    best_nll, bc_step = metric["all"]["nll"], step
                    best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    save_policy(
        directory / "bc.pt", model, stage="bc", seed=args.seed, selected_step=bc_step
    )
    bc_test = imitation_metrics(model, data, test_rows, device)
    restored, _ = load_policy(directory / "bc.pt", device)
    sample = batch(data, test_rows[:16], device)
    with torch.inference_mode():
        torch.testing.assert_close(
            model.distribution_value(*sample)[0].probs,
            restored.distribution_value(*sample)[0].probs,
        )
    del restored, data
    torch.set_num_threads(1)
    initial = evaluate_policy(
        model, args.validation_pairs, VALID_DEAL_BASE, 730_000_000 + args.seed * 10000
    )
    best_score = float(initial.mean())
    best_update = 0
    save_policy(
        directory / "ppo_best.pt",
        model,
        stage="bc_ppo",
        seed=args.seed,
        selected_update=0,
    )
    curve.append(
        {
            "stage": "ppo",
            "update": 0,
            "train_hands": 0,
            "validation": profit_metrics(initial),
        }
    )
    print(
        f"seed={args.seed} PPO initial validation {profit_metrics(initial)}", flush=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    ppo_rng = np.random.default_rng(args.seed + 100)
    decisions = 0
    with MemoryMeter() as ppo_memory:
        for update in range(1, args.ppo_updates + 1):
            torch.set_num_threads(1)
            rollout_data, profits = rollout(
                model,
                args.rollout_hands,
                PPO_DEAL_BASE
                + args.seed * 1_000_000
                + (update - 1) * args.rollout_hands,
                710_000_000 + args.seed * 1_000_000 + update,
            )
            decisions += len(rollout_data["action"])
            torch.set_num_threads(args.threads)
            metric = update_ppo(model, optimizer, rollout_data, ppo_rng, args, device)
            point = {
                "stage": "ppo",
                "update": update,
                "train_hands": update * args.rollout_hands,
                "learner_decisions": decisions,
                "rollout_mean_chips": float(profits.mean()),
                **metric,
            }
            if update % args.ppo_eval_every == 0 or update == args.ppo_updates:
                torch.set_num_threads(1)
                validation = evaluate_policy(
                    model,
                    args.validation_pairs,
                    VALID_DEAL_BASE,
                    730_000_000 + args.seed * 10000,
                )
                point["validation"] = profit_metrics(validation)
                if validation.mean() > best_score:
                    best_score, best_update = float(validation.mean()), update
                    save_policy(
                        directory / "ppo_best.pt",
                        model,
                        stage="bc_ppo",
                        seed=args.seed,
                        selected_update=update,
                    )
                save_policy(
                    directory / "ppo_last.pt",
                    model,
                    stage="bc_ppo",
                    seed=args.seed,
                    selected_update=update,
                )
                print(f"seed={args.seed} PPO update={update} {point}", flush=True)
            if update in args.ppo_snapshot_updates:
                # Keep selection within this budget; later validation cannot change it.
                for policy in ("best", "last"):
                    shutil.copyfile(
                        directory / f"ppo_{policy}.pt",
                        directory / f"ppo_{update}_{policy}.pt",
                    )
            curve.append(point)
            if update % 8 == 0:
                write_json(
                    directory / "progress.json",
                    {"curve": curve, "best_update": best_update},
                )
    write_json(
        directory / "training.json",
        {
            "seed": args.seed,
            "bc_selected_step": bc_step,
            "bc_test": bc_test,
            "ppo_selected_update": best_update,
            "total_training_hands": args.ppo_updates * args.rollout_hands,
            "learner_decisions": decisions,
            "bc_memory": bc_memory.result(),
            "ppo_memory": ppo_memory.result(),
            "curve": curve,
            "checkpoint_check": True,
            "source_sha256": source_hashes(),
        },
    )
    print(f"Training completed: {directory}", flush=True)


def evaluate(args):
    directory = args.out_dir / f"seed{args.seed}"
    output = directory / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    arrays, metrics = {}, {}
    for name in ("heuristic", "random", "bc", "ppo_best", "ppo_last"):
        model, saved = (
            (None, {})
            if name in ("heuristic", "random")
            else load_policy(directory / f"{name}.pt")
        )
        values = evaluate_policy(
            model,
            args.test_pairs,
            TEST_DEAL_BASE,
            830_000_000 + args.seed * 100_000,
            baseline=name,
        )
        arrays[name] = values
        metrics[name] = {
            **profit_metrics(values),
            "checkpoint_update": saved.get("selected_update"),
        }
        print(f"seed={args.seed} TEST {name}: {metrics[name]}", flush=True)
    np.testing.assert_array_equal(arrays["heuristic"].sum(1), 0)
    metrics["ppo_best_minus_bc"] = profit_metrics(arrays["ppo_best"] - arrays["bc"])
    metrics["ppo_last_minus_bc"] = profit_metrics(arrays["ppo_last"] - arrays["bc"])
    np.savez_compressed(directory / "evaluation.npz", **arrays)
    write_json(
        output,
        {
            "seed": args.seed,
            "deal_seed_base": TEST_DEAL_BASE,
            "metrics": metrics,
            "policy_decoding": "stochastic categorical for BC and PPO, same action RNG seeds, heuristic deterministic",
            "seconds": time.perf_counter() - started,
            "source_sha256": source_hashes(),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "evaluate"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-seed", type=int, default=20260919)
    parser.add_argument("--demonstrations", type=Path)
    parser.add_argument("--bc-hands", type=int, default=12000)
    parser.add_argument("--bc-steps", type=int, default=4000)
    parser.add_argument("--bc-eval-every", type=int, default=500)
    parser.add_argument("--ppo-updates", type=int, default=128)
    parser.add_argument("--ppo-eval-every", type=int, default=16)
    parser.add_argument("--ppo-snapshot-updates", type=int, nargs="*", default=[])
    parser.add_argument("--rollout-hands", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--test-pairs", type=int, default=4000)
    args = parser.parse_args(argv)
    if (
        args.bc_hands < 60
        or args.dim % 4
        or args.seed < 0
        or args.seed > 80
        or min(
            args.bc_steps,
            args.bc_eval_every,
            args.ppo_updates,
            args.ppo_eval_every,
            args.rollout_hands,
            args.ppo_epochs,
            args.batch,
            args.dim,
            args.layers,
            args.threads,
        )
        < 1
        or min(args.validation_pairs, args.test_pairs) < 2
    ):
        parser.error(
            "Need positive budgets, hands>=60, dim divisible by 4, pairs>=2, seed in [0,80]"
        )
    if args.ppo_updates * args.rollout_hands >= 1_000_000 or args.bc_hands >= 1_000_000:
        parser.error("Pilot deal seed partitions support fewer than 1M hands per run")
    if any(
        update < 1
        or update > args.ppo_updates
        or (update % args.ppo_eval_every and update != args.ppo_updates)
        for update in args.ppo_snapshot_updates
    ):
        parser.error("Snapshot updates must be evaluated updates within the budget")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    {"collect": collect, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
