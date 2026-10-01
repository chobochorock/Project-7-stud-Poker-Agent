"""Raw 85D event Transformer: outcome-sampling neural CFR vs PPO self-play.

Usage: python -m agents.state_action_embedding.raw_cfr_ppo --help
Input: original seven-poker v3 observer event histories; no skip-gram weights.
Output: fresh family data directory, SQLite metrics, model snapshots, replay NPZ.
Metric: paired chips/hand and pre-mask invalid mass, NOT exploitability.
Limit: high-variance sampled regrets, bounded replay; see RAW_CFR_PPO.md.
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
import sqlite3
import time

from . import action_window as window
from . import bc_ppo as ppo
from . import transformer_legality as transformer

np, torch, nn, F = ppo.np, ppo.torch, ppo.nn, ppo.F
FAMILY = Path(__file__).parent
RAW_REFERENCE = (
    FAMILY / "data/transformer_legality_20260928/transformer/raw/seed11/model.pt"
)


def prefix(game, seat):
    """Exactly the old six retained event kinds, before the requested action."""
    events = np.asarray(game.events[seat], dtype=np.int16)
    events = events[np.isin(events[:, 0], window.ACTION_KINDS)]
    if not 0 < len(events) <= transformer.MAX_EVENTS:
        raise ValueError("Invalid full event history; truncation is forbidden")
    return events


class RawPolicy(nn.Module):
    def __init__(self, normalization, dim=32, mode="ppo"):
        super().__init__()
        self.dim, self.mode = dim, mode
        self.history = transformer.HistoryTransformer(85, dim)
        self.history.head = nn.Linear(2 * dim, ppo.ACTION_COUNT)
        self.value_head = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.history.head.weight)
        nn.init.zeros_(self.history.head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer("mean", normalization["mean"].float().clone())
        self.register_buffer("std", normalization["std"].float().clone())
        self.register_buffer("paid_mean", torch.tensor(normalization["paid_mean"]))
        self.register_buffer("paid_std", torch.tensor(normalization["paid_std"]))
        mapping = torch.full((10,), -1, dtype=torch.long)
        mapping[list(window.ACTION_KINDS)] = torch.arange(6)
        self.register_buffer("kind_map", mapping)

    def encode(self, tokens):
        valid = tokens[..., 0] != 0
        rows = tokens[valid]
        kind, subject, card, bet, street = rows[:, :5].long().T
        environment = (
            (kind == window.ANTE) | (kind == window.PRIVATE) | (kind == window.PUBLIC)
        )
        card_present = (
            (kind == window.PRIVATE)
            | (kind == window.PUBLIC)
            | (kind == window.DISCARD)
            | (kind == window.REVEAL)
        )
        paid_present = (kind == window.ANTE) | (kind == window.BET)
        categories = (
            self.kind_map[kind],
            torch.where(environment, 2, subject),
            torch.where(environment, subject, 2),
            card,
            torch.where(kind == window.BET, bet - 1, 8),
            street,
        )
        if (categories[0] < 0).any():
            raise ValueError("Only the six raw event kinds are allowed")
        numeric = torch.where(
            paid_present, (rows[:, 5].float() - self.paid_mean) / self.paid_std, 0
        )
        x = torch.cat(
            [F.one_hot(v, n).float() for v, n in zip(categories, (6, 3, 3, 53, 9, 7))]
            + [
                torch.stack(
                    (
                        numeric,
                        card_present.float(),
                        (card > 0).float(),
                        paid_present.float(),
                    ),
                    1,
                )
            ],
            1,
        )
        output = torch.zeros((*tokens.shape[:2], 85), device=tokens.device)
        output[valid] = (x - self.mean) / self.std
        return output

    def raw_outputs(self, tokens, lengths):
        scores, hidden = self.history(self.encode(tokens))
        rows = torch.arange(len(tokens), device=tokens.device)
        return scores[rows, lengths - 1], self.value_head(
            hidden[rows, lengths - 1]
        ).squeeze(-1)

    def distribution_value(self, tokens, lengths, legal):
        scores, values = self.raw_outputs(tokens, lengths)
        if self.mode == "ppo":
            return (
                torch.distributions.Categorical(
                    logits=scores.masked_fill(~legal, -1e9)
                ),
                values,
            )
        positive = scores.clamp_min(0) * legal
        mass = positive.sum(-1, keepdim=True)
        probabilities = torch.where(
            mass > 0,
            positive / mass.clamp_min(1e-30),
            legal / legal.sum(-1, keepdim=True),
        )
        return torch.distributions.Categorical(probs=probabilities), values


def normalization():
    saved = torch.load(RAW_REFERENCE, weights_only=True, map_location="cpu")
    return dict(
        mean=saved["encoding"]["mean"],
        std=saved["encoding"]["std"],
        paid_mean=saved["normalizer"]["mean"],
        paid_std=saved["normalizer"]["std"],
    )


@torch.inference_mode()
def inference(model, tokens, legal):
    inputs = torch.from_numpy(tokens[None]).long()
    lengths = torch.tensor([len(tokens)])
    mask = torch.from_numpy(legal[None])
    scores, values = model.raw_outputs(inputs, lengths)
    if model.mode == "ppo":
        p = scores.masked_fill(~mask, -1e9).softmax(-1)[0].numpy().astype(float)
    else:
        p = np.maximum(scores[0].numpy().astype(float), 0) * legal
        if p.sum() == 0:
            p = legal.astype(float)
    p /= p.sum()
    return p, float(values.item()), scores[0].numpy()


def outcome_targets(rows, reward):
    """Unbiased per-infoset regret increment, zero if that infoset is unvisited.

    Opponent/chance are sampled on-policy, so their likelihood ratios cancel.
    Only traverser decisions are retained. Correct own prefix AND own tail;
    this is not terminal reward minus a baseline with a CFR name attached.
    """
    reach, prefixes = 1.0, []
    for row in rows:
        prefixes.append(reach)
        reach *= row["sampling"][row["action"]]
    value, samples = float(reward), []
    for row, prefix_reach in reversed(list(zip(rows, prefixes))):
        child_values = np.zeros(len(row["legal"]), np.float64)
        action = row["action"]
        child_values[action] = value / row["sampling"][action]
        value = float(row["probabilities"] @ child_values)
        target = np.where(row["legal"], (child_values - value) / prefix_reach, 0)
        if not np.isfinite(target).all():
            raise FloatingPointError("Nonfinite importance-weighted regret")
        samples.append(
            dict(
                tokens=row["tokens"],
                legal=row["legal"],
                target=target.astype(np.float32),
            )
        )
    return samples


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity, self.seen, self.rows = capacity, 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, rows):
        for row in rows:
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row)
            else:
                index = int(self.rng.integers(self.seen))
                if index < self.capacity:
                    self.rows[index] = row


def tensors(rows):
    lengths = torch.tensor([len(r["tokens"]) for r in rows])
    tokens = torch.zeros((len(rows), int(lengths.max()), 6), dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, : lengths[i]] = torch.from_numpy(row["tokens"])
    return tokens, lengths, torch.from_numpy(np.asarray([r["legal"] for r in rows]))


class PlayingAgent(ppo.PokerAgent):
    def __init__(self, game, seat, model, seed, records, exploration=0.0):
        super().__init__(f"p{seat}")
        self.game, self.seat, self.model = game, seat, model
        self.rng, self.records = np.random.default_rng(seed), records
        self.exploration = exploration

    def decide(self, valid=None):
        tokens, legal = prefix(self.game, self.seat), ppo.legal_mask(valid)
        probabilities, value, scores = inference(self.model, tokens, legal)
        sampling = (
            1 - self.exploration
        ) * probabilities + self.exploration * legal / legal.sum()
        action = int(self.rng.choice(ppo.ACTION_COUNT, p=sampling))
        if not legal[action]:
            raise AssertionError("Illegal execution; no fallback allowed")
        self.records.append(
            dict(
                tokens=tokens,
                legal=legal,
                action=action,
                value=value,
                log_prob=float(np.log(sampling[action])),
                probabilities=probabilities,
                sampling=sampling,
            )
        )
        return action

    def choose_action(self, state, valid_actions):
        return ppo.BETTING_ACTIONS[self.decide(valid_actions)]

    def choose_discard_and_reveal(self, hidden_cards):
        return ppo.DISCARD_PAIRS[self.decide() - 8]


def ppo_batch(model, hands, seed_base, action_seed):
    records, advantage, returns = [], [], []
    for hand in range(hands):
        game, trajectory = ppo.EventGame(), [[], []]
        agents = {
            f"p{s}": PlayingAgent(
                game, s, model, action_seed + 2 * hand + s, trajectory[s]
            )
            for s in range(2)
        }
        result = ppo.play(game, agents, seed_base + hand)
        for seat in range(2):
            adv, ret = ppo.gae(
                np.array([r["value"] for r in trajectory[seat]], np.float32),
                (result["final_chips"][f"p{seat}"] - ppo.STACK) / ppo.REWARD_SCALE,
            )
            records.extend(trajectory[seat])
            advantage.extend(adv)
            returns.extend(ret)
    data = ppo.pack(records)
    data.update(
        old_log_prob=np.array([r["log_prob"] for r in records], np.float32),
        advantage=np.asarray(advantage, np.float32),
        returns=np.asarray(returns, np.float32),
    )
    return data


def cfr_batch(model, hands, seed_base, action_seed, exploration):
    samples, decisions = [], 0
    for hand in range(hands):
        traverser = hand % 2
        game, records = ppo.EventGame(), [[], []]
        agents = {
            f"p{s}": PlayingAgent(
                game,
                s,
                model,
                action_seed + hand * 2 + s,
                records[s],
                exploration if s == traverser else 0.0,
            )
            for s in range(2)
        }
        result = ppo.play(game, agents, seed_base + hand)
        samples.extend(
            outcome_targets(
                records[traverser],
                (result["final_chips"][f"p{traverser}"] - ppo.STACK) / ppo.REWARD_SCALE,
            )
        )
        decisions += sum(map(len, records))
    return samples, decisions


def regret_loss(scores, target, legal, kind="mse"):
    """Equal history weights; root_mae=sqrt(MAE), root_mean=sqrt(signed mean)."""
    error = scores - target
    if kind == "root_mean":
        return torch.sqrt((error * legal).sum(1).div(legal.sum(1)).mean())
    if kind in ("l1", "root_mae"):
        mae = (error.abs() * legal).sum(1).div(legal.sum(1)).mean()
        # At exact fit, choose zero gradient instead of sqrt's singular derivative.
        return mae.sqrt() if kind == "root_mae" and mae.detach() > 0 else mae
    mse = (error.square() * legal).sum(1).div(legal.sum(1)).mean()
    if kind == "mse":
        return mse
    if kind == "rmse":
        return torch.sqrt(mse + 1e-12)
    raise ValueError(f"Unknown regret loss: {kind}")


def fit_cfr(model, reservoir, optimizer, args, rng, diagnostics=None):
    losses = []
    for _ in range(args.fit_steps):
        rows = [
            reservoir.rows[i]
            for i in rng.integers(len(reservoir.rows), size=args.batch)
        ]
        tokens, lengths, legal = tensors(rows)
        target = torch.from_numpy(np.asarray([r["target"] for r in rows]))
        scores, _ = model.raw_outputs(tokens, lengths)
        loss = regret_loss(scores, target, legal, getattr(args, "cfr_loss", "mse"))
        optimizer.zero_grad(set_to_none=True)
        grad_norm = torch.tensor(float("nan"))
        if torch.isfinite(loss):
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            mean_error = ((scores - target) * legal).sum(1).div(legal.sum(1)).mean()
            if getattr(args, "out_dir", None) is not None:
                np.savez_compressed(
                    args.out_dir / "failed_fit_batch.npz",
                    tokens=tokens.numpy(),
                    lengths=lengths.numpy(),
                    legal=legal.numpy(),
                    scores=scores.detach().numpy(),
                    target=target.numpy(),
                    mean_error=float(mean_error.detach()),
                    loss=float(loss.detach()),
                    grad_norm=float(grad_norm),
                    completed_fit_steps=len(losses),
                )
            raise FloatingPointError(
                f"CFR {getattr(args, 'cfr_loss', 'mse')}: signed_mean_error={float(mean_error.detach()):.12g}, "
                f"loss={float(loss.detach()):.12g}, grad_norm={float(grad_norm):.12g}, "
                f"completed_fit_steps={len(losses)}; stopped before invalid optimizer step"
            )
        optimizer.step()
        losses.append(float(loss.detach()))
        if diagnostics is not None:
            diagnostics.append((losses[-1], float(grad_norm)))
    return float(np.mean(losses))


def legal_panel(count):
    """Fixed held-out low-fold histories, actor's own view, action NOT in prefix."""
    with np.load(window.DEFAULT_CORPUS / "separate_raw.npz", allow_pickle=False) as raw:
        events, states = raw["action"], raw["state"]
        sequences = np.flatnonzero(raw["split"] == 1)
        rng = np.random.default_rng(928001)
        rows = []
        for i in rng.permutation(sequences):
            start, end = raw["action_offsets"][i : i + 2]
            lo, hi = raw["state_offsets"][i : i + 2]
            selected = events[start:end]
            bets = np.flatnonzero(selected[:, 0] == window.BET)
            if len(bets) != hi - lo:
                raise AssertionError("Corpus BET/state misalignment")
            own = np.flatnonzero(selected[bets, 1] == 0)
            if not len(own):
                continue
            index = int(rng.choice(own))
            tokens = selected[: bets[index]]
            tokens = tokens[np.isin(tokens[:, 0], window.ACTION_KINDS)].astype(np.int16)
            mask = np.zeros(20, bool)
            mask[:8] = states[lo + index, 213:221].astype(bool)
            if not mask[int(selected[bets[index], 3]) - 1]:
                raise AssertionError("Wrong pre-action legality label")
            rows.append(
                dict(
                    tokens=tokens,
                    legal=mask,
                    hand=int(raw["hand"][i]),
                    street=int(selected[bets[index], 4]),
                )
            )
            if len(rows) == count:
                break
    return rows


@torch.inference_mode()
def legality(model, panel):
    invalid, top_invalid = [], []
    for start in range(0, len(panel), 64):
        tokens, lengths, legal = tensors(panel[start : start + 64])
        scores, _ = model.raw_outputs(tokens, lengths)
        scores, legal = scores[:, :8], legal[:, :8]
        if model.mode == "ppo":
            probabilities = scores.softmax(1)
        else:
            positive = scores.clamp_min(0)
            mass = positive.sum(1, keepdim=True)
            probabilities = torch.where(
                mass > 0,
                positive / mass.clamp_min(1e-30),
                torch.full_like(positive, 1 / 8),
            )
        invalid.extend((probabilities * ~legal).sum(1).tolist())
        # Fraction of invalid actions among tied maxima; no arbitrary argmax.
        ties = probabilities == probabilities.max(1, keepdim=True).values
        top_invalid.extend(((ties & ~legal).sum(1) / ties.sum(1)).tolist())
    return dict(
        invalid_mass=float(np.mean(invalid)),
        invalid_top1=float(np.mean(top_invalid)),
        rows=len(panel),
    )


def evaluate(models, opponent, pairs, seed_base):
    profits = np.zeros((pairs, 2), np.int32)
    pick = np.random.default_rng(seed_base + 990001)
    for pair in range(pairs):
        model = models[int(pick.integers(len(models)))]
        for seat in range(2):
            game = ppo.EventGame()
            hero = PlayingAgent(
                game, seat, model, seed_base + pair * 2 + seat + 81000000, []
            )
            other = ppo.opponent_agent(
                game,
                1 - seat,
                None if opponent == "heuristic" else "random",
                seed_base + pair * 2 + 1 - seat + 81000000,
            )
            result = ppo.play(
                game, {hero.name: hero, other.name: other}, seed_base + pair
            )
            profits[pair, seat] = result["final_chips"][hero.name] - ppo.STACK
    return profits


def save_model(path, model, **extra):
    torch.save(
        dict(
            weights=model.state_dict(),
            dim=model.dim,
            mode=model.mode,
            normalization=dict(
                mean=model.mean,
                std=model.std,
                paid_mean=float(model.paid_mean),
                paid_std=float(model.paid_std),
            ),
            **extra,
        ),
        path,
    )


def load_model(path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = RawPolicy(saved["normalization"], saved["dim"], saved["mode"]).eval()
    model.load_state_dict(saved["weights"])
    return model


def record_eval(
    db,
    folder,
    model,
    snapshots,
    panel,
    step,
    seed,
    args,
    hands,
    decisions,
    seconds,
    final=False,
):
    invalid_current = legality(model, panel)
    for split, pairs, opponents, kinds, base in (
        ("monitor", args.monitor_pairs, ("heuristic",), ("current",), 928100000),
        *(
            (
                [
                    (
                        "test",
                        args.test_pairs,
                        ("heuristic", "random"),
                        ("initial", "current", "average"),
                        928200000,
                    )
                ]
            )
            if final
            else []
        ),
    ):
        for kind in kinds:
            selected = (
                [model]
                if kind == "current"
                else [snapshots[0]]
                if kind == "initial"
                else snapshots
            )
            invalid = (
                invalid_current
                if kind == "current"
                else legality(snapshots[0], panel)
                if kind == "initial"
                else None
            )
            for opponent in opponents:
                with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                    profits = evaluate(selected, opponent, pairs, base)
                metric = ppo.profit_metrics(profits)
                db.execute(
                    "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        model.mode,
                        seed,
                        step,
                        split,
                        kind,
                        opponent,
                        metric["chips_per_hand"],
                        *metric["pair_normal_95pct"],
                        hands,
                        decisions,
                        seconds,
                        json.dumps(invalid),
                    ),
                )
                np.savez_compressed(
                    folder / f"{split}_{kind}_{opponent}_{step:04d}.npz",
                    profits=profits,
                )
                db.commit()
                print(
                    f"{model.mode} seed={seed} step={step} {split}/{kind}/{opponent} chips={metric['chips_per_hand']:.3f} CI={metric['pair_normal_95pct']} invalid={None if invalid is None else round(invalid['invalid_mass'], 4)}",
                    flush=True,
                )


def run(args):
    methods = getattr(args, "methods", ["cfr", "ppo"])
    if (
        min(
            args.updates,
            args.hands,
            args.fit_steps,
            args.batch,
            args.capacity,
            args.eval_every,
            args.panel,
            args.monitor_pairs,
            args.test_pairs,
            args.ppo_epochs,
        )
        < 1
        or args.hands % 2
        or args.test_pairs < 2
        or args.monitor_pairs < 2
        or not 0 < args.exploration <= 1
        or not np.isfinite(args.lr)
        or args.lr <= 0
        or args.dim < 4
        or args.dim % 4
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
        or not methods
        or len(set(methods)) != len(methods)
        or not set(methods) <= {"cfr", "ppo"}
        or getattr(args, "cfr_loss", "mse")
        not in ("mse", "rmse", "l1", "root_mean", "root_mae")
    ):
        raise ValueError("Invalid budget/model/sampling configuration")
    if not args.out_dir.resolve().is_relative_to((FAMILY / "data").resolve()):
        raise ValueError("Outputs belong in the model family data directory")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    normal, panel = normalization(), legal_panel(args.panel)
    source_paths = [
        Path(__file__),
        Path(window.__file__),
        Path(ppo.__file__),
        Path(transformer.__file__),
        FAMILY / "action2vec.py",
        ppo.ROOT / "environments/seven_stud/poker_env.py",
    ]
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "raw_reference": str(RAW_REFERENCE),
        "raw_reference_sha256": hashlib.sha256(RAW_REFERENCE.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
        "torch": str(torch.__version__),
        "rules": ppo.BETTING_RULES_VERSION,
    }
    ppo.write_json(args.out_dir / "config.json", config)
    for path in source_paths:
        (args.out_dir / (path.stem + "_source.py")).write_bytes(path.read_bytes())
    panel_data = ppo.pack([{**row, "action": 0} for row in panel])
    del panel_data["action"]
    panel_data.update(
        hand=np.array([row["hand"] for row in panel]),
        street=np.array([row["street"] for row in panel]),
    )
    np.savez_compressed(args.out_dir / "legality_panel.npz", **panel_data)
    db = sqlite3.connect(args.out_dir / "metrics.sqlite")
    db.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE updates(method TEXT,seed INT,step INT,hands INT,decisions INT,train_seconds REAL,loss REAL,buffer_size INT,target_max REAL,optimizer_steps INT); CREATE TABLE evaluations(method TEXT,seed INT,step INT,split TEXT,policy TEXT,opponent TEXT,chips REAL,ci_low REAL,ci_high REAL,hands INT,decisions INT,train_seconds REAL,legality TEXT);"
    )
    db.execute(
        "CREATE TABLE fit_diagnostics(method TEXT,seed INT,step INT,fit_step INT,loss REAL,grad_norm REAL)"
    )
    db.execute("INSERT INTO metadata VALUES ('status','running')")
    db.commit()
    try:
        for seed in args.seeds:
            for method in methods:
                torch.manual_seed(seed)
                model = RawPolicy(normal, args.dim, method).eval()
                folder = args.out_dir / f"{method}_seed{seed}"
                folder.mkdir()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
                rng, reservoir = np.random.default_rng(seed + 101), Reservoir(
                    args.capacity, seed + 201
                )
                snapshots, hands, decisions, seconds = [], 0, 0, 0.0
                save_model(folder / "initial.pt", model)
                record_eval(
                    db,
                    folder,
                    model,
                    [model],
                    panel,
                    0,
                    seed,
                    args,
                    hands,
                    decisions,
                    seconds,
                )
                for step in range(1, args.updates + 1):
                    fit_diagnostics = []
                    snapshot = copy.deepcopy(model).eval().requires_grad_(False)
                    snapshots.append(snapshot)
                    save_model(folder / f"behavior_{step:04d}.pt", snapshot)
                    started = time.perf_counter()
                    deal_seed, action_seed = (
                        928000000000 + seed * 1000000 + hands,
                        929000000000 + seed * 1000000 + hands * 2,
                    )
                    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(
                        sink
                    ):
                        if method == "cfr":
                            rows, visited = cfr_batch(
                                model,
                                args.hands,
                                deal_seed,
                                action_seed,
                                args.exploration,
                            )
                            reservoir.add(rows)
                            loss = fit_cfr(
                                model, reservoir, optimizer, args, rng, fit_diagnostics
                            )
                            target_max = float(
                                max(np.abs(r["target"]).max() for r in rows)
                            )
                            opt_steps = args.fit_steps
                        else:
                            data = ppo_batch(model, args.hands, deal_seed, action_seed)
                            metric = ppo.update_ppo(
                                model, optimizer, data, rng, args, "cpu"
                            )
                            loss, visited, opt_steps, target_max = (
                                metric["policy_loss"],
                                len(data["action"]),
                                metric["optimizer_steps"],
                                0.0,
                            )
                    model.eval()
                    seconds += time.perf_counter() - started
                    hands += args.hands
                    decisions += visited
                    db.execute(
                        "INSERT INTO updates VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            method,
                            seed,
                            step,
                            hands,
                            decisions,
                            seconds,
                            loss,
                            len(reservoir.rows),
                            target_max,
                            opt_steps,
                        ),
                    )
                    db.executemany(
                        "INSERT INTO fit_diagnostics VALUES (?,?,?,?,?,?)",
                        [
                            (method, seed, step, index + 1, fit_loss, grad_norm)
                            for index, (fit_loss, grad_norm) in enumerate(
                                fit_diagnostics
                            )
                        ],
                    )
                    db.commit()
                    save_model(folder / "last.pt", model, step=step)
                    print(
                        f"{method} seed={seed} update={step} hands={hands} decisions={decisions} train_s={seconds:.1f} loss={loss:.4g} target_max={target_max:.4g}",
                        flush=True,
                    )
                    if step % args.eval_every == 0 or step == args.updates:
                        record_eval(
                            db,
                            folder,
                            model,
                            snapshots,
                            panel,
                            step,
                            seed,
                            args,
                            hands,
                            decisions,
                            seconds,
                            final=step == args.updates,
                        )
                if method == "cfr":
                    packed = ppo.pack([{**r, "action": 0} for r in reservoir.rows])
                    packed["target"] = np.array([r["target"] for r in reservoir.rows])
                    np.savez_compressed(folder / "regret_replay.npz", **packed)
        db.execute("UPDATE metadata SET value='completed' WHERE key='status'")
        db.commit()
    except BaseException as exc:
        db.execute(
            "UPDATE metadata SET value='interrupted_or_failed' WHERE key='status'"
        )
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('failure',?)", (str(exc),))
        db.commit()
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument(
        "--methods", nargs="+", choices=("cfr", "ppo"), default=["cfr", "ppo"]
    )
    parser.add_argument(
        "--cfr-loss",
        choices=("mse", "rmse", "l1", "root_mean", "root_mae"),
        default="mse",
    )
    for name, default in (
        ("updates", 16),
        ("hands", 256),
        ("fit-steps", 32),
        ("batch", 64),
        ("capacity", 20000),
        ("dim", 32),
        ("eval-every", 8),
        ("panel", 512),
        ("monitor-pairs", 64),
        ("test-pairs", 256),
        ("ppo-epochs", 4),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exploration", type=float, default=0.6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
