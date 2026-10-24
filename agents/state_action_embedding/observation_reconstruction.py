"""Reconstruct pre-bet observations from visible raw event prefixes.

Usage (project root, existing PyTorch Python 3.12 runtime):
    python -m agents.state_action_embedding.observation_reconstruction --out-dir RUN
    python -m agents.state_action_embedding.observation_reconstruction --out-dir RUN --resume
Output: family data/RUN, audited labels, checkpoints, SQLite metrics, 64D latents.
Limit: fixed initial stacks, behavior-policy data, no poker policy optimization.
See OBSERVATION_RECONSTRUCTION.md for budgets and interpretation boundaries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

from . import action2vec as events
from .action_window import ACTION_KINDS, DEFAULT_CORPUS, field_arrays
from .experiment import BETTING_ACTIONS, ROOT, STACK, np, torch, nn, F, write_json
from .separate_skipgram import play, state_features
from .transformer_legality import HistoryTransformer
from environments.seven_stud.poker_env import (
    ALL_CARDS,
    AGGRESSIVE_ACTIONS,
    PokerGame,
    get_public_betting_priority,
)

FAMILY = Path(__file__).parent
GROUPS = ("cards", "street", "actor", "legal", "chips", "counts", "allin")
WEIGHTS = (1, 1, 1, 1, 1, 1, 0.1)
CHIPS = (
    "pot",
    "highest",
    "own_stack",
    "own_invested",
    "own_bet",
    "opp_stack",
    "opp_invested",
    "opp_bet",
    "call",
)
COUNTS = ("raises", "own_bets", "cap")
UNITS = np.array([1000] * 9 + [6, 3, 3], np.float32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_observation(prefix):
    """Replay observable ledgers only; TURN and hidden opponent cards are forbidden."""
    if not np.isin(prefix[:, 0], ACTION_KINDS).all():
        raise ValueError("Audit may only read model-visible retained events")
    cards, stacks = events.reconstruct_prefix(prefix)
    street = int(prefix[-1, 4])
    bets = prefix[(prefix[:, 0] == events.BET) & (prefix[:, 4] == street)]
    first_seat = int(prefix[prefix[:, 0] == events.ANTE][0, 1])
    public = [[ALL_CARDS[i] for i in np.flatnonzero(cards[g])] for g in (1, 3)]
    actor = (
        1 - int(bets[-1, 1])
        if len(bets)
        else max(
            range(2),
            key=lambda p: (get_public_betting_priority(public[p]), p == first_seat),
        )
    )
    # Engine supplies public betting rules, not simulator snapshots or hidden cards.
    game = PokerGame(["self", "opponent"], log_file=None, game_mode="cash")
    game.street = events.STREETS[street]
    game.pot = int((STACK - stacks).sum())
    game.betting_history = [
        dict(
            street=game.street,
            actor_index=int(row[1]),
            action=BETTING_ACTIONS[row[3] - 1],
        )
        for row in bets
    ]
    game.raise_count = sum(
        row["action"] in AGGRESSIVE_ACTIONS for row in game.betting_history
    )
    for p, player in enumerate(game.players):
        player.chips = int(stacks[p])
        player.invested = STACK - player.chips
        player.current_bet = int(bets[bets[:, 1] == p, 5].sum())
        player.is_all_in = player.chips == 0
        player.public_cards = public[p]
    game.current_highest_bet = max(p.current_bet for p in game.players)
    game.players[0].hidden_cards = [ALL_CARDS[i] for i in np.flatnonzero(cards[0])]
    discard = np.flatnonzero(cards[2])
    game.players[0].discarded_card = ALL_CARDS[discard[0]] if len(discard) else None
    state = game.get_ai_state(
        game.players[0], game.get_valid_actions(game.players[actor])
    )
    return state_features(events.encode_event(state, "CHECK", actor == 0))


def load_data(path, normalization=None, audit=True):
    with np.load(path, allow_pickle=False) as source:
        raw = dict(source)
    packed, ends, sequence, starts = [], [], [], [0]
    max_error = 0.0
    for i, (left, right) in enumerate(
        zip(raw["action_offsets"][:-1], raw["action_offsets"][1:])
    ):
        stream = raw["action"][left:right]
        selected = np.isin(stream[:, 0], ACTION_KINDS)
        retained = stream[selected]
        bet_positions = np.flatnonzero(stream[:, 0] == events.BET)
        lo, hi = raw["state_offsets"][i : i + 2]
        if len(bet_positions) != hi - lo:
            raise AssertionError("State/BET count mismatch")
        cumulative = np.cumsum(selected)
        for k, position in enumerate(bet_positions):
            if stream[position - 1, 0] != events.TURN:
                raise AssertionError("BET must follow its alignment TURN")
            end = int(cumulative[position] - 1)
            target = raw["state"][lo + k]
            if stream[position, 3] - 1 != raw["betting_action"][lo + k]:
                raise AssertionError("Selected action alignment mismatch")
            if int(target[211:213].argmax()) != 1 - int(stream[position, 1]):
                raise AssertionError("Actor alignment mismatch")
            if not target[213 + int(stream[position, 3]) - 1]:
                raise AssertionError("Recorded action is illegal")
            if audit:
                expected = audit_observation(retained[:end])
                error = float(np.max(np.abs(expected - target)))
                max_error = max(max_error, error)
                if error > 1e-6:
                    raise AssertionError(
                        f"Reconstruction audit failed sequence={i} bet={k} error={error}"
                    )
            ends.append(end)
            sequence.append(i)
        packed.append(retained)
        starts.append(starts[-1] + len(retained))
    tokens = np.concatenate(packed)
    split_events = np.repeat(raw["split"], np.diff(starts))
    if normalization is None:
        train = split_events == 0
        paid = tokens[train & np.isin(tokens[:, 0], (events.ANTE, events.BET)), 5]
        normalizer = dict(mean=float(paid.mean()), std=float(paid.std()) or 1.0)
        features = field_arrays(tokens, normalizer)[0]
        mean, scale = features[train].mean(0), features[train].std(0)
        scale[scale < 1e-6] = 1
        normalization = dict(paid=normalizer, mean=mean, scale=scale)
    else:
        features = field_arrays(tokens, normalization["paid"])[0]
    features = (features - normalization["mean"]) / normalization["scale"]
    lengths = np.diff(starts)
    if lengths.max() > 128:
        raise ValueError("Context overflow: no truncation")
    x = np.zeros((len(packed), int(lengths.max()), 85), np.float32)
    for i, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
        x[i, : hi - lo] = features[lo:hi]
    sequence = np.asarray(sequence, np.int64)
    return dict(
        x=torch.from_numpy(x),
        y=torch.from_numpy(raw["state"]),
        ends=np.asarray(ends),
        sequence=sequence,
        hand=raw["hand"][sequence],
        split=raw["split"][sequence],
        offsets=raw["state_offsets"],
        normalization=normalization,
        audit_error=max_error,
        tokens=tokens,
        starts=np.asarray(starts),
    )


class ReconstructionModel(nn.Module):
    def __init__(self, kind="transformer", width=32):
        super().__init__()
        self.kind, self.width = kind, width
        if kind == "transformer":
            self.backbone = HistoryTransformer(85, width)
            self.backbone.head = nn.Identity()
        else:
            self.backbone = nn.Sequential(
                nn.Linear(85, width), nn.Tanh(), nn.Linear(width, width * 2), nn.Tanh()
            )
        self.head = nn.Linear(width * 2, 235)

    def forward(self, x, lengths):
        rows = torch.arange(len(x))
        if self.kind == "transformer":
            latent = self.backbone(x)[1][rows, lengths - 1]
        else:
            latent = self.backbone(x[rows, lengths - 1])
        return self.head(latent), latent


def statistics(target):
    rates = target[:, :208].mean(0)
    positive_weight = torch.where(rates > 0, (1 - rates) / rates.clamp_min(1e-6), 1.0)
    numeric = target[:, 221:233]
    scale = numeric.std(0, unbiased=False)
    scale[scale < 1e-6] = 1
    return dict(mean=numeric.mean(0), scale=scale, positive_weight=positive_weight)


def losses(output, target, stats):
    numeric = (target[:, 221:233] - stats["mean"]) / stats["scale"]
    groups = torch.stack(
        [
            F.binary_cross_entropy_with_logits(
                output[:, :208],
                target[:, :208],
                pos_weight=stats["positive_weight"],
                reduction="none",
            ).mean(1),
            F.cross_entropy(
                output[:, 208:211], target[:, 208:211].argmax(1), reduction="none"
            ),
            F.cross_entropy(
                output[:, 211:213], target[:, 211:213].argmax(1), reduction="none"
            ),
            F.binary_cross_entropy_with_logits(
                output[:, 213:221], target[:, 213:221], reduction="none"
            ).mean(1),
            (output[:, 221:230] - numeric[:, :9]).square().mean(1),
            (output[:, 230:233] - numeric[:, 9:]).square().mean(1),
            F.binary_cross_entropy_with_logits(
                output[:, 233:235], target[:, 233:235], reduction="none"
            ).mean(1),
        ],
        1,
    )
    return groups


def decode(output, stats):
    result = output.clone()
    for lo, hi in ((0, 208), (213, 221), (233, 235)):
        result[:, lo:hi] = output[:, lo:hi].sigmoid()
    for lo, hi in ((208, 211), (211, 213)):
        result[:, lo:hi] = output[:, lo:hi].softmax(1)
    result[:, 221:233] = output[:, 221:233] * stats["scale"] + stats["mean"]
    return result


def batch(data, rows):
    lengths = torch.from_numpy(data["ends"][rows])
    x = data["x"][data["sequence"][rows], : int(lengths.max())]
    return x, lengths, data["y"][rows]


def sampled_rows(data, hands, rng, count):
    chosen = rng.choice(hands, count) * 2 + rng.integers(0, 2, count)
    left, right = data["offsets"][chosen], data["offsets"][chosen + 1]
    return left + (rng.random(count) * (right - left)).astype(np.int64)


def hand_macro(values, hands):
    _, inverse = np.unique(hands, return_inverse=True)
    count = np.bincount(inverse)
    return np.array(
        [
            np.mean(np.bincount(inverse, weights=values[:, j]) / count)
            for j in range(values.shape[1])
        ]
    )


@torch.inference_mode()
def predict(model, data, rows, stats):
    model.eval()
    predictions, latent, group_losses = [], [], []
    for start in range(0, len(rows), 256):
        x, lengths, target = batch(data, rows[start : start + 256])
        output, z = model(x, lengths)
        predictions.append(decode(output, stats).numpy())
        latent.append(z.numpy())
        group_losses.append(losses(output, target, stats).numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(latent),
        np.concatenate(group_losses),
    )


def metric_rows(prediction, truth):
    result = {}
    matches = []
    for i, name in enumerate(("hidden", "public", "discard", "opp_public")):
        p, t = (
            prediction[:, i * 52 : (i + 1) * 52] >= 0.5,
            truth[:, i * 52 : (i + 1) * 52] > 0.5,
        )
        tp = (p & t).sum(1)
        result[f"{name}_precision"] = tp / np.maximum(1, p.sum(1))
        result[f"{name}_recall"] = tp / np.maximum(1, t.sum(1))
        result[f"{name}_f1"] = 2 * tp / np.maximum(1, p.sum(1) + t.sum(1))
        result[f"{name}_exact"] = (p == t).all(1)
        matches.append(result[f"{name}_exact"])
    for name, lo, hi in (("street", 208, 211), ("actor", 211, 213)):
        result[f"{name}_accuracy"] = prediction[:, lo:hi].argmax(1) == truth[
            :, lo:hi
        ].argmax(1)
        matches.append(result[f"{name}_accuracy"])
    p, t = prediction[:, 213:221] >= 0.5, truth[:, 213:221] > 0.5
    result["legal_exact"] = (p == t).all(1)
    result["legal_false_positive"] = (p & ~t).mean(1)
    result["legal_false_negative"] = (~p & t).mean(1)
    matches.append(result["legal_exact"])
    for j, name in enumerate(BETTING_ACTIONS):
        result[f"legal_{name}_fp"] = p[:, j] & ~t[:, j]
        result[f"legal_{name}_fn"] = ~p[:, j] & t[:, j]
    error = (prediction[:, 221:233] - truth[:, 221:233]) * UNITS
    for j, name in enumerate((*CHIPS, *COUNTS)):
        result[f"{name}_mae"] = np.abs(error[:, j])
    result["chip_mae"] = np.abs(error[:, :9]).mean(1)
    result["chip_mse"] = np.square(error[:, :9]).mean(1)
    result["count_mae"] = np.abs(error[:, 9:]).mean(1)
    result["count_exact"] = (
        np.rint(prediction[:, 230:233] * UNITS[9:])
        == np.rint(truth[:, 230:233] * UNITS[9:])
    ).all(1)
    result["observation_exact_no_allin"] = np.stack(matches, 1).all(1) & (
        np.abs(error) <= 0.5
    ).all(1)
    q = prediction[:, 221:230] * 1000
    result["pot_consistency_mae"] = np.abs(q[:, 0] - q[:, 3] - q[:, 6])
    result["stack_consistency_mae"] = (
        np.abs(q[:, 2] + q[:, 3] - 1000) + np.abs(q[:, 5] + q[:, 6] - 1000)
    ) / 2
    result["highest_consistency_mae"] = np.abs(q[:, 1] - np.maximum(q[:, 4], q[:, 7]))
    result["call_consistency_mae"] = np.abs(q[:, 8] - np.maximum(0, q[:, 1] - q[:, 4]))
    result["card_collision"] = (
        (prediction[:, :208] >= 0.5).reshape(-1, 4, 52).sum(1) > 1
    ).any(1)
    return result


def metrics(prediction, truth, hands, lengths):
    rows = metric_rows(prediction, truth)
    names = list(rows)
    values = np.column_stack(list(rows.values())).astype(float)
    scopes = dict(
        all=np.ones(len(truth), bool), own=truth[:, 212] == 1, other=truth[:, 211] == 1
    )
    scopes.update({f"street_{s+5}": truth[:, 208 + s] == 1 for s in range(3)})
    scopes.update(
        short=lengths < 25, medium=(lengths >= 25) & (lengths < 35), long=lengths >= 35
    )
    result = {}
    for name, selection in scopes.items():
        if not selection.any():
            continue
        result[name] = dict(
            n=int(selection.sum()),
            hand_macro=dict(
                zip(names, hand_macro(values[selection], hands[selection]).tolist())
            ),
            micro=dict(zip(names, values[selection].mean(0).tolist())),
        )
    rare_p, rare_t = prediction[:, 233:235] >= 0.5, truth[:, 233:235] > 0.5
    tp = (rare_p & rare_t).sum(0)
    result["allin_auxiliary"] = dict(
        positive=rare_t.sum(0).tolist(),
        predicted=rare_p.sum(0).tolist(),
        recall=(tp / np.maximum(1, rare_t.sum(0))).tolist(),
        precision=(tp / np.maximum(1, rare_p.sum(0))).tolist(),
    )
    # Bootstrap whole hands, not correlated prefixes; conditional on this trained model.
    selection = scopes["own"]
    unique = np.unique(hands[selection])
    primary = ("hidden_exact", "legal_exact", "chip_mae")
    per_hand = np.array(
        [[rows[k][selection & (hands == h)].mean() for k in primary] for h in unique]
    )
    rng = np.random.default_rng(771)
    estimates = per_hand[rng.integers(0, len(unique), (400, len(unique)))].mean(1)
    result["own_hand_bootstrap95"] = {
        k: np.quantile(estimates[:, j], [0.025, 0.975]).tolist()
        for j, k in enumerate(primary)
    }
    return result


def record(db, condition, seed, step, split, values):
    db.execute(
        "INSERT INTO metrics VALUES (?,?,?,?,?,?)",
        (
            condition,
            seed,
            step,
            split,
            time.time(),
            json.dumps(values, allow_nan=False),
        ),
    )
    db.commit()


def checkpoint(model, stats, data, step):
    return dict(
        model=model.state_dict(),
        kind=model.kind,
        width=model.width,
        statistics=stats,
        normalization={
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in data["normalization"].items()
        },
        step=step,
    )


def train(data, directory, kind, seed, steps, db, args, tiny=False):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ReconstructionModel(kind, args.width)
    hands = np.unique(data["hand"][data["split"] == 0])
    if tiny:
        hands = np.random.default_rng(11).choice(hands, 32, replace=False)
    training = np.flatnonzero(np.isin(data["hand"], hands))
    validation = training if tiny else np.flatnonzero(data["split"] == 1)
    stats = statistics(data["y"][training])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best, stale, first = float("inf"), 0, 0
    last = directory / "last.pt"
    if args.resume and last.exists():
        saved = torch.load(last, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        stats, first, best, stale = (
            saved["statistics"],
            saved["step"],
            saved["best_loss"],
            saved["stale"],
        )
        rng.bit_generator.state = json.loads(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
    weight = torch.tensor(WEIGHTS)
    started = time.perf_counter()
    label = f"tiny_{kind}" if tiny else kind
    for step in range(first + 1, steps + 1):
        rows = sampled_rows(data, hands, rng, args.batch)
        x, lengths, target = batch(data, rows)
        model.train()
        output, _ = model(x, lengths)
        group = losses(output, target, stats).mean(0)
        loss = (group * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(grad) or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss/gradient")
        project = model.backbone.project if kind == "transformer" else model.backbone[0]
        project_grad = float(project.weight.grad.norm())
        optimizer.step()
        if step % args.log_every == 0 or step == steps:
            pred, _, val_losses = predict(model, data, validation, stats)
            mean_losses = hand_macro(val_losses, data["hand"][validation])
            val_loss = float(np.dot(mean_losses, WEIGHTS) / sum(WEIGHTS))
            improved = val_loss < best
            best, stale = (val_loss, 0) if improved else (best, stale + 1)
            payload = checkpoint(model, stats, data, step)
            if improved:
                torch.save(payload, directory / "best.pt")
            payload.update(
                optimizer=optimizer.state_dict(),
                sampler_rng=json.dumps(rng.bit_generator.state),
                torch_rng=torch.get_rng_state(),
                best_loss=best,
                stale=stale,
                train_hands=torch.from_numpy(hands),
            )
            torch.save(payload, last)
            score_rows = metric_rows(pred, data["y"][validation].numpy())
            scores = {
                key: float(hand_macro(value[:, None], data["hand"][validation])[0])
                for key, value in score_rows.items()
                if key
                in (
                    "hidden_exact",
                    "public_exact",
                    "discard_exact",
                    "opp_public_exact",
                    "legal_exact",
                    "chip_mae",
                    "count_exact",
                    "observation_exact_no_allin",
                )
            }
            entry = dict(
                loss=val_loss,
                group_loss=dict(zip(GROUPS, mean_losses.tolist())),
                train_group_loss=dict(zip(GROUPS, group.detach().tolist())),
                preclip_gradient=grad,
                clipped_project_gradient=project_grad,
                seconds=time.perf_counter() - started,
                **scores,
            )
            record(db, label, seed, step, "train" if tiny else "validation", entry)
            print(
                f"{label} seed={seed} step={step} loss={val_loss:.4f} card={scores['hidden_exact']:.3f} legal={scores['legal_exact']:.3f} chipMAE={scores['chip_mae']:.2f}",
                flush=True,
            )
            if not tiny and stale >= 8:
                break
    saved = torch.load(directory / "best.pt", weights_only=True)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["statistics"], saved["step"], training


def collect_fresh(path, hands, seed, cohort="lowfold"):
    """No recalibration and no selection using reconstruction outcomes."""
    from agent.heuristic_agent import HeuristicPokerAgent

    game = events.EventGame()
    states, tokens, so, ao, rewards = [], [], [0], [0], []
    rates = np.array([0.02625, 0.06125, 0.0875])
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in range(hands):
            if cohort == "lowfold":
                result, _ = play(game, seed + hand, rates)
            else:
                events.random.seed(seed + hand)
                agents = {
                    f"p{i}": (
                        events.UniformAgent(f"p{i}", seed + 2 * hand + i)
                        if cohort == "uniform"
                        else HeuristicPokerAgent(f"p{i}")
                    )
                    for i in range(2)
                }
                result = game.play_hand(agents)
            for viewer in range(2):
                state = state_features(np.asarray(game.raw_states)[:, viewer])
                stream = np.asarray(game.events[viewer], np.int16)
                states.append(state)
                tokens.append(stream)
                so.append(so[-1] + len(state))
                ao.append(ao[-1] + len(stream))
                rewards.append(result["final_chips"][f"p{viewer}"] - STACK)
    action = np.concatenate(tokens)
    np.savez_compressed(
        path,
        state=np.concatenate(states),
        action=action,
        state_offsets=np.asarray(so),
        action_offsets=np.asarray(ao),
        betting_action=action[action[:, 0] == events.BET, 3] - 1,
        hand=np.repeat(np.arange(hands), 2),
        split=np.ones(2 * hands, np.int8),
        return_chips=np.asarray(rewards),
    )
    write_json(
        path.with_suffix(".json"),
        dict(
            hands=hands,
            seed=seed,
            cohort=cohort,
            fold_rates=rates.tolist() if cohort == "lowfold" else None,
        ),
    )


def export(model, stats, data, directory, split, rows):
    pred, latent, _ = predict(model, data, rows, stats)
    truth = data["y"][rows].numpy()
    score = metrics(pred, truth, data["hand"][rows], data["ends"][rows])
    np.savez_compressed(
        directory / f"{split}_latents.npz",
        latent=latent,
        predicted=pred,
        truth=truth,
        hand=data["hand"][rows],
        rows=rows,
        lengths=data["ends"][rows],
    )
    write_json(directory / f"{split}_metrics.json", score)
    return score


def run(args):
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to((FAMILY / "data").resolve()):
        raise ValueError("Outputs must remain in family data/")
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory, or --resume")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    source = args.corpus / "separate_raw.npz"
    config = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "resume"
        },
        data_sha256=sha(source),
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                FAMILY / "transformer_legality.py",
                FAMILY / "action_window.py",
                FAMILY / "action2vec.py",
                FAMILY / "experiment.py",
                FAMILY / "separate_skipgram.py",
                ROOT / "environments/seven_stud/poker_env.py",
            )
        },
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device="cpu"),
        initial_condition="fixed 1000 chips / ante 1; no state or TURN input",
        metric="observation reconstruction and clustering diagnostics, NOT strategy/exploitability",
    )
    if args.resume:
        old = json.loads((args.out_dir / "config.json").read_text())
        if old != config:
            raise ValueError(
                "Resume requires unchanged arguments, code and source data"
            )
    else:
        write_json(args.out_dir / "config.json", config)
        for relative in config["source_sha256"]:
            destination = args.out_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
    print("Auditing every original pre-bet observation...", flush=True)
    data = load_data(source)
    write_json(
        args.out_dir / "audit.json",
        dict(
            rows=len(data["y"]),
            max_absolute_error=data["audit_error"],
            retained_events=len(data["tokens"]),
            max_retained_length=int(np.diff(data["starts"]).max()),
            allowed_kinds=list(ACTION_KINDS),
            labels_from_turn_alignment_only=True,
        ),
    )
    summary = []
    with sqlite3.connect(args.out_dir / "metrics.sqlite") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS metrics (condition TEXT, seed INTEGER, step INTEGER, split TEXT, timestamp REAL, values_json TEXT)"
        )
        model, stats, step, training = train(
            data,
            args.out_dir / "tiny",
            "transformer",
            11,
            args.tiny_steps,
            db,
            args,
            tiny=True,
        )
        tiny = export(model, stats, data, args.out_dir / "tiny", "train", training)
        write_json(args.out_dir / "tiny_result.json", tiny)
        diagnostic_hands = np.random.default_rng(12).choice(
            np.unique(data["hand"][data["split"] == 1]), 32, replace=False
        )
        export(
            model,
            stats,
            data,
            args.out_dir / "tiny",
            "heldout32",
            np.flatnonzero(np.isin(data["hand"], diagnostic_hands)),
        )
        if args.tiny_only:
            print(f"Tiny-only diagnostics: {args.out_dir}", flush=True)
            return
        for kind in ("last_event", "transformer"):
            for seed in args.seeds:
                directory = args.out_dir / kind / f"seed{seed}"
                model, stats, step, _ = train(
                    data, directory, kind, seed, args.steps, db, args
                )
                valid = np.flatnonzero(data["split"] == 1)
                score = export(model, stats, data, directory, "validation", valid)
                record(db, kind, seed, step, "validation_final", score)
                if kind == "transformer":
                    train_rows = np.flatnonzero(data["split"] == 0)
                    export(model, stats, data, directory, "train", train_rows)
                summary.append(
                    dict(
                        kind=kind,
                        seed=seed,
                        step=step,
                        parameters=sum(p.numel() for p in model.parameters()),
                        validation=score,
                    )
                )
        train_mask = data["split"] == 0
        train_y = data["y"][train_mask].numpy()
        _, ids = np.unique(data["hand"][train_mask], return_inverse=True)
        constant = np.average(train_y, axis=0, weights=1 / np.bincount(ids)[ids])
        for cohort, count, seed_base in (
            ("lowfold", args.test_hands, 915200000),
            ("uniform", args.shift_hands, 916200000),
            ("heuristic", args.shift_hands, 917200000),
        ):
            if count <= 0:
                continue
            path = args.out_dir / f"fresh_{cohort}.npz"
            if not path.exists():
                collect_fresh(path, count, seed_base, cohort)
            fresh = load_data(path, data["normalization"])
            write_json(
                args.out_dir / f"fresh_{cohort}_audit.json",
                dict(
                    rows=len(fresh["y"]),
                    max_absolute_error=fresh["audit_error"],
                    sha256=sha(path),
                ),
            )
            rows = np.arange(len(fresh["y"]))
            constant_score = metrics(
                np.broadcast_to(constant, fresh["y"].shape),
                fresh["y"].numpy(),
                fresh["hand"],
                fresh["ends"],
            )
            write_json(args.out_dir / f"constant_{cohort}.json", constant_score)
            for result in summary:
                directory = args.out_dir / result["kind"] / f"seed{result['seed']}"
                saved = torch.load(directory / "best.pt", weights_only=True)
                model = ReconstructionModel(saved["kind"], saved["width"])
                model.load_state_dict(saved["model"])
                score = export(
                    model, saved["statistics"], fresh, directory, f"test_{cohort}", rows
                )
                record(
                    db,
                    result["kind"],
                    result["seed"],
                    saved["step"],
                    f"test_{cohort}",
                    score,
                )
                result[f"test_{cohort}"] = score
            del fresh
    write_json(args.out_dir / "summary.json", summary)
    print(f"Finished: {args.out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--tiny-steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--test-hands", type=int, default=1000)
    parser.add_argument("--shift-hands", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.tiny_steps,
            args.batch,
            args.log_every,
            args.test_hands,
            args.threads,
        )
        <= 0
        or args.shift_hands < 0
    ):
        parser.error("Budgets must be positive (shift-hands may be zero)")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Duplicate seeds")
    run(args)


if __name__ == "__main__":
    main()
