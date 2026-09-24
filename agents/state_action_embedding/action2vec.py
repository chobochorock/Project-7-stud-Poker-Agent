"""Action/chance event pretraining and honest pre-terminal reward forecasting.

Usage (project root, Python 3.12):
    python -m agents.state_action_embedding.action2vec collect --out-dir RUN
    python -m agents.state_action_embedding.action2vec train --out-dir RUN --method players
    python -m agents.state_action_embedding.action2vec train --out-dir RUN --method all
    python -m agents.state_action_embedding.action2vec train --out-dir RUN --method none

Input: seven-poker v3 heads-up cash, random / existing heuristic / mixed opponents.
Output: event sequences, inference checkpoints, held-out predictions and metrics.
Target: opponent betting actions after observed chance, and terminal net chips
predicted BEFORE a sampled betting action. No future cards/payouts are inputs.
Players-only pretraining masks direct chance loss, not chance context/gradients.
The all baseline adds visible chance-card prediction, not private oracle labels.
This is forecasting under collection policies, NOT optimal Q or exploitability.
See ACTION2VEC.md for target timing, baselines, and measurement limitations.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import time

from .experiment import (
    BETTING_ACTIONS,
    BETTING_RULES_VERSION,
    CARD_IDS,
    MemoryMeter,
    PokerGame,
    ROOT,
    STACK,
    UniformAgent,
    encode_event,
    nn,
    np,
    torch,
    F,
    write_json,
)
from agent.heuristic_agent import HeuristicPokerAgent


PAD, START, ANTE, PRIVATE, PUBLIC, DISCARD, REVEAL, STREET, TURN, BET = range(10)
KINDS = (
    "pad",
    "start",
    "ante",
    "private_deal",
    "public_deal",
    "discard",
    "reveal",
    "street",
    "turn",
    "bet",
)
STREETS = ("setup", "ante", "discard_reveal", "4th", "5th", "6th", "7th_hidden")
COHORTS = ("random", "heuristic", "mixed")
MAX_LENGTH = 128


def viewed_event(
    kind: int, subject: int, card: int, action: int, street: int, paid: int, viewer: int
) -> tuple[int, ...]:
    """Private deals/discards keep their existence but hide opponents' values."""
    if kind in (PRIVATE, DISCARD) and subject != viewer:
        card = 0
    relative = 2 if subject == 2 else int(subject != viewer)
    return kind, relative, card, action, street, paid


class EventGame(PokerGame):
    """Instrument existing engine hooks; no replacement of betting/dealing rules."""

    def __init__(self):
        super().__init__(["p0", "p1"], log_file=None, game_mode="cash")
        self.events = [[], []]
        self.legal = []
        self.decisions = []
        self.raw_states = []

    def emit(
        self,
        kind: int,
        subject: int = 2,
        card: int = 0,
        action: int = 0,
        paid: int = 0,
        legal=None,
    ) -> None:
        street = STREETS.index(self.street)
        for viewer in range(2):
            self.events[viewer].append(
                viewed_event(kind, subject, card, action, street, paid, viewer)
            )
        self.legal.append([a in (legal or []) for a in BETTING_ACTIONS])

    def start_game(self) -> None:
        self.events, self.legal, self.decisions, self.raw_states = [[], []], [], [], []
        self.street = "setup"
        self.emit(START, paid=self.starting_chips)
        super().start_game()

    def _commit_chips(self, player, requested_amount, count_for_round):
        paid = super()._commit_chips(player, requested_amount, count_for_round)
        if not count_for_round:
            self.emit(ANTE, self.players.index(player), paid=paid)
        return paid

    def log_global_state(self, event_message: str = "") -> None:
        if event_message == "initial ante and four hidden cards":
            for card_index in range(4):
                for seat, player in enumerate(self.players):
                    self.emit(
                        PRIVATE,
                        seat,
                        CARD_IDS[str(player.hidden_cards[card_index])] + 1,
                    )
        elif event_message == "discard one card and reveal one card":
            for seat, player in enumerate(self.players):
                self.emit(DISCARD, seat, CARD_IDS[str(player.discarded_card)] + 1)
                self.emit(REVEAL, seat, CARD_IDS[str(player.public_cards[0])] + 1)

    def deal_cards_to_active(self, is_public: bool = True) -> None:
        self.emit(STREET)
        before = [len(p.get_all_cards()) for p in self.players]
        super().deal_cards_to_active(is_public)
        for seat, player in enumerate(self.players):
            if len(player.get_all_cards()) > before[seat]:
                card = (player.public_cards if is_public else player.hidden_cards)[-1]
                self.emit(
                    PUBLIC if is_public else PRIVATE, seat, CARD_IDS[str(card)] + 1
                )

    def apply_action(self, player, action):
        seat = self.players.index(player)
        legal = self.get_valid_actions(player)
        self.emit(TURN, seat)
        self.decisions.append(len(self.legal) - 1)
        # A constant dummy action field is used only by the raw-state baseline.
        self.raw_states.append(
            [
                encode_event(
                    self.get_ai_state(viewer, legal), "CHECK", viewer is player
                )
                for viewer in self.players
            ]
        )
        before = player.chips
        raised = super().apply_action(player, action)
        self.emit(
            BET,
            seat,
            action=BETTING_ACTIONS.index(action) + 1,
            paid=before - player.chips,
            legal=legal,
        )
        return raised


def reconstruct_prefix(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Audit observable cards and chip stacks from events, not engine snapshots."""
    cards = np.zeros((4, 52), np.float32)
    stacks = np.array([STACK, STACK], dtype=np.int32)
    for kind, subject, card, action, street, paid in events:
        if kind in (ANTE, BET):
            stacks[subject] -= paid
        if card == 0:
            continue
        if kind == PRIVATE and subject == 0:
            cards[0, card - 1] = 1
        elif kind == PUBLIC:
            cards[1 if subject == 0 else 3, card - 1] = 1
        elif kind == DISCARD and subject == 0:
            cards[0, card - 1] = 0
            cards[2, card - 1] = 1
        elif kind == REVEAL:
            if subject == 0:
                cards[0, card - 1] = 0
            cards[1 if subject == 0 else 3, card - 1] = 1
    return cards, stacks


def collect(args: argparse.Namespace) -> None:
    path = args.out_dir / "events.npz"
    if path.exists():
        raise FileExistsError(path)
    started = time.perf_counter()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    splits = np.zeros(args.hands, np.int8)
    cohorts = np.arange(args.hands) % 3
    for cohort in range(3):
        rows = rng.permutation(np.flatnonzero(cohorts == cohort))
        splits[rows[int(len(rows) * 0.8) : int(len(rows) * 0.9)]] = 1
        splits[rows[int(len(rows) * 0.9) :]] = 2
    sequences, legal, anchors, raw, rewards = [], [], [], [], []
    random_agents = {
        f"p{i}": UniformAgent(f"p{i}", args.seed + i + 2) for i in range(2)
    }
    heuristic_agents = {f"p{i}": HeuristicPokerAgent(f"p{i}") for i in range(2)}
    game = EventGame()
    with open(os.devnull, "w") as sink:
        for hand in range(args.hands):
            cohort = cohorts[hand]
            if cohort == 0:
                agents = random_agents
            elif cohort == 1:
                agents = heuristic_agents
            else:
                heuristic_seat = (hand // 3) % 2
                agents = {
                    f"p{i}": (
                        heuristic_agents if i == heuristic_seat else random_agents
                    )[f"p{i}"]
                    for i in range(2)
                }
            with contextlib.redirect_stdout(sink):
                result = game.play_hand(agents)
            if sum(result["final_chips"].values()) != 2 * STACK:
                raise AssertionError("Chip conservation failed")
            choice = rng.integers(len(game.decisions))
            anchor = game.decisions[choice]
            for viewer in range(2):
                sequence = np.asarray(game.events[viewer], np.int16)
                current = game.raw_states[choice][viewer]
                cards, stacks = reconstruct_prefix(sequence[: anchor + 1])
                np.testing.assert_array_equal(cards.ravel(), current[:208])
                np.testing.assert_allclose(
                    stacks / STACK, current[[-12, -9]], atol=1e-6
                )
                sequences.append(sequence)
                legal.append(game.legal.copy())
                anchors.append(anchor)
                raw.append(current)
                rewards.append(result["final_chips"][f"p{viewer}"] - STACK)
            if (hand + 1) % 1500 == 0:
                print(f"collected {hand + 1}/{args.hands} hands", flush=True)
    lengths = np.asarray([len(s) for s in sequences], np.int16)
    width = int(lengths.max())
    if width > MAX_LENGTH:
        raise ValueError(f"Full histories exceed model context {MAX_LENGTH}: {width}")
    tokens = np.zeros((len(sequences), width, 6), np.int16)
    masks = np.zeros((len(sequences), width, 8), bool)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = sequence
        masks[row, : len(sequence)] = legal[row]
    arrays = {
        "tokens": tokens,
        "legal": masks,
        "lengths": lengths,
        "anchor": np.asarray(anchors, np.int16),
        "raw": np.asarray(raw, np.float32),
        "reward": np.asarray(rewards, np.float32),
        "hand": np.repeat(np.arange(args.hands), 2),
        "split": np.repeat(splits, 2),
        "cohort": np.repeat(cohorts, 2),
    }
    np.savez_compressed(path, **arrays)
    write_json(
        args.out_dir / "dataset.json",
        {
            "hands": args.hands,
            "seed": args.seed,
            "rules_version": BETTING_RULES_VERSION,
            "game": "seven-poker v3 heads-up cash, 1000 chips, ante 1",
            "cohorts": {
                name: int((cohorts == i).sum()) for i, name in enumerate(COHORTS)
            },
            "split_hands": [int((splits == i).sum()) for i in range(3)],
            "mean_events": float(lengths.mean()),
            "max_events": width,
            "player_view_events": int(lengths.sum()),
            "bytes_compressed": path.stat().st_size,
            "bytes_arrays": sum(a.nbytes for a in arrays.values()),
            "seconds": time.perf_counter() - started,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    ROOT / "environments/seven_stud/poker_env.py",
                    ROOT / "agents/baselines/heuristic_agent.py",
                )
            },
            "timing": "reward anchor is TURN before its betting action; no payouts/showdown inputs",
        },
    )
    print(path, flush=True)


class EventTransformer(nn.Module):
    def __init__(self, dim: int = 64, layers: int = 2):
        super().__init__()
        if dim < 4 or dim % 4 or layers < 1:
            raise ValueError("Need dim divisible by four and at least one layer")
        self.dim = dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, dim) for size in (10, 3, 53, 9, 7)]
        )
        self.amount = nn.Linear(1, dim, bias=False)
        self.position = nn.Embedding(MAX_LENGTH, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, 4, dim * 2, dropout=0.0, batch_first=True, norm_first=True
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, 8)
        self.chance_head = nn.Linear(dim, 52)
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1)
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        width = tokens.shape[1]
        if width > MAX_LENGTH:
            raise ValueError("Context truncation is not allowed in this experiment")
        local = sum(
            embedding(tokens[..., index])
            for index, embedding in enumerate(self.embeddings)
        )
        amount = torch.log1p(tokens[..., 5:6].float()) / math.log1p(STACK)
        hidden = self.input_norm(local + self.amount(amount))
        hidden = hidden + self.position(torch.arange(width, device=tokens.device))
        future = torch.ones(width, width, dtype=torch.bool, device=tokens.device).triu(
            1
        )
        padding = tokens[..., 0] == PAD
        for layer in self.layers:
            hidden = layer(hidden, src_mask=future, src_key_padding_mask=padding)
        return self.output_norm(hidden)

    def value(self, tokens: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        hidden = self(tokens)
        return self.value_head(
            hidden[torch.arange(len(tokens), device=tokens.device), anchors]
        ).squeeze(-1)


def prediction_masks(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = tokens[:, 1:]
    opponent = (target[..., 0] == BET) & (target[..., 1] == 1)
    chance = ((target[..., 0] == PRIVATE) | (target[..., 0] == PUBLIC)) & (
        target[..., 2] > 0
    )
    return opponent, chance


def sequence_loss(
    model: EventTransformer, tokens: torch.Tensor, legal: torch.Tensor, method: str
) -> torch.Tensor:
    hidden = model(tokens[:, :-1])
    opponent, chance = prediction_masks(tokens)
    action_logits = model.action_head(hidden[opponent])
    action_logits = action_logits.masked_fill(~legal[:, 1:][opponent], -1e9)
    action_targets = tokens[:, 1:, 3][opponent] - 1
    loss = F.cross_entropy(action_logits, action_targets, reduction="sum")
    count = opponent.sum()
    if method == "all" and chance.any():
        loss = loss + F.cross_entropy(
            model.chance_head(hidden[chance]),
            tokens[:, 1:, 2][chance] - 1,
            reduction="sum",
        )
        count = count + chance.sum()
    return loss / count.clamp_min(1)


def batches(data: dict, rows: np.ndarray, device: str, prefix: bool = False):
    width = int(
        (data["anchor"][rows] + 1).max() if prefix else data["lengths"][rows].max()
    )
    tokens = torch.as_tensor(
        data["tokens"][rows, :width], dtype=torch.long, device=device
    )
    if prefix:
        after = (
            torch.arange(width, device=device)[None, :]
            > torch.as_tensor(data["anchor"][rows], device=device)[:, None]
        )
        tokens[after] = 0
    return tokens, torch.as_tensor(data["legal"][rows, :width], device=device)


def action_metrics(
    model: EventTransformer, data: dict, rows: np.ndarray, device: str
) -> dict:
    model.eval()
    nll = correct = uniform_nll = count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            tokens, legal = batches(data, rows[start : start + 128], device)
            opponent, _ = prediction_masks(tokens)
            hidden = model(tokens[:, :-1])
            mask = legal[:, 1:][opponent]
            logits = model.action_head(hidden[opponent]).masked_fill(~mask, -1e9)
            target = tokens[:, 1:, 3][opponent] - 1
            nll += F.cross_entropy(logits, target, reduction="sum").item()
            correct += (logits.argmax(-1) == target).sum().item()
            uniform_nll += mask.sum(-1).float().log().sum().item()
            count += len(target)
    return {
        "nll": nll / count,
        "accuracy": correct / count,
        "uniform_legal_nll": uniform_nll / count,
        "opponent_actions": count,
    }


def reward_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = prediction - truth
    denominator = float(np.square(truth - truth.mean()).sum())
    return {
        "rmse_chips": float(np.sqrt(np.square(error).mean())),
        "mae_chips": float(np.abs(error).mean()),
        "r2": 1 - float(np.square(error).sum()) / denominator if denominator else None,
        "bias_chips": float(error.mean()),
        "samples": len(truth),
    }


def ridge_prediction(
    features: np.ndarray, target: np.ndarray, split: np.ndarray
) -> np.ndarray:
    train, valid = split == 0, split == 1
    mean, std = features[train].mean(0), features[train].std(0).clip(1e-4)
    x = np.column_stack(((features - mean) / std, np.ones(len(features)))).astype(
        np.float64
    )
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0
    cross = x[train].T @ x[train]
    rhs = x[train].T @ target[train]
    best, prediction = math.inf, None
    for strength in (0.01, 0.1, 1, 10, 100):
        weights = np.linalg.solve(cross + strength * penalty, rhs)
        candidate = x @ weights
        loss = np.square(candidate[valid] - target[valid]).mean()
        if loss < best:
            best, prediction = loss, candidate
    return prediction


def extract_features(model: EventTransformer, data: dict, device: str) -> np.ndarray:
    model.eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(data["tokens"]), 128):
            rows = np.arange(start, min(start + 128, len(data["tokens"])))
            tokens, _ = batches(data, rows, device, prefix=True)
            hidden = model(tokens)
            features.append(
                hidden[
                    torch.arange(len(rows), device=device),
                    torch.as_tensor(
                        data["anchor"][rows], dtype=torch.long, device=device
                    ),
                ]
                .cpu()
                .numpy()
            )
    return np.concatenate(features)


def predict_value(
    model: EventTransformer,
    data: dict,
    rows: np.ndarray,
    device: str,
    mean: float,
    scale: float,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            chosen = rows[start : start + 128]
            tokens, _ = batches(data, chosen, device, prefix=True)
            anchors = torch.as_tensor(
                data["anchor"][chosen], dtype=torch.long, device=device
            )
            output.append(model.value(tokens, anchors).cpu().numpy() * scale + mean)
    return np.concatenate(output)


def save_model(path: Path, model: EventTransformer, **metadata) -> None:
    torch.save(
        {"dim": model.dim, "weights": model.state_dict(), "kinds": KINDS, **metadata},
        path,
    )


def load_model(path: Path, device: str = "cpu") -> tuple[EventTransformer, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    model = EventTransformer(saved["dim"]).to(device)
    model.load_state_dict(saved["weights"])
    return model.eval(), saved


def train(args: argparse.Namespace) -> None:
    directory = args.out_dir / f"{args.method}_seed{args.seed}"
    directory.mkdir(exist_ok=False)
    with np.load(args.out_dir / "events.npz", allow_pickle=False) as archive:
        data = dict(archive)
    train_rows, valid_rows, test_rows = (
        np.flatnonzero(data["split"] == i) for i in range(3)
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EventTransformer(args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    curve = []
    best, best_weights, best_step = math.inf, None, 0
    with MemoryMeter() as pre_memory:
        for step in range(args.pretrain_steps + 1 if args.method != "none" else 1):
            if step:
                model.train()
                rows = rng.choice(train_rows, size=args.batch)
                tokens, legal = batches(data, rows, device)
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(model, tokens, legal, args.method)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            if step % 200 == 0 or step == args.pretrain_steps or args.method == "none":
                metrics = action_metrics(model, data, valid_rows, device)
                item = {"stage": "pretrain", "step": step, **metrics}
                curve.append(item)
                print(f"{args.method} seed={args.seed} {item}", flush=True)
                if metrics["nll"] < best:
                    best, best_step = metrics["nll"], step
                    best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    save_model(
        directory / "pretrained.pt",
        model,
        method=args.method,
        seed=args.seed,
        selected_step=best_step,
    )
    pre_actions = {
        name: action_metrics(
            model, data, test_rows[data["cohort"][test_rows] == cohort], device
        )
        for cohort, name in enumerate(COHORTS)
    }
    pre_actions["overall"] = action_metrics(model, data, test_rows, device)
    features = extract_features(model, data, device)
    frozen = ridge_prediction(features, data["reward"], data["split"])
    raw = ridge_prediction(data["raw"], data["reward"], data["split"])
    mean = float(data["reward"][train_rows].mean())
    scale = max(float(data["reward"][train_rows].std()), 1)
    constant = np.full(len(test_rows), mean)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    rng = np.random.default_rng(args.seed + 1000)
    best, value_step, value_weights = math.inf, 0, None
    with MemoryMeter() as value_memory:
        for step in range(args.value_steps + 1):
            if step:
                model.train()
                rows = rng.choice(train_rows, size=args.batch)
                tokens, _ = batches(data, rows, device, prefix=True)
                anchors = torch.as_tensor(
                    data["anchor"][rows], dtype=torch.long, device=device
                )
                target = torch.as_tensor(
                    (data["reward"][rows] - mean) / scale, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model.value(tokens, anchors), target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            if step % 200 == 0 or step == args.value_steps:
                prediction = predict_value(model, data, valid_rows, device, mean, scale)
                metrics = reward_metrics(prediction, data["reward"][valid_rows])
                curve.append({"stage": "value", "step": step, **metrics})
                print(
                    f"{args.method} seed={args.seed} value step={step}: {metrics}",
                    flush=True,
                )
                if metrics["rmse_chips"] < best:
                    best, value_step = metrics["rmse_chips"], step
                    value_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(value_weights)
    save_model(
        directory / "value.pt",
        model,
        method=args.method,
        seed=args.seed,
        mean=mean,
        scale=scale,
        selected_step=value_step,
    )
    prediction = predict_value(model, data, test_rows, device, mean, scale)
    restored, saved = load_model(directory / "value.pt", device)
    again = predict_value(
        restored, data, test_rows[:128], device, saved["mean"], saved["scale"]
    )
    np.testing.assert_allclose(again, prediction[: len(again)], rtol=1e-5, atol=1e-4)
    forecasts = {
        "value": prediction,
        "frozen": frozen[test_rows],
        "raw": raw[test_rows],
        "constant": constant,
    }
    metrics = {}
    for name, predictions in forecasts.items():
        metrics[name] = {
            "overall": reward_metrics(predictions, data["reward"][test_rows])
        }
        for cohort, label in enumerate(COHORTS):
            mask = data["cohort"][test_rows] == cohort
            metrics[name][label] = reward_metrics(
                predictions[mask], data["reward"][test_rows][mask]
            )
    np.savez_compressed(
        directory / "predictions.npz",
        **forecasts,
        truth=data["reward"][test_rows],
        hand=data["hand"][test_rows],
        cohort=data["cohort"][test_rows],
        anchor=data["anchor"][test_rows],
    )
    write_json(
        directory / "metrics.json",
        {
            "method": args.method,
            "seed": args.seed,
            "dim": args.dim,
            "pretrain_steps": args.pretrain_steps if args.method != "none" else 0,
            "value_steps": args.value_steps,
            "selected_pretrain_step": best_step,
            "selected_value_step": value_step,
            "batch": args.batch,
            "device": device,
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "pretrain_memory": pre_memory.result(),
            "value_memory": value_memory.result(),
            "action_test": pre_actions,
            "reward_test": metrics,
            "curve": curve,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_verified": True,
        },
    )
    print(f"completed {directory}: {metrics['value']['overall']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--method", choices=("none", "players", "all"), default="players"
    )
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--value-steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.hands < 60
        or min(
            args.pretrain_steps, args.value_steps, args.batch, args.dim, args.threads
        )
        < 1
        or args.dim % 4
    ):
        parser.error("Need hands>=60, positive budgets, and dim divisible by four")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    {"collect": collect, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
