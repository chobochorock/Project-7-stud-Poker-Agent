from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from torch import nn
from torch.nn import functional as F

from ev_rollout import ACTIONS, SCHEMA_VERSION
from poker_env import EV_RAISE_CAP


ACTION_NAMES = tuple(ACTIONS)
ACTION_COUNT = len(ACTION_NAMES)
CARD_SLOTS = 12
MAX_HISTORY = 36
PAD = 255
RAW_FEATURES = (
    CARD_SLOTS * 18
    + MAX_HISTORY * (6 + ACTION_COUNT)
    + 4
    + 2
    + ACTION_COUNT
    + 7
)


@dataclass
class UCTDataset:
    cards: np.ndarray
    history: np.ndarray
    scalars: np.ndarray
    street: np.ndarray
    seat: np.ndarray
    legal_mask: np.ndarray
    policy: np.ndarray
    q: np.ndarray
    visits: np.ndarray
    validation: np.ndarray

    def __len__(self) -> int:
        return len(self.street)


class RawMLP(nn.Module):
    def __init__(self, hidden: int = 256, latent: int = 32):
        super().__init__()
        self.hidden = hidden
        self.latent_size = latent
        self.encoder = nn.Sequential(
            nn.Linear(RAW_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, latent),
        )
        self.policy_head = nn.Linear(latent, ACTION_COUNT)
        self.q_head = nn.Linear(latent, ACTION_COUNT)

    def forward(
        self,
        cards: torch.Tensor,
        history: torch.Tensor,
        scalars: torch.Tensor,
        street: torch.Tensor,
        seat: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = raw_features(cards, history, scalars, street, seat, legal_mask)
        latent = self.encoder(features)
        return latent, self.policy_head(latent), self.q_head(latent)


def raw_features(
    cards: torch.Tensor,
    history: torch.Tensor,
    scalars: torch.Tensor,
    street: torch.Tensor,
    seat: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    cards = cards.long()
    card_present = cards.ne(PAD)
    safe_cards = cards.clamp(0, 51)
    card_rank = F.one_hot(safe_cards // 4, 13)
    card_suit = F.one_hot(safe_cards % 4, 4)
    card_features = torch.cat(
        (card_rank, card_suit, card_present.unsqueeze(-1).long()), dim=-1
    ).float()
    card_features *= card_present.unsqueeze(-1)

    history = history.long()
    history_present = history[..., 0].ne(PAD)
    safe_history = history.clamp_min(0)
    history_features = torch.cat(
        (
            F.one_hot(safe_history[..., 0].clamp_max(3), 4),
            F.one_hot(safe_history[..., 1].clamp_max(1), 2),
            F.one_hot(safe_history[..., 2].clamp_max(ACTION_COUNT - 1), ACTION_COUNT),
        ),
        dim=-1,
    ).float()
    history_features *= history_present.unsqueeze(-1)

    legal = legal_actions(legal_mask).float()
    return torch.cat(
        (
            card_features.flatten(1),
            history_features.flatten(1),
            F.one_hot(street.long(), 4).float(),
            F.one_hot(seat.long(), 2).float(),
            legal,
            scalars.float(),
        ),
        dim=1,
    )


def legal_actions(mask: torch.Tensor) -> torch.Tensor:
    bits = 1 << torch.arange(ACTION_COUNT, device=mask.device)
    return mask.long().unsqueeze(1).bitwise_and(bits).ne(0)


def load_uct_dataset(
    path: Path | Sequence[Path],
    *,
    search_version: str,
    opponent_policy: str,
    simulation_budget: int,
    max_rows: int = 0,
    min_node_simulations: int = 0,
    validation_fraction: float = 0.1,
    q_normalization: str = "ante",
) -> UCTDataset:
    paths = (path,) if isinstance(path, Path) else tuple(path)
    if not paths:
        raise ValueError("At least one input database is required.")
    for input_path in paths:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if q_normalization not in {"ante", "pot"}:
        raise ValueError("q_normalization must be ante or pot.")
    if min_node_simulations < 0:
        raise ValueError("min_node_simulations must be non-negative.")

    action_columns = ", ".join(
        column
        for action in ACTION_NAMES
        for column in (f"{action.lower()}_visits", f"{action.lower()}_return_sum")
    )
    where = "search_version = ? AND opponent_policy = ? AND simulation_budget = ?"
    parameters = (search_version, opponent_policy, simulation_budget)
    if min_node_simulations:
        where += " AND simulations >= ?"
        parameters += (min_node_simulations,)
    available = 0
    for input_path in paths:
        uri = f"file:{input_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            available += int(
                connection.execute(
                    f"SELECT COUNT(*) FROM uct_nodes WHERE {where}", parameters
                ).fetchone()[0]
            )
    rows = min(available, max_rows) if max_rows > 0 else available
    if rows == 0:
        raise ValueError("No matching UCT rows found.")

    cards = np.full((rows, CARD_SLOTS), PAD, dtype=np.uint8)
    history = np.full((rows, MAX_HISTORY, 3), PAD, dtype=np.uint8)
    scalars = np.empty((rows, 7), dtype=np.float32)
    street = np.empty(rows, dtype=np.uint8)
    seat = np.empty(rows, dtype=np.uint8)
    legal_mask = np.empty(rows, dtype=np.uint8)
    policy = np.empty((rows, ACTION_COUNT), dtype=np.float32)
    q = np.zeros((rows, ACTION_COUNT), dtype=np.float32)
    visits = np.empty((rows, ACTION_COUNT), dtype=np.uint32)
    validation = np.empty(rows, dtype=np.bool_)

    sql = (
        f"SELECT state_json, seat_index, legal_mask, {action_columns} "
        f"FROM uct_nodes WHERE {where}"
    )
    started = time.perf_counter()
    index = 0
    for input_path in paths:
        remaining = rows - index
        if remaining <= 0:
            break
        uri = f"file:{input_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            for row in connection.execute(f"{sql} LIMIT ?", parameters + (remaining,)):
                state_json = str(row[0])
                payload = json.loads(state_json)
                if len(payload) != 12 or payload[0] != SCHEMA_VERSION:
                    raise ValueError("Unsupported state schema in UCT database.")

                street[index] = int(payload[1])
                seat[index] = int(row[1])
                legal_mask[index] = int(row[2])
                cards[index] = _card_slots(payload[2])

                events = payload[11]
                if len(events) > MAX_HISTORY:
                    raise ValueError(
                        f"History length {len(events)} exceeds MAX_HISTORY={MAX_HISTORY}."
                    )
                if events:
                    history[index, : len(events)] = np.asarray(events, dtype=np.uint8)

                ante = float(payload[3])
                if ante <= 0:
                    raise ValueError("Ante must be positive for normalization.")
                scalars[index] = np.asarray(
                    [
                        math.log1p(float(payload[4]) / ante),
                        math.log1p(float(payload[5]) / ante),
                        math.log1p(float(payload[6]) / ante),
                        math.log1p(float(payload[7]) / ante),
                        math.log1p(float(payload[8]) / ante),
                        math.log1p(float(payload[9]) / ante),
                        float(payload[10]) / EV_RAISE_CAP,
                    ],
                    dtype=np.float32,
                )

                action_visits = np.asarray(row[3::2], dtype=np.uint32)
                return_sums = np.asarray(row[4::2], dtype=np.float64)
                total_visits = int(action_visits.sum())
                if total_visits <= 0:
                    raise ValueError("UCT row has no action visits.")
                visits[index] = action_visits
                policy[index] = action_visits / total_visits
                visited = action_visits > 0
                q_scale = max(ante, float(payload[4])) if q_normalization == "pot" else ante
                q[index, visited] = (
                    return_sums[visited] / action_visits[visited] / q_scale
                ).astype(np.float32)

                digest = hashlib.blake2b(
                    f"{state_json}|{row[1]}".encode("utf-8"), digest_size=8
                ).digest()
                validation[index] = (
                    int.from_bytes(digest, "little") / 2**64 < validation_fraction
                )
                if (index + 1) % 100_000 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"loaded {index + 1:,}/{rows:,} rows "
                        f"({(index + 1) / elapsed:,.0f}/s)"
                    )
                index += 1

    train_rows = int((~validation).sum())
    validation_rows = int(validation.sum())
    if train_rows == 0 or validation_rows == 0:
        raise ValueError("Stable split produced an empty train or validation set.")
    print(f"dataset: {rows:,} rows ({train_rows:,} train, {validation_rows:,} validation)")
    return UCTDataset(
        cards,
        history,
        scalars,
        street,
        seat,
        legal_mask,
        policy,
        q,
        visits,
        validation,
    )


def _card_slots(groups: Sequence[object]) -> np.ndarray:
    if len(groups) != 4:
        raise ValueError("Expected four card groups.")
    hidden = sorted(int(card) for card in groups[0])
    own_public = [int(card) for card in groups[1]]
    discarded = [] if int(groups[2]) < 0 else [int(groups[2])]
    opponent_public = [int(card) for card in groups[3]]
    specifications = ((hidden, 3), (own_public, 4), (discarded, 1), (opponent_public, 4))
    output = np.full(CARD_SLOTS, PAD, dtype=np.uint8)
    offset = 0
    for values, size in specifications:
        if len(values) > size or any(not 0 <= card < 52 for card in values):
            raise ValueError("Invalid card group in UCT state.")
        output[offset : offset + len(values)] = values
        offset += size
    return output


def _batch(data: UCTDataset, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "cards": torch.as_tensor(data.cards[indices], device=device),
        "history": torch.as_tensor(data.history[indices], device=device),
        "scalars": torch.as_tensor(data.scalars[indices], device=device),
        "street": torch.as_tensor(data.street[indices], device=device),
        "seat": torch.as_tensor(data.seat[indices], device=device),
        "legal_mask": torch.as_tensor(data.legal_mask[indices], device=device),
        "policy": torch.as_tensor(data.policy[indices], device=device),
        "q": torch.as_tensor(data.q[indices], device=device),
        "visits": torch.as_tensor(data.visits[indices], device=device),
    }


def _model_outputs(
    model: RawMLP, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(
        batch["cards"],
        batch["history"],
        batch["scalars"],
        batch["street"],
        batch["seat"],
        batch["legal_mask"],
    )


def train_raw_mlp(
    data: UCTDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    policy_loss_weight: float,
    seed: int,
) -> tuple[RawMLP, dict[str, float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RawMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(~data.validation)
    validation_indices = np.flatnonzero(data.validation)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        loss_sum = 0.0
        samples = 0
        started = time.perf_counter()
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            batch = _batch(data, indices, device)
            _, logits, predicted_q = _model_outputs(model, batch)
            legal = legal_actions(batch["legal_mask"])
            masked_logits = logits.masked_fill(~legal, -1e9)
            policy_loss = -(batch["policy"] * F.log_softmax(masked_logits, dim=1)).sum(1).mean()

            q_weights = batch["visits"].float().sqrt()
            q_error = F.smooth_l1_loss(predicted_q, batch["q"], reduction="none")
            q_loss = (q_error * q_weights).sum() / q_weights.sum().clamp_min(1)
            loss = policy_loss_weight * policy_loss + q_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            samples += len(indices)

        metrics = evaluate_raw_mlp(model, data, validation_indices, device, batch_size)
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch}/{epochs}: loss={loss_sum / samples:.5f}, "
            f"val_regret={metrics['regret']:.5f}, "
            f"visit_agreement={metrics['visit_agreement']:.3f}, {elapsed:.1f}s"
        )
        if best_metrics is None or metrics["regret"] < best_metrics["regret"]:
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics


@torch.no_grad()
def evaluate_raw_mlp(
    model: RawMLP,
    data: UCTDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    totals = _empty_metrics()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = _batch(data, selected, device)
        _, logits, predicted_q = _model_outputs(model, batch)
        legal = legal_actions(batch["legal_mask"])
        predicted_policy = F.softmax(logits.masked_fill(~legal, -1e9), dim=1)
        _add_metrics(
            totals,
            predicted_q.cpu().numpy(),
            predicted_policy.cpu().numpy(),
            data,
            selected,
        )
    return _finish_metrics(totals)


@torch.no_grad()
def extract_latent(
    model: RawMLP,
    data: UCTDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    latent = np.empty((len(data), model.latent_size), dtype=np.float32)
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        batch = _batch(data, indices, device)
        values, _, _ = _model_outputs(model, batch)
        latent[indices] = values.cpu().numpy()
    return latent


def fit_kmeans(
    latent: np.ndarray,
    data: UCTDataset,
    train_indices: np.ndarray,
    fit_indices: np.ndarray,
    *,
    clusters: int,
    top_k: int,
    temperature: float,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    fit_latent = _normalize(latent[fit_indices])
    model = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=min(max(batch_size, clusters * 3), len(fit_indices)),
        n_init=3,
        max_iter=100,
        random_state=seed,
    ).fit(fit_latent)
    centers = _normalize(model.cluster_centers_.astype(np.float32))

    def hard_responsibility(values: np.ndarray) -> np.ndarray:
        assignments = np.argmax(_normalize(values) @ centers.T, axis=1)
        result = np.zeros((len(values), clusters), dtype=np.float32)
        result[np.arange(len(values)), assignments] = 1
        return result

    component_q, component_policy, support = _component_stats(
        latent, data, train_indices, hard_responsibility, clusters, batch_size
    )

    def soft_responsibility(values: np.ndarray) -> np.ndarray:
        scores = _normalize(values) @ centers.T / temperature
        return _top_k_softmax(scores, top_k)

    validation_indices = np.flatnonzero(data.validation)
    metrics = {
        "hard": evaluate_cluster(
            latent,
            data,
            validation_indices,
            hard_responsibility,
            component_q,
            component_policy,
            batch_size,
        ),
        "soft": evaluate_cluster(
            latent,
            data,
            validation_indices,
            soft_responsibility,
            component_q,
            component_policy,
            batch_size,
        ),
    }
    artifact = {
        "centers": centers,
        "component_q": component_q,
        "component_policy": component_policy,
        "support": support,
        "temperature": np.asarray(temperature, dtype=np.float32),
        "top_k": np.asarray(top_k, dtype=np.int32),
    }
    return artifact, metrics


def _gmm_log_joint(
    values: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> torch.Tensor:
    precision = variances.reciprocal()
    quadratic = (
        values.square() @ precision.T
        - 2.0 * values @ (means * precision).T
        + (means.square() * precision).sum(dim=1)
    )
    normalizer = values.shape[1] * math.log(2.0 * math.pi) + variances.log().sum(dim=1)
    return weights.log()[None, :] - 0.5 * (quadratic + normalizer[None, :])


@torch.no_grad()
def _fit_diagonal_gmm(
    values: np.ndarray,
    *,
    clusters: int,
    device: torch.device,
    batch_size: int,
    max_iterations: int,
    seed: int,
    tolerance: float = 1e-3,
    regularization: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    initializer = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=min(max(batch_size, clusters * 3), len(values)),
        n_init=1,
        max_iter=100,
        random_state=seed,
    ).fit(values)
    counts = np.bincount(initializer.labels_, minlength=clusters).astype(np.float32)
    weights = torch.as_tensor(np.maximum(counts, 1.0), device=device)
    weights /= weights.sum()
    means = torch.as_tensor(initializer.cluster_centers_, device=device)
    variances = torch.ones_like(means)

    previous_bound = -math.inf
    lower_bound = -math.inf
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        mass = torch.zeros(clusters, device=device)
        value_sum = torch.zeros_like(means)
        square_sum = torch.zeros_like(means)
        log_likelihood = 0.0

        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], device=device)
            log_joint = _gmm_log_joint(batch, weights, means, variances)
            log_normalizer = torch.logsumexp(log_joint, dim=1)
            responsibility = torch.exp(log_joint - log_normalizer[:, None])
            mass += responsibility.sum(dim=0)
            value_sum += responsibility.T @ batch
            square_sum += responsibility.T @ batch.square()
            log_likelihood += float(log_normalizer.sum().item())

        safe_mass = mass.clamp_min(1e-8)
        means = value_sum / safe_mass[:, None]
        variances = (square_sum / safe_mass[:, None] - means.square()).clamp_min(
            regularization
        )
        weights = safe_mass / safe_mass.sum()
        lower_bound = log_likelihood / len(values)
        change = lower_bound - previous_bound
        print(
            f"EM {iteration}/{max_iterations}: lower_bound={lower_bound:.6f}, "
            f"change={change:.6f}"
        )
        iterations = iteration
        if iteration > 1 and abs(change) < tolerance:
            converged = True
            break
        previous_bound = lower_bound

    return (
        weights.cpu().numpy(),
        means.cpu().numpy(),
        variances.cpu().numpy(),
        converged,
        iterations,
        lower_bound,
    )


def fit_gmm(
    latent: np.ndarray,
    data: UCTDataset,
    train_indices: np.ndarray,
    fit_indices: np.ndarray,
    *,
    clusters: int,
    top_k: int,
    batch_size: int,
    em_iterations: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | bool | str]]:
    fit_latent = latent[fit_indices].astype(np.float64)
    shrinkage = LedoitWolf().fit(fit_latent)
    eigenvalues, eigenvectors = np.linalg.eigh(shrinkage.covariance_)
    whitening = eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8)))[None, :]
    whitening_mean = shrinkage.location_
    whitened_fit = ((fit_latent - whitening_mean) @ whitening).astype(np.float32)
    weights, means, variances, converged, iterations, lower_bound = (
        _fit_diagonal_gmm(
            whitened_fit,
            clusters=clusters,
            device=device,
            batch_size=max(batch_size, 32_768) if device.type == "cuda" else batch_size,
            max_iterations=em_iterations,
            seed=seed,
        )
    )
    whitening_mean_tensor = torch.as_tensor(
        whitening_mean.astype(np.float32), device=device
    )
    whitening_tensor = torch.as_tensor(whitening.astype(np.float32), device=device)
    weights_tensor = torch.as_tensor(weights, device=device)
    means_tensor = torch.as_tensor(means, device=device)
    variances_tensor = torch.as_tensor(variances, device=device)

    def responsibility(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value_tensor = torch.as_tensor(values, device=device)
            whitened = (value_tensor - whitening_mean_tensor) @ whitening_tensor
            log_joint = _gmm_log_joint(
                whitened, weights_tensor, means_tensor, variances_tensor
            )
            count = min(top_k, clusters)
            selected_log, selected = torch.topk(log_joint, count, dim=1)
            selected_weight = torch.softmax(selected_log, dim=1)
            result = torch.zeros_like(log_joint)
            result.scatter_(1, selected, selected_weight)
            return result.cpu().numpy()

    component_q, component_policy, support = _component_stats(
        latent, data, train_indices, responsibility, clusters, batch_size
    )
    validation_indices = np.flatnonzero(data.validation)
    metrics: dict[str, float | int | bool | str] = evaluate_cluster(
        latent,
        data,
        validation_indices,
        responsibility,
        component_q,
        component_policy,
        batch_size,
    )
    metrics.update(
        {
            "backend": f"torch-{device.type}",
            "converged": converged,
            "iterations": iterations,
            "lower_bound": lower_bound,
        }
    )
    artifact = {
        "whitening_mean": whitening_mean.astype(np.float32),
        "whitening_transform": whitening.astype(np.float32),
        "weights": weights.astype(np.float32),
        "means": means.astype(np.float32),
        "variances": variances.astype(np.float32),
        "component_q": component_q,
        "component_policy": component_policy,
        "support": support,
        "top_k": np.asarray(top_k, dtype=np.int32),
    }
    return artifact, metrics


def _component_stats(
    latent: np.ndarray,
    data: UCTDataset,
    indices: np.ndarray,
    responsibility: Callable[[np.ndarray], np.ndarray],
    clusters: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.zeros(clusters, dtype=np.float64)
    policy_sum = np.zeros((clusters, ACTION_COUNT), dtype=np.float64)
    q_sum = np.zeros((clusters, ACTION_COUNT), dtype=np.float64)
    q_weight = np.zeros((clusters, ACTION_COUNT), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        weights = responsibility(latent[selected]).astype(np.float64)
        action_weight = np.sqrt(data.visits[selected].astype(np.float64))
        support += weights.sum(axis=0)
        policy_sum += weights.T @ data.policy[selected]
        q_sum += weights.T @ (data.q[selected] * action_weight)
        q_weight += weights.T @ action_weight

    global_policy = data.policy[indices].mean(axis=0)
    global_q_weight = np.sqrt(data.visits[indices].astype(np.float64)).sum(axis=0)
    global_q = (
        data.q[indices] * np.sqrt(data.visits[indices].astype(np.float64))
    ).sum(axis=0) / np.maximum(global_q_weight, 1e-12)
    component_policy = np.divide(
        policy_sum,
        support[:, None],
        out=np.tile(global_policy, (clusters, 1)),
        where=support[:, None] > 1e-12,
    )
    component_q = np.divide(
        q_sum,
        q_weight,
        out=np.tile(global_q, (clusters, 1)),
        where=q_weight > 1e-12,
    )
    return (
        component_q.astype(np.float32),
        component_policy.astype(np.float32),
        support.astype(np.float32),
    )


def evaluate_cluster(
    latent: np.ndarray,
    data: UCTDataset,
    indices: np.ndarray,
    responsibility: Callable[[np.ndarray], np.ndarray],
    component_q: np.ndarray,
    component_policy: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    totals = _empty_metrics()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        weights = responsibility(latent[selected])
        _add_metrics(
            totals,
            weights @ component_q,
            weights @ component_policy,
            data,
            selected,
        )
    return _finish_metrics(totals)


def _empty_metrics() -> dict[str, float]:
    return {
        "rows": 0.0,
        "q_error": 0.0,
        "q_values": 0.0,
        "policy_ce": 0.0,
        "q_agreement": 0.0,
        "visit_agreement": 0.0,
        "regret": 0.0,
    }


def _add_metrics(
    totals: dict[str, float],
    predicted_q: np.ndarray,
    predicted_policy: np.ndarray,
    data: UCTDataset,
    indices: np.ndarray,
) -> None:
    legal = ((data.legal_mask[indices, None] >> np.arange(ACTION_COUNT)) & 1).astype(bool)
    visited = data.visits[indices] > 0
    target_policy = data.policy[indices]
    target_q = data.q[indices]

    masked_policy = np.where(legal, np.maximum(predicted_policy, 0), 0)
    normalizer = masked_policy.sum(axis=1, keepdims=True)
    masked_policy = np.divide(
        masked_policy,
        normalizer,
        out=legal / np.maximum(legal.sum(axis=1, keepdims=True), 1),
        where=normalizer > 1e-12,
    )
    policy_ce = -(target_policy * np.log(np.maximum(masked_policy, 1e-12))).sum(axis=1)

    comparable = legal & visited
    predicted_action = np.argmax(np.where(legal, predicted_q, -np.inf), axis=1)
    target_action = np.argmax(np.where(comparable, target_q, -np.inf), axis=1)
    visit_action = np.argmax(target_policy, axis=1)
    rows = np.arange(len(indices))
    target_best = target_q[rows, target_action]
    chosen_target = target_q[rows, predicted_action]

    totals["rows"] += len(indices)
    totals["q_error"] += np.abs(predicted_q - target_q)[visited].sum()
    totals["q_values"] += visited.sum()
    totals["policy_ce"] += policy_ce.sum()
    totals["q_agreement"] += (predicted_action == target_action).sum()
    totals["visit_agreement"] += (predicted_action == visit_action).sum()
    totals["regret"] += np.maximum(0, target_best - chosen_target).sum()


def _finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    rows = max(totals["rows"], 1)
    return {
        "q_mae": totals["q_error"] / max(totals["q_values"], 1),
        "policy_cross_entropy": totals["policy_ce"] / rows,
        "q_best_agreement": totals["q_agreement"] / rows,
        "visit_agreement": totals["visit_agreement"] / rows,
        "regret": totals["regret"] / rows,
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _top_k_softmax(scores: np.ndarray, top_k: int) -> np.ndarray:
    count = min(top_k, scores.shape[1])
    selected = np.argpartition(scores, -count, axis=1)[:, -count:]
    selected_scores = np.take_along_axis(scores, selected, axis=1)
    selected_scores -= selected_scores.max(axis=1, keepdims=True)
    weights = np.exp(selected_scores)
    weights /= weights.sum(axis=1, keepdims=True)
    result = np.zeros_like(scores, dtype=np.float32)
    np.put_along_axis(result, selected, weights.astype(np.float32), axis=1)
    return result


def _top_k_probabilities(probabilities: np.ndarray, top_k: int) -> np.ndarray:
    count = min(top_k, probabilities.shape[1])
    selected = np.argpartition(probabilities, -count, axis=1)[:, -count:]
    weights = np.take_along_axis(probabilities, selected, axis=1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
    result = np.zeros_like(probabilities, dtype=np.float32)
    np.put_along_axis(result, selected, weights.astype(np.float32), axis=1)
    return result


def _fit_indices(
    train_indices: np.ndarray, cluster_max_rows: int, seed: int
) -> np.ndarray:
    if cluster_max_rows <= 0 or len(train_indices) <= cluster_max_rows:
        return train_indices
    return np.random.default_rng(seed).choice(
        train_indices, size=cluster_max_rows, replace=False
    )


def _save_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_training(
    input_path: Path | Sequence[Path],
    output_dir: Path,
    *,
    search_version: str = "uct-v2",
    opponent_policy: str = "random",
    simulation_budget: int = 512,
    max_rows: int = 0,
    min_node_simulations: int = 0,
    cluster_max_rows: int = 100_000,
    validation_fraction: float = 0.1,
    epochs: int = 8,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    policy_loss_weight: float = 1.0,
    q_normalization: str = "ante",
    clusters: int = 256,
    top_k: int = 4,
    temperature: float = 0.1,
    em_iterations: int = 30,
    seed: int = 7,
    device_name: str = "auto",
) -> dict[str, object]:
    if clusters <= 0 or top_k <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("clusters, top_k, epochs, and batch_size must be positive.")
    if temperature <= 0 or em_iterations <= 0:
        raise ValueError("temperature and em_iterations must be positive.")
    if policy_loss_weight < 0:
        raise ValueError("policy_loss_weight must be non-negative.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    print(f"device: {device}")
    started = time.perf_counter()
    data = load_uct_dataset(
        input_path,
        search_version=search_version,
        opponent_policy=opponent_policy,
        simulation_budget=simulation_budget,
        max_rows=max_rows,
        min_node_simulations=min_node_simulations,
        validation_fraction=validation_fraction,
        q_normalization=q_normalization,
    )
    train_indices = np.flatnonzero(~data.validation)
    fit_indices = _fit_indices(train_indices, cluster_max_rows, seed)
    if len(fit_indices) < clusters:
        raise ValueError("Clustering fit rows must be at least the component count.")

    model, raw_metrics = train_raw_mlp(
        data,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        policy_loss_weight=policy_loss_weight,
        seed=seed,
    )
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "raw_features": RAW_FEATURES,
            "latent_size": model.latent_size,
            "action_names": ACTION_NAMES,
            "search_version": search_version,
            "opponent_policy": opponent_policy,
            "simulation_budget": simulation_budget,
            "policy_loss_weight": policy_loss_weight,
            "q_normalization": q_normalization,
        },
        output_dir / "raw_mlp.pt",
    )

    latent = extract_latent(model, data, device, batch_size)
    kmeans_artifact, kmeans_metrics = fit_kmeans(
        latent,
        data,
        train_indices,
        fit_indices,
        clusters=clusters,
        top_k=top_k,
        temperature=temperature,
        batch_size=batch_size,
        seed=seed,
    )
    np.savez_compressed(output_dir / "spherical_kmeans.npz", **kmeans_artifact)

    metrics: dict[str, object] = {
        "config": {
            "input": (
                str(input_path.resolve())
                if isinstance(input_path, Path)
                else [str(path.resolve()) for path in input_path]
            ),
            "search_version": search_version,
            "opponent_policy": opponent_policy,
            "simulation_budget": simulation_budget,
            "rows": len(data),
            "min_node_simulations": min_node_simulations,
            "train_rows": len(train_indices),
            "validation_rows": int(data.validation.sum()),
            "cluster_fit_rows": len(fit_indices),
            "clusters": clusters,
            "top_k": top_k,
            "temperature": temperature,
            "policy_loss_weight": policy_loss_weight,
            "q_normalization": q_normalization,
            "seed": seed,
            "device": str(device),
        },
        "raw_mlp": raw_metrics,
        "spherical_kmeans": kmeans_metrics,
    }
    _save_json(output_dir / "metrics.json", metrics)

    print("fitting diagonal GMM/EM...")
    gmm_artifact, gmm_metrics = fit_gmm(
        latent,
        data,
        train_indices,
        fit_indices,
        clusters=clusters,
        top_k=top_k,
        batch_size=batch_size,
        em_iterations=em_iterations,
        seed=seed,
        device=device,
    )
    np.savez_compressed(output_dir / "diagonal_gmm.npz", **gmm_artifact)
    metrics["diagonal_gmm"] = gmm_metrics
    metrics["elapsed_seconds"] = time.perf_counter() - started
    _save_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train raw MLP, spherical k-means, and diagonal GMM baselines."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("models/clustering_v1"))
    parser.add_argument("--search-version", default="uct-v2")
    parser.add_argument("--opponent-policy", default="random")
    parser.add_argument("--simulation-budget", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0, help="0 uses every matching row.")
    parser.add_argument("--min-node-simulations", type=int, default=0)
    parser.add_argument("--cluster-max-rows", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=1.0,
        help="Set to 0 when action visits are search allocation rather than a policy target.",
    )
    parser.add_argument(
        "--q-normalization",
        choices=("ante", "pot"),
        default="ante",
        help="Normalize action returns by ante or max(ante, current pot).",
    )
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--em-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(
        args.input,
        args.output,
        search_version=args.search_version,
        opponent_policy=args.opponent_policy,
        simulation_budget=args.simulation_budget,
        max_rows=args.max_rows,
        min_node_simulations=args.min_node_simulations,
        cluster_max_rows=args.cluster_max_rows,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_loss_weight=args.policy_loss_weight,
        q_normalization=args.q_normalization,
        clusters=args.clusters,
        top_k=args.top_k,
        temperature=args.temperature,
        em_iterations=args.em_iterations,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
