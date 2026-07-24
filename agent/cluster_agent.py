from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from agent.base import PokerAgent
from clustering_train import (
    ACTION_NAMES,
    CARD_SLOTS,
    MAX_HISTORY,
    PAD,
    RawMLP,
    _card_slots,
)
from ev_rollout import canonical_state
from poker_env import EV_RAISE_CAP


class ClusterPokerAgent(PokerAgent):
    """Play stackless heads-up EV using a saved soft cluster policy."""

    def __init__(
        self,
        name: str,
        model_dir: Path,
        *,
        clusterer: str = "kmeans",
        decision: str = "policy",
        seed: int = 7,
        device_name: str = "auto",
    ):
        super().__init__(name)
        if clusterer not in {"raw", "kmeans", "gmm"}:
            raise ValueError("clusterer must be raw, kmeans or gmm.")
        if decision not in {"policy", "q"}:
            raise ValueError("decision must be policy or q.")
        self.clusterer = clusterer
        self.decision = decision
        self.rng = random.Random(seed)
        self.device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available() else
            "cpu" if device_name == "auto" else device_name
        )

        checkpoint = torch.load(
            model_dir / "raw_mlp.pt", map_location=self.device, weights_only=True
        )
        if decision == "policy" and float(checkpoint.get("policy_loss_weight", 1.0)) == 0:
            raise ValueError("This model was trained without a valid policy target; use decision='q'.")
        hidden = int(checkpoint["state_dict"]["encoder.0.weight"].shape[0])
        self.model = RawMLP(hidden=hidden, latent=int(checkpoint["latent_size"])).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.artifact: dict[str, np.ndarray] = {}
        if clusterer != "raw":
            artifact_name = (
                "spherical_kmeans.npz" if clusterer == "kmeans" else "diagonal_gmm.npz"
            )
            with np.load(model_dir / artifact_name, allow_pickle=False) as artifact:
                self.artifact = {key: artifact[key] for key in artifact.files}
        self.component_q = self.artifact.get("component_q")
        self.component_policy = self.artifact.get("component_policy")
        self.cluster_mass = np.zeros(
            0 if self.component_q is None else len(self.component_q), dtype=np.float64
        )
        self.decisions = 0
        self.top_weight_sum = 0.0
        self.entropy_sum = 0.0
        self.action_counts = {action: 0 for action in ACTION_NAMES}

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def choose_action(
        self, state: dict[str, Any], valid_actions: Sequence[str]
    ) -> str | None:
        valid = tuple(valid_actions)
        if not valid:
            return None
        if state.get("game_mode") != "ev" or state.get("seat_count") != 2:
            raise ValueError("Cluster agent only supports stackless heads-up EV states.")

        latent, logits, raw_q = self._outputs(state, valid)
        legal_indices = np.asarray([ACTION_NAMES.index(action) for action in valid])
        if self.clusterer == "raw":
            if self.decision == "q":
                action = ACTION_NAMES[legal_indices[np.argmax(raw_q[legal_indices])]]
            else:
                legal_logits = logits[legal_indices] - logits[legal_indices].max()
                probabilities = np.exp(legal_logits)
                probabilities /= probabilities.sum()
                action = ACTION_NAMES[
                    self.rng.choices(legal_indices.tolist(), weights=probabilities, k=1)[0]
                ]
        else:
            responsibility = self._responsibility(latent)
            if self.decision == "q":
                scores = responsibility @ self.component_q
                action = ACTION_NAMES[legal_indices[np.argmax(scores[legal_indices])]]
            else:
                probabilities = np.maximum(
                    (responsibility @ self.component_policy)[legal_indices], 0
                )
                if probabilities.sum() <= 1e-12:
                    probabilities = np.ones(len(legal_indices))
                probabilities /= probabilities.sum()
                action = ACTION_NAMES[
                    self.rng.choices(legal_indices.tolist(), weights=probabilities, k=1)[0]
                ]

            positive = responsibility[responsibility > 0]
            self.cluster_mass += responsibility
            self.top_weight_sum += float(responsibility.max())
            self.entropy_sum -= float(np.sum(positive * np.log(positive)))

        self.decisions += 1
        self.action_counts[action] += 1
        return action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) != 4:
            raise ValueError("Cluster discard currently expects four hidden cards.")
        discard = self.rng.randrange(4)
        reveal = self.rng.choice([index for index in range(4) if index != discard])
        return discard, reveal

    def diagnostics(self) -> dict[str, object]:
        if self.decisions == 0:
            return {"decisions": 0}
        if self.clusterer == "raw":
            return {
                "decisions": self.decisions,
                "action_counts": {key: value for key, value in self.action_counts.items() if value},
            }
        usage = self.cluster_mass / self.cluster_mass.sum()
        positive = usage[usage > 0]
        return {
            "decisions": self.decisions,
            "average_top_responsibility": self.top_weight_sum / self.decisions,
            "average_responsibility_entropy": self.entropy_sum / self.decisions,
            "effective_clusters_used": float(np.exp(-np.sum(positive * np.log(positive)))),
            "action_counts": {key: value for key, value in self.action_counts.items() if value},
        }

    @torch.no_grad()
    def _outputs(
        self, state: dict[str, Any], valid_actions: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        payload = json.loads(canonical_state(state))
        cards = _card_slots(payload[2])[None, :]
        history = np.full((1, MAX_HISTORY, 3), PAD, dtype=np.uint8)
        events = payload[11]
        if len(events) > MAX_HISTORY:
            raise ValueError(f"History length {len(events)} exceeds {MAX_HISTORY}.")
        if events:
            history[0, : len(events)] = np.asarray(events, dtype=np.uint8)
        ante = float(payload[3])
        scalars = np.asarray(
            [[
                math.log1p(float(payload[index]) / ante)
                for index in range(4, 10)
            ] + [float(payload[10]) / EV_RAISE_CAP]],
            dtype=np.float32,
        )
        legal_mask = sum(1 << ACTION_NAMES.index(action) for action in valid_actions)
        latent, logits, q = self.model(
            torch.as_tensor(cards, device=self.device),
            torch.as_tensor(history, device=self.device),
            torch.as_tensor(scalars, device=self.device),
            torch.as_tensor([payload[1]], device=self.device),
            torch.as_tensor([state["seat_index"]], device=self.device),
            torch.as_tensor([legal_mask], device=self.device),
        )
        return (
            latent[0].cpu().numpy(),
            logits[0].cpu().numpy(),
            q[0].cpu().numpy(),
        )

    def _responsibility(self, latent: np.ndarray) -> np.ndarray:
        if self.clusterer == "kmeans":
            normalized = latent / max(float(np.linalg.norm(latent)), 1e-12)
            scores = (
                normalized @ self.artifact["centers"].T
                / float(self.artifact["temperature"])
            )
        else:
            whitened = (
                latent - self.artifact["whitening_mean"]
            ) @ self.artifact["whitening_transform"]
            variances = np.maximum(self.artifact["variances"], 1e-12)
            scores = np.log(np.maximum(self.artifact["weights"], 1e-30)) - 0.5 * (
                np.sum(np.log(2.0 * math.pi * variances), axis=1)
                + np.sum((whitened - self.artifact["means"]) ** 2 / variances, axis=1)
            )

        top_k = min(int(self.artifact["top_k"]), len(scores))
        selected = np.argpartition(scores, -top_k)[-top_k:]
        selected_scores = scores[selected] - scores[selected].max()
        selected_weights = np.exp(selected_scores)
        selected_weights /= selected_weights.sum()
        responsibility = np.zeros(len(scores), dtype=np.float32)
        responsibility[selected] = selected_weights
        return responsibility
