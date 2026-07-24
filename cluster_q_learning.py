from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent.cluster_agent import ClusterPokerAgent
from clustering_train import ACTION_NAMES
from poker_env import PokerGame


class ClusterQLearningAgent(ClusterPokerAgent):
    """TD(0) control over a fixed soft cluster representation."""

    def __init__(
        self,
        name: str,
        model_dir: Path,
        *,
        clusterer: str = "gmm",
        q_table: np.ndarray | None = None,
        alpha: float = 0.02,
        epsilon: float = 0.1,
        td_clip: float = 10.0,
        seed: int = 7,
        device_name: str = "cpu",
    ):
        if clusterer == "raw":
            raise ValueError("Cluster Q-learning requires kmeans or gmm.")
        if not 0 < alpha <= 1 or not 0 <= epsilon <= 1 or td_clip <= 0:
            raise ValueError("Need 0 < alpha <= 1, 0 <= epsilon <= 1, and td_clip > 0.")
        super().__init__(
            name,
            model_dir,
            clusterer=clusterer,
            decision="q",
            seed=seed,
            device_name=device_name,
        )
        self.q_table = self.component_q.copy() if q_table is None else q_table
        if self.q_table.shape != self.component_q.shape:
            raise ValueError("Shared Q-table shape does not match the cluster artifact.")
        self.component_q = self.q_table
        self.alpha = alpha
        self.epsilon = epsilon
        self.td_clip = td_clip
        self.pending: tuple[np.ndarray, int, float, float] | None = None
        self.td_updates = 0
        self.abs_td_sum = 0.0
        self.clipped_updates = 0

    def choose_action(
        self, state: dict[str, Any], valid_actions: Sequence[str]
    ) -> str | None:
        valid = tuple(valid_actions)
        if not valid:
            return None
        latent, _, _ = self._outputs(state, valid)
        responsibility = self._responsibility(latent)
        legal = np.asarray([ACTION_NAMES.index(action) for action in valid])
        scores = responsibility @ self.q_table
        scale = max(float(state.get("ante", 1)), float(state.get("pot", 0)), 1.0)
        invested = float(state.get("my_invested", 0))

        if self.pending is not None:
            previous_r, previous_action, previous_scale, previous_invested = self.pending
            target = (
                scale * float(scores[legal].max()) - (invested - previous_invested)
            ) / previous_scale
            self._update(previous_r, previous_action, target)

        if self.rng.random() < self.epsilon:
            action_index = self.rng.choice(legal.tolist())
        else:
            best = scores[legal].max()
            action_index = self.rng.choice(legal[np.isclose(scores[legal], best)].tolist())
        action = ACTION_NAMES[action_index]
        self.pending = (responsibility, action_index, scale, invested)

        positive = responsibility[responsibility > 0]
        self.cluster_mass += responsibility
        self.top_weight_sum += float(responsibility.max())
        self.entropy_sum -= float(np.sum(positive * np.log(positive)))
        self.decisions += 1
        self.action_counts[action] += 1
        return action

    def observe_reward(self, reward: int, final_state: dict[str, Any] | None = None) -> None:
        if self.pending is None:
            return
        responsibility, action, scale, invested = self.pending
        self._update(responsibility, action, (float(reward) + invested) / scale)
        self.pending = None

    def _update(self, responsibility: np.ndarray, action: int, target: float) -> None:
        prediction = float(responsibility @ self.q_table[:, action])
        td_error = target - prediction
        clipped = float(np.clip(td_error, -self.td_clip, self.td_clip))
        self.q_table[:, action] += self.alpha * clipped * responsibility
        self.td_updates += 1
        self.abs_td_sum += abs(td_error)
        self.clipped_updates += int(clipped != td_error)

    def diagnostics(self) -> dict[str, object]:
        result = super().diagnostics()
        result.update(
            {
                "td_updates": self.td_updates,
                "mean_abs_td": self.abs_td_sum / max(self.td_updates, 1),
                "clipped_updates": self.clipped_updates,
            }
        )
        return result


def run_cluster_q_learning(
    model_dir: Path,
    output_dir: Path,
    *,
    clusterer: str = "gmm",
    hands: int = 20_000,
    alpha: float = 0.02,
    epsilon: float = 0.1,
    td_clip: float = 10.0,
    ante: int = 1000,
    seed: int = 7,
    progress_hands: int = 1000,
    device_name: str = "cpu",
) -> dict[str, object]:
    if hands <= 0 or ante <= 0 or progress_hands < 0:
        raise ValueError("hands and ante must be positive; progress_hands must be non-negative.")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    agent_a = ClusterQLearningAgent(
        "Player_1",
        model_dir,
        clusterer=clusterer,
        alpha=alpha,
        epsilon=epsilon,
        td_clip=td_clip,
        seed=seed * 2 + 1,
        device_name=device_name,
    )
    initial_q = agent_a.q_table.copy()
    agent_b = ClusterQLearningAgent(
        "Player_2",
        model_dir,
        clusterer=clusterer,
        q_table=agent_a.q_table,
        alpha=alpha,
        epsilon=epsilon,
        td_clip=td_clip,
        seed=seed * 2 + 2,
        device_name=device_name,
    )

    started = time.perf_counter()
    for hand_index in range(hands):
        random.seed(seed + hand_index)
        game = PokerGame(["Player_1", "Player_2"], log_file=None, ante=ante, game_mode="ev")
        with contextlib.redirect_stdout(io.StringIO()):
            game.play_hand({"Player_1": agent_a, "Player_2": agent_b})
        if progress_hands and (hand_index + 1) % progress_hands == 0:
            elapsed = time.perf_counter() - started
            print(f"{hand_index + 1:,}/{hands:,} hands ({(hand_index + 1) / elapsed:,.1f}/s)")

    output_dir.mkdir(parents=True)
    artifact_name = "spherical_kmeans.npz" if clusterer == "kmeans" else "diagonal_gmm.npz"
    for name in ("raw_mlp.pt", "spherical_kmeans.npz", "diagonal_gmm.npz"):
        shutil.copy2(model_dir / name, output_dir / name)
    artifact_path = output_dir / artifact_name
    with np.load(artifact_path, allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in artifact.files}
    arrays["component_q"] = agent_a.q_table
    np.savez_compressed(artifact_path, **arrays)

    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "source_model": str(model_dir.resolve()),
        "output_model": str(output_dir.resolve()),
        "clusterer": clusterer,
        "hands": hands,
        "alpha": alpha,
        "epsilon": epsilon,
        "td_clip": td_clip,
        "elapsed_seconds": elapsed,
        "hands_per_second": hands / max(elapsed, 1e-9),
        "q_mean_absolute_change": float(np.mean(np.abs(agent_a.q_table - initial_q))),
        "q_max_absolute_change": float(np.max(np.abs(agent_a.q_table - initial_q))),
        "agent_a": agent_a.diagnostics(),
        "agent_b": agent_b.diagnostics(),
    }
    (output_dir / "q_learning.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TD(0) self-play on a cluster Q-table.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusterer", choices=("kmeans", "gmm"), default="gmm")
    parser.add_argument("--hands", type=int, default=20_000)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--td-clip", type=float, default=10.0)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-hands", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_cluster_q_learning(
        args.model_dir,
        args.output,
        clusterer=args.clusterer,
        hands=args.hands,
        alpha=args.alpha,
        epsilon=args.epsilon,
        td_clip=args.td_clip,
        ante=args.ante,
        seed=args.seed,
        progress_hands=args.progress_hands,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
