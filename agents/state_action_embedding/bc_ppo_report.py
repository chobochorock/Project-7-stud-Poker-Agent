"""Report completed heuristic-BC/PPO runs without importing PyTorch.

Usage: python -m agents.state_action_embedding.bc_ppo_report --out-dir RUN
Outputs: summary.json, comparison.png, learning.png. Uncertainty is paired-deal
bootstrap conditional on trained models; training-seed SD is reported separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("random", "heuristic", "bc", "ppo_best", "ppo_last")
LABELS = (
    "Random",
    "Heuristic",
    "BC",
    "BC + PPO\nselected",
    "BC + PPO\nlast",
)
COLORS = ("#9b9b9b", "#555555", "#427ba1", "#168568", "#b44c68")


def summarize(values: np.ndarray) -> dict:
    """Input shape: training seeds x shuffled deals x two exchanged seats."""
    values = np.asarray(values, dtype=np.float64)
    deal_means = values.mean(axis=(0, 2))
    seed_means = values.mean(axis=(1, 2))
    rng = np.random.default_rng(20260919)
    boot = [
        deal_means[rng.integers(len(deal_means), size=len(deal_means))].mean()
        for _ in range(3000)
    ]
    return {
        "mean_chips_per_hand": float(deal_means.mean()),
        "paired_deal_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        "training_seed_sd": float(seed_means.std()),
        "per_seed_chips_per_hand": seed_means.tolist(),
        "positive_fraction": float((values > 0).mean()),
        "zero_fraction": float((values == 0).mean()),
        "negative_fraction": float((values < 0).mean()),
        "mean_positive_outcome_chips": float(values[values > 0].mean())
        if (values > 0).any()
        else 0.0,
        "mean_negative_outcome_chips": float(values[values < 0].mean())
        if (values < 0).any()
        else 0.0,
        "loss_100chips_or_more_fraction": float((values <= -100).mean()),
        "distinct_deals": len(deal_means),
        "hands_per_seed": values.shape[1] * values.shape[2],
    }


def audit_demonstrations(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        tokens, lengths, hands, split = (
            data[k] for k in ("tokens", "lengths", "hand", "split")
        )
        assert (lengths > 0).all()
        for partition in range(3):
            mine = set(hands[split == partition])
            others = set(hands[split != partition])
            assert mine.isdisjoint(others)
        private = np.isin(tokens[..., 0], [3, 5]) & (tokens[..., 1] == 1)
        assert (tokens[..., 2][private] == 0).all()
        decision = tokens[np.arange(len(tokens)), lengths - 1]
        assert (decision[:, 0] == 8).all() and (decision[:, 1] == 0).all()
        assert (decision[:, 3] == 0).all()
        assert data["legal"][np.arange(len(tokens)), data["action"]].all()
        counts = np.bincount(data["action"][split == 0], minlength=20) + 1
        test = split == 2
        probability = data["legal"][test] * counts
        probability = probability / probability.sum(-1, keepdims=True)
        target = data["action"][test]
        nll = -np.log(probability[np.arange(len(target)), target])
        correct = probability.argmax(-1) == target
        frequency = {
            label: {
                "accuracy": float(correct[mask].mean()),
                "nll": float(nll[mask].mean()),
            }
            for label, mask in (
                ("all", target >= 0),
                ("bet", target < 8),
                ("discard_reveal", target >= 8),
            )
        }
        return {
            "hand_split_disjoint": True,
            "opponent_private_values_hidden": True,
            "actor_view_and_no_target_in_query": True,
            "decisions_by_split": [int((split == i).sum()) for i in range(3)],
            "train_frequency_legal_masked_baseline": frequency,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.out_dir.glob("seed*/evaluation.json"))
    if not paths:
        parser.error("No completed evaluation.json files")
    runs, arrays, training, configs = [], [], [], []
    for path in paths:
        runs.append(json.loads(path.read_text()))
        training.append(json.loads((path.parent / "training.json").read_text()))
        configs.append(json.loads((path.parent / "config.json").read_text()))
        with np.load(path.parent / "evaluation.npz", allow_pickle=False) as data:
            arrays.append(dict(data))
    for run in runs:
        assert run["deal_seed_base"] == runs[0]["deal_seed_base"]
    for config in configs:
        for key in (
            "dim",
            "bc_steps",
            "ppo_updates",
            "rollout_hands",
            "batch",
            "source_sha256",
        ):
            assert config[key] == configs[0][key], f"Unmatched configuration: {key}"
    raw = {policy: np.stack([a[policy] for a in arrays]) for policy in POLICIES}
    np.testing.assert_array_equal(raw["heuristic"].sum(-1), 0)
    assert summarize(np.zeros((2, 8, 2)))["paired_deal_bootstrap_95pct"] == [0, 0]
    summary = {
        "seeds": [r["seed"] for r in runs],
        "policies": {policy: summarize(values) for policy, values in raw.items()},
        "ppo_best_minus_bc": summarize(raw["ppo_best"] - raw["bc"]),
        "ppo_last_minus_bc": summarize(raw["ppo_last"] - raw["bc"]),
        "bc_heldout_imitation": {
            stage: {
                metric: {
                    "mean": float(
                        np.mean([t["bc_test"][stage][metric] for t in training])
                    ),
                    "seed_sd": float(
                        np.std([t["bc_test"][stage][metric] for t in training])
                    ),
                }
                for metric in ("nll", "accuracy")
            }
            for stage in ("all", "bet", "discard_reveal")
        },
        "selected_bc_steps": [t["bc_selected_step"] for t in training],
        "selected_ppo_updates": [t["ppo_selected_update"] for t in training],
        "ppo_hands_per_seed": [t["total_training_hands"] for t in training],
        "learner_decisions_per_seed": [t["learner_decisions"] for t in training],
        "resources": {
            phase: {
                key: float(np.mean([t[f"{phase}_memory"][key] for t in training]))
                for key in ("seconds", "rss_sampled_peak_mib")
            }
            for phase in ("bc", "ppo")
        },
        "audit": audit_demonstrations(args.out_dir / "demonstrations.npz"),
        "scope": "Fixed heuristic opponent, stochastic BC/PPO, not self-play/Nash/exploitability; no PPO-from-scratch comparison.",
        "uncertainty": "Bootstrap shuffled-deal pairs after averaging per-model returns. Conditional on trained models; seed variation reported separately. Positive fractions of delta arrays mean improvement frequency, not game win rate.",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    ax = axes[0]
    metrics = [summary["policies"][p] for p in POLICIES]
    means = np.array([m["mean_chips_per_hand"] for m in metrics])
    intervals = np.array([m["paired_deal_bootstrap_95pct"] for m in metrics])
    ax.bar(range(5), means, color=COLORS)
    ax.errorbar(
        range(5),
        means,
        yerr=np.stack((means - intervals[:, 0], intervals[:, 1] - means)).clip(0),
        fmt="none",
        color="black",
        capsize=4,
    )
    for i, metric in enumerate(metrics):
        ax.scatter(
            np.full(len(runs), i),
            metric["per_seed_chips_per_hand"],
            color="black",
            s=12,
            zorder=3,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(5), LABELS, fontsize=8)
    ax.set_title("Test net chips/hand (higher better)")
    ax.set_ylabel("Net chips / hand")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for i, policy in enumerate(("ppo_best", "ppo_last")):
        metric = summary[f"{policy}_minus_bc"]
        mean = metric["mean_chips_per_hand"]
        low, high = metric["paired_deal_bootstrap_95pct"]
        ax.errorbar(
            i,
            mean,
            yerr=[[max(0, mean - low)], [max(0, high - mean)]],
            fmt="o",
            color=COLORS[3 + i],
            capsize=5,
        )
        ax.text(i, high + 0.1, f"{mean:+.3f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(2), ["Validation-selected", "Final budget"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("PPO improvement over BC (paired deals)")
    ax.set_ylabel("Change in chips / hand")
    ax.grid(axis="y", alpha=0.2)
    ax.margins(y=0.2)

    ax = axes[2]
    x = np.arange(5)
    loss = np.array([m["negative_fraction"] for m in metrics])
    tie = np.array([m["zero_fraction"] for m in metrics])
    win = np.array([m["positive_fraction"] for m in metrics])
    ax.bar(x, loss, color="#b44c68", label="Loss")
    ax.bar(x, tie, bottom=loss, color="#aaaaaa", label="Tie")
    ax.bar(x, win, bottom=loss + tie, color="#168568", label="Win")
    ax.set_xticks(x, LABELS, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Hand outcomes (not the reward objective)")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"Heuristic imitation -> PPO against the same fixed heuristic | {len(runs)} seeds\n{summary['policies']['bc']['hands_per_seed']} unseen seat-paired hands per policy/seed; 95% paired-deal bootstrap; dots = training seeds",
        fontsize=12,
    )
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for t in training:
        bc = [p for p in t["curve"] if p["stage"] == "bc"]
        label = f"seed {t['seed']}"
        axes[0].plot(
            [p["step"] for p in bc], [p["bet"]["accuracy"] for p in bc], label=label
        )
        axes[1].plot(
            [p["step"] for p in bc],
            [p["discard_reveal"]["accuracy"] for p in bc],
            label=label,
        )
        ppo = [p for p in t["curve"] if p["stage"] == "ppo" and "validation" in p]
        axes[2].plot(
            [p["train_hands"] for p in ppo],
            [p["validation"]["chips_per_hand"] for p in ppo],
            "o-",
            markersize=3,
            label=label,
        )
    for ax, title in zip(
        axes,
        (
            "BC: betting top-1 agreement",
            "BC: discard/reveal pair agreement",
            "PPO: validation chips/hand",
        ),
    ):
        ax.set_title(title)
        ax.set_xlabel("BC updates" if ax is not axes[2] else "Fresh PPO training hands")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].axhline(0, color="black", linewidth=0.8)
    fig.suptitle(
        "Validation curves only | selected before held-out match evaluation",
        fontsize=12,
    )
    fig.savefig(args.out_dir / "learning.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("scope", "uncertainty")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
