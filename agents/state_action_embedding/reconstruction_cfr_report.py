"""Plot matched raw/latent policy payoffs and known-count state reconstruction.

Usage: python -m agents.state_action_embedding.reconstruction_cfr_report --run RUN
Gap: python -m agents.state_action_embedding.reconstruction_cfr_report --gap-run OUT
Particle audit: add --gap-reference PREVIOUS_GAP_RUN to compare the same root subset.
Uses the existing matplotlib Python; no Torch dependency or new model training.
Output: RUN/RESULTS.md, comparison.png, state_topk.png. Missing BR-gap/LBR is explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(run):
    records = []
    for path in sorted(run.glob("*/evaluation_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError("No completed evaluations")
    configs = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in run.glob("*/config.json")
    }
    conditions = (
        "reference",
        "encoder_seed",
        "buckets",
        "seed",
        "batch",
        "exploration",
        "protocol",
    )
    if "raw" in configs and "latent" in configs:
        if any(configs["raw"][key] != configs["latent"][key] for key in conditions):
            raise ValueError(
                "Raw/latent experiment conditions differ; do not label them matched"
            )
    static_path = run / "topk_static.json"
    lines = [
        "# Frozen Reconstruction Buckets: Completed Measurements",
        "",
        "- Frozen raw/latent k-means; batched signed outcome-sampling MCCFR.",
        "- One hand = one traverser's sampled trajectory. Matched policy batches and legal/depth partitions.",
        "- Current and average policies are separate. Uniform H4 for every player/opponent.",
        "- Local BR-gap, LBR and exact exploitability: **not evaluated**. Old C++ results are not a matched baseline.",
        "- Reconstruction weights are frozen: checkpoint-to-checkpoint changes reflect visited-state distribution, not encoder learning.",
        "- Only the completed budgets below are results; future budgets in the README are not measurements.",
        "- One training seed per method; intervals are normal approximations, not robust tail guarantees.",
        "- Source conditions/commands: [README](../../RECONSTRUCTION_CFR.md).",
        "",
    ]
    for method, config in sorted(configs.items()):
        lines.append(
            f"- {method}: encoder seed{config['encoder_seed']}, solver seed{config['seed']}, "
            f"K={config['buckets']}, batch={config['batch']}, epsilon={config['exploration']}, "
            f"evaluation pairs/opponent/checkpoint={config['eval_pairs']}."
        )
    lines.append("")
    if static_path.exists():
        static = json.loads(static_path.read_text(encoding="utf-8"))
        lines += [
            "## Static State Reconstruction",
            "",
            "Existing held-out low-fold panel; own betting turns, prefixes macro-averaged within each hand. k comes from observable event counts.",
            "This reconstructs the current observation, not the next state. Exact means all k cards are correct; recall is the fraction recovered.",
            "",
            "| Scope | Card group | Top-k exact | Recall@k |",
            "|---|---|---:|---:|",
        ]
        for scope, values in static["scopes"].items():
            for group in ("hidden", "public", "discard", "opp_public"):
                m = values["hand_macro"]
                lines.append(
                    f"| {scope} | {group} | {m[group + '_topk_exact']:.2%} | {m[group + '_topk_recall']:.2%} |"
                )
        overall = static["scopes"]["all"]["hand_macro"]
        lines += [
            "",
            f"Saved/online latent reload maximum absolute error: {static['latent_reload_max_error']:.3g}.",
            f"All card groups jointly exact: {overall['cards_topk_exact']:.2%}; cross-group card collisions: {overall['topk_card_collision']:.2%} (not repaired).",
            f"Legal-mask exact: {overall['legal_exact']:.2%}; street accuracy: {overall['street_accuracy']:.2%}; actor accuracy: {overall['actor_accuracy']:.2%}.",
            f"Chip-field MAE: {overall['chip_mae']:.3f} chips; count-field MAE: {overall['count_mae']:.3f}.",
            "",
        ]
        figure, panels = plt.subplots(1, 2, figsize=(12, 4))
        names = ("hidden", "public", "discard", "opp_public")
        positions = np.arange(len(names))
        for index, (scope, color) in enumerate(
            (("street_5", "#227c70"), ("street_6", "#bf3e54"), ("street_7", "#4765b3"))
        ):
            values = static["scopes"][scope]["hand_macro"]
            for panel, suffix in zip(panels, ("topk_exact", "topk_recall")):
                panel.bar(
                    positions + (index - 1) * 0.24,
                    [100 * values[f"{name}_{suffix}"] for name in names],
                    width=0.24,
                    color=color,
                    label=scope.replace("_", " "),
                )
        for panel, title in zip(panels, ("Top-k exact set", "Recall@k")):
            panel.set_xticks(
                positions, ("Own hidden", "Own public", "Discard", "Opp. public")
            )
            panel.set_ylim(0, 105)
            panel.set_ylabel("Percent")
            panel.set_title(title)
            panel.legend()
            panel.grid(axis="y", alpha=0.2)
        figure.suptitle(
            "Frozen encoder state reconstruction: known observable card counts"
        )
        figure.tight_layout()
        figure.savefig(run / "state_topk.png", dpi=150)
        plt.close(figure)
        lines += ["![State reconstruction](state_topk.png)", ""]
    lines += [
        "## Policy Evaluation",
        "",
        "Chips/hand; positive means the tested policy wins. Each pair swaps policy seats with the same shuffled deck.",
        "",
        "| Hands | Method | Policy | Opponent | Mean | Normal 95% interval | Missing buckets |",
        "|---:|---|---|---|---:|---|---:|",
    ]
    for row in sorted(
        records, key=lambda v: (v["hands"], v["opponent"], v["policy"], v["method"])
    ):
        lo, hi = row["normal95"]
        lines.append(
            f"| {row['hands']} | {row['method']} | {row['policy']} | {row['opponent']} | {row['mean_chips']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['missing']} |"
        )
    lines += [
        "",
        "## Paired Latent Minus Raw",
        "",
        "Positive favors latent. These are paired differences, not a comparison of overlapping independent intervals.",
        "",
        "| Hands | Policy | Opponent | Difference | Normal 95% interval |",
        "|---:|---|---|---:|---|",
    ]
    indexed = {
        (r["method"], r["hands"], r["policy"], r["opponent"]): r for r in records
    }
    for row in records:
        if row["method"] != "latent":
            continue
        baseline = indexed.get(("raw", row["hands"], row["policy"], row["opponent"]))
        if baseline is None:
            lines.append(f"Missing matched raw baseline at {row['hands']} hands.")
            continue
        delta = np.array(row["pair_payoffs"]) - np.array(baseline["pair_payoffs"])
        mean, se = delta.mean(), delta.std(ddof=1) / np.sqrt(len(delta))
        lines.append(
            f"| {row['hands']} | {row['policy']} | {row['opponent']} | {mean:.3f} | [{mean - 1.96 * se:.3f}, {mean + 1.96 * se:.3f}] |"
        )
    lines += [
        "",
        "## State Reconstruction on Policy Visits",
        "",
        "Uniform-opponent panel, hand-macro averages. Actual observed legal actions control execution, not predicted masks.",
        "",
        "| Hands | Method | Policy | Hidden exact | Hidden recall | Discard exact | Opp.public recall | Legal exact | Chip MAE |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        if row["opponent"] == "uniform":
            m = row["reconstruction"]
            lines.append(
                f"| {row['hands']} | {row['method']} | {row['policy']} | {m['hidden_topk_exact']:.2%} | {m['hidden_topk_recall']:.2%} | {m['discard_topk_exact']:.2%} | {m['opp_public_topk_recall']:.2%} | {m['legal_exact']:.2%} | {m['chip_mae']:.2f} |"
            )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for method, color in (("raw", "#227c70"), ("latent", "#bf3e54")):
        for policy, style in (("current", "--"), ("average", "-")):
            label = f"{method} k-means / {policy}"
            for column, opponent in enumerate(("uniform", "heuristic")):
                selected = sorted(
                    [
                        v
                        for v in records
                        if v["method"] == method
                        and v["policy"] == policy
                        and v["opponent"] == opponent
                    ],
                    key=lambda v: v["hands"],
                )
                if not selected:
                    continue
                x = [v["hands"] for v in selected]
                axes[0, column].errorbar(
                    x,
                    [v["mean_chips"] for v in selected],
                    yerr=[1.96 * v["se"] for v in selected],
                    color=color,
                    linestyle=style,
                    marker="o",
                    capsize=3,
                    label=label,
                )
                if opponent == "uniform":
                    for index, metric in enumerate(
                        ("hidden_topk_exact", "hidden_topk_recall")
                    ):
                        axes[1, index].plot(
                            x,
                            [100 * v["reconstruction"][metric] for v in selected],
                            color=color,
                            linestyle=style,
                            marker="o",
                            label=label,
                        )
    for ax, title in zip(
        axes.flat,
        (
            "Net chips vs uniform",
            "Net chips vs heuristic",
            "Own hidden: top-k exact (%)",
            "Own hidden: recall@k (%)",
        ),
    ):
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel("Training hands")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_ylabel("Chips / hand")
    fig.suptitle("Frozen observation reconstruction + bucket MCCFR", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Raw k-means overlay; signed payoff axis. No Local BR-gap / LBR / exploitability measurement.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(run / "comparison.png", dpi=150)
    plt.close(fig)
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run / "RESULTS.md")


def gap_report(directory, reference=None):
    """Summarize matched original-information-set roots, independent of bucket IDs."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "per_root.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in ("method", "hands", "policy", "root"))
        if key in indexed or row["particles"] != config["particles"]:
            raise ValueError("Duplicate row or mismatched particle count")
        indexed[key] = row
    expected = len(config["methods"]) * len(config["hands"]) * 2 * config["roots"]
    if len(rows) != expected or set(config["methods"]) != {"raw", "latent"}:
        raise ValueError("Complete raw and latent evaluations required")
    for row in rows:
        peer = indexed[
            (
                "latent" if row["method"] == "raw" else "raw",
                row["hands"],
                row["policy"],
                row["root"],
            )
        ]
        if any(
            row[k] != peer[k]
            for k in ("root_seed", "observation_sha256", "viewer", "legal")
        ):
            raise ValueError("Unmatched root information set")

    def interval(values):
        values = np.asarray(values, float)
        mean = float(values.mean())
        se = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return mean, se

    groups, summaries = {}, []
    for hands in sorted(config["hands"]):
        for policy in ("current", "average"):
            for method in ("raw", "latent"):
                selected = [
                    indexed[(method, hands, policy, root)]
                    for root in range(config["roots"])
                ]
                groups[(method, hands, policy)] = selected
                values = {
                    metric: interval([r[metric] for r in selected])
                    for metric in (
                        "local_br_gap",
                        "half_particle_gap",
                        "crossfit_deviation",
                    )
                }
                summaries.append(
                    dict(
                        method=method,
                        hands=hands,
                        policy=policy,
                        **{
                            metric: dict(
                                mean=mean,
                                se=se,
                                normal95=[mean - 1.96 * se, mean + 1.96 * se],
                            )
                            for metric, (mean, se) in values.items()
                        },
                        missing=sum(r["missing"] + r["root_missing"] for r in selected),
                    )
                )
    lines = [
        "# Root Local BR-gap: Reconstruction Buckets",
        "",
        f"- {config['roots']} common fifth-street first-decision information sets, {config['particles']} conditional hidden worlds/root, evaluation seed {config['seed']}.",
        "- Current and average policies evaluated separately; signed outcome-sampling MCCFR training, no new training.",
        "- Uniform discard/reveal posterior. Raw and latent use identical roots, particles and action RNG seeds.",
        "- Metric: max_a mean_world Q(I,a) - sum_a sigma(I,a) mean_world Q(I,a), in ante units. Lower is better.",
        "- Root information sets and their weights stay fixed, not bucket IDs or cluster sizes. This permits comparison after refinement.",
        "- First-decision diagnostic only: not exact exploitability, a full best response, or LBR profit.",
        "- Finite-sample max is optimistic. Crossfit selects an action on one half and evaluates on the other, then swaps; it is a separate signed deviation estimate, NOT unbiased exploitability.",
        "- One training seed; root-level normal intervals are descriptive. Old C++ fixed-H4 results are unmatched and not overlaid.",
        "- Raw current observation vs latent event history remain different inputs. Frozen low-fold encoder/centroids were not refit.",
        "",
        "| Hands | Policy | Method | Local gap | Normal 95% interval | Half-particle gap | Crossfit deviation | Missing |",
        "|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in summaries:
        gap = row["local_br_gap"]
        lo, hi = gap["normal95"]
        lines.append(
            f"| {row['hands']} | {row['policy']} | {row['method']} | {gap['mean']:.4f} | [{lo:.4f}, {hi:.4f}] | {row['half_particle_gap']['mean']:.4f} | {row['crossfit_deviation']['mean']:.4f} | {row['missing']} |"
        )
    lines += [
        "",
        "## Paired Latent Minus Raw",
        "",
        "Negative favors latent. Ratios divide population-panel means, not individual root gaps.",
        "",
        "| Hands | Policy | Difference | Normal 95% interval | Latent/raw mean gap |",
        "|---:|---|---:|---|---:|",
    ]
    pairs = []
    for hands in sorted(config["hands"]):
        for policy in ("current", "average"):
            raw, latent = (
                np.array([r["local_br_gap"] for r in groups[(method, hands, policy)]])
                for method in ("raw", "latent")
            )
            mean, se = interval(latent - raw)
            ratio = float(latent.mean() / raw.mean()) if raw.mean() > 0 else None
            pairs.append(
                dict(
                    hands=hands,
                    policy=policy,
                    difference=mean,
                    normal95=[mean - 1.96 * se, mean + 1.96 * se],
                    ratio=ratio,
                )
            )
            ratio_text = "undefined" if ratio is None else f"{ratio:.4f}"
            lines.append(
                f"| {hands} | {policy} | {mean:.4f} | [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}] | {ratio_text} |"
            )
    if reference is not None:
        reference_config = json.loads(
            (reference / "config.json").read_text(encoding="utf-8")
        )
        if any(
            config[key] != reference_config[key]
            for key in ("seed", "protocol", "training_configs")
        ):
            raise ValueError(
                "Particle audit requires the same policies and evaluation protocol"
            )
        reference_rows = {
            tuple(row[k] for k in ("method", "hands", "policy", "root")): row
            for row in map(
                json.loads,
                (reference / "per_root.jsonl").read_text(encoding="utf-8").splitlines(),
            )
        }
        lines += [
            "",
            "## Particle Count Check on the Same Roots",
            "",
            f"Only the same {config['roots']} roots are compared here; the larger reference panel is NOT substituted for this subset.",
            "Policies are unchanged. Differences below come from evaluation sampling, not training or refinement.",
            "",
            f"| Hands | Policy | Method | Gap ({reference_config['particles']} particles) | Gap ({config['particles']} particles) |",
            "|---:|---|---|---:|---:|",
        ]
        for row in summaries:
            selected = [
                reference_rows[(row["method"], row["hands"], row["policy"], root)]
                for root in range(config["roots"])
            ]
            for old, new in zip(
                selected, groups[(row["method"], row["hands"], row["policy"])]
            ):
                if any(
                    old[k] != new[k]
                    for k in ("root_seed", "observation_sha256", "legal")
                ):
                    raise ValueError("Particle audit roots differ")
            old_mean = np.mean([r["local_br_gap"] for r in selected])
            lines.append(
                f"| {row['hands']} | {row['policy']} | {row['method']} | {old_mean:.4f} | {row['local_br_gap']['mean']:.4f} |"
            )
    lines += [
        "",
        "![Local BR-gap comparison](local_br_gap.png)",
        "",
        "Error bars in the top panels show one standard error and are displayed only when the lower endpoint is positive; full normal intervals are tabulated above. Lines connect measured budgets, not fitted convergence forecasts.",
        "",
        "## Sampling Diagnostics",
        "",
        f"Half-particle uses {config['particles']//2} worlds; its average exceeds/equal the full-panel plug-in gap by convexity. A large discrepancy warns of max-selection noise.",
        "Crossfit may be negative, so its panel uses a linear signed axis. It measures the out-of-sample gain of a sampled one-step action choice, not the exact best action.",
        "Raw/latent numbers on each root include the full policy continuation. A smaller local gap across different continuations is not itself a proof of strategic superiority.",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, policy in enumerate(("current", "average")):
        for method, color in (("raw", "#227c70"), ("latent", "#bf3e54")):
            selected = [
                r for r in summaries if r["method"] == method and r["policy"] == policy
            ]
            x = np.array([r["hands"] for r in selected])
            y = np.array([r["local_br_gap"]["mean"] for r in selected])
            se = np.array([r["local_br_gap"]["se"] for r in selected])
            valid = y > 0
            axes[0, column].plot(
                x[valid], y[valid], marker="o", color=color, label=f"{method} k-means"
            )
            valid_error = (y - se) > 0
            axes[0, column].errorbar(
                x[valid_error],
                y[valid_error],
                yerr=se[valid_error],
                fmt="none",
                color=color,
                capsize=3,
            )
            if not valid.all():
                lines.append(
                    f"Zero/nonpositive aggregate gaps for {method}/{policy} omitted from logarithmic panel; see table."
                )
            axes[1, 1].errorbar(
                x,
                [r["crossfit_deviation"]["mean"] for r in selected],
                yerr=[r["crossfit_deviation"]["se"] for r in selected],
                color=color,
                linestyle="--" if policy == "current" else "-",
                marker="o",
                capsize=3,
                label=f"{method} / {policy}",
            )
        selected = [
            r for r in pairs if r["policy"] == policy and r["ratio"] is not None
        ]
        axes[1, 0].plot(
            [r["hands"] for r in selected],
            [r["ratio"] for r in selected],
            marker="o",
            label=policy,
        )
        axes[0, column].set_title(f"Local BR-gap / {policy}")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Ante (log scale)")
    axes[1, 0].set_title("Latent / raw mean Local BR-gap")
    axes[1, 0].axhline(1, color="#555555", linewidth=0.8)
    axes[1, 1].set_title("Held-out one-step deviation (crossfit)")
    axes[1, 1].set_ylabel("Ante, signed")
    axes[1, 1].axhline(0, color="#555555", linewidth=0.8)
    for panel in axes.flat:
        panel.set_xscale("log")
        panel.set_xlabel("Training hands")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8)
    fig.suptitle("Fixed original information sets: reconstruction-bucket MCCFR")
    fig.text(
        0.5,
        0.015,
        f"{config['roots']} paired roots; {config['particles']} hidden worlds/root; uniform H4. Not exploitability or LBR.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(directory / "local_br_gap.png", dpi=150)
    plt.close(fig)
    (directory / "summary.json").write_text(
        json.dumps(dict(groups=summaries, paired=pairs), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(directory / "RESULTS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--gap-run", type=Path)
    parser.add_argument("--gap-reference", type=Path)
    args = parser.parse_args()
    if args.gap_reference and not args.gap_run:
        parser.error("--gap-reference requires --gap-run")
    gap_report(args.gap_run, args.gap_reference) if args.gap_run else report(args.run)
