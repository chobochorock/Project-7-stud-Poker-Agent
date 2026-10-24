"""Report the matched raw/fixed/split 7-stud pilot, without importing Torch.

Usage: python -m agents.state_action_embedding.refinement_report --run RUN
Optional: --initial-run PER_HAND_RUN adds its verified, matched uniform panel.
Output: RUN/RESULTS.md, summary.json, local_br_gap.png, refinement_cost.png.
Requires completed evaluations for every saved snapshot; no fitted forecasts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ARMS = ("raw", "fixed", "split")
LABELS = {"raw": "Raw k-means", "fixed": "Latent fixed", "split": "Latent split"}
COLORS = {"raw": "#227c70", "fixed": "#4765b3", "split": "#bf3e54"}


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return dict(mean=mean, se=se, normal95=[mean - 1.96 * se, mean + 1.96 * se])


def initial_panel(directory, evaluation, identities):
    """Reuse actual uniform-policy rows, never assign them a fictitious budget."""
    meta = json.loads((directory / "per_hand_config.json").read_text(encoding="utf-8"))
    if (meta["particles"], meta["seed"]) != (
        evaluation["particles"],
        evaluation["seed"],
    ):
        raise ValueError("Initial policy uses a different evaluator")
    selected = [
        row
        for row in read_rows(directory / "per_hand_gap.jsonl")
        if row["arm"] == "raw"
        and row["policy"] == "current"
        and row["root"] < evaluation["roots"]
    ]
    if len(selected) != evaluation["roots"] or {v["root"] for v in selected} != set(
        range(evaluation["roots"])
    ):
        raise ValueError("Incomplete initial root panel")
    for row in selected:
        signature = tuple(row[k] for k in ("root_seed", "observation_sha256")) + (
            tuple(row["legal"]),
        )
        legal, strategy = np.asarray(row["legal"], bool), np.asarray(row["strategy"])
        if (
            row["policy_hands"] != 0
            or identities[row["root"]] != signature
            or not legal.any()
            or not np.allclose(strategy, legal.astype(float) / legal.sum())
        ):
            raise ValueError("Initial policy or observation does not match")
    return dict(
        **interval([v["local_br_gap"] for v in selected]),
        crossfit=interval([v["crossfit_deviation"] for v in selected]),
        source=str(directory.resolve()),
    )


def report(run, initial_run=None):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "gap_config.json").read_text(encoding="utf-8"))
    rows = read_rows(run / "gap.jsonl")
    groups, identities = {}, {}
    for row in rows:
        key = tuple(row[k] for k in ("arm", "hands", "phase", "policy"))
        panel = groups.setdefault(key, {})
        if row["root"] in panel or row["particles"] != evaluation["particles"]:
            raise ValueError("Duplicate or inconsistent evaluation")
        panel[row["root"]] = row
        signature = tuple(row[k] for k in ("root_seed", "observation_sha256")) + (
            tuple(row["legal"]),
        )
        if identities.setdefault(row["root"], signature) != signature:
            raise ValueError("Unmatched original information set")
    expected_groups = set()
    for path in evaluation["snapshots"]:
        path = Path(path)
        phase = (
            "checkpoint"
            if path.stem.startswith("checkpoint")
            else ("before" if path.stem.startswith("before") else "after")
        )
        hand = int(path.stem.split("_")[-1])
        expected_groups.update(
            (path.parent.name, hand, phase, policy) for policy in ("current", "average")
        )
    if set(groups) != expected_groups:
        raise ValueError("Incomplete snapshot evaluations")
    summaries = {}
    for key, panel in groups.items():
        expected = (
            evaluation["roots"] if key[2] == "checkpoint" else evaluation["split_roots"]
        )
        if set(panel) != set(range(expected)):
            raise ValueError("Incomplete root panel")
        groups[key] = [panel[i] for i in range(expected)]
        summaries[key] = {
            metric: interval([v[metric] for v in panel.values()])
            for metric in ("local_br_gap", "half_particle_gap", "crossfit_deviation")
        }
    hands = sorted({key[1] for key in groups if key[2] == "checkpoint"})
    initial = (
        initial_panel(initial_run, evaluation, identities) if initial_run else None
    )
    lines = [
        "# 7-stud Parent-local Refinement Pilot",
        "",
        "- IRA-inspired sampled heuristic; NOT original FPIRA/CFR+IRA, exact exploitability or LBR.",
        f"- Training seed {config['seed']}; zero initialization; signed outcome sampling; batch128, epsilon0.6; uniform H4.",
        f"- Evaluation seed {evaluation['seed']}; {evaluation['roots']} fixed original fifth-street roots, {evaluation['particles']} conditional particles; mean Q before max.",
        "- One training seed and heavy-tailed payoffs. Normal intervals are descriptive, not a guarantee.",
        "- New raw/fixed baselines use the same 10k batching boundaries. Do not substitute the old 100k training run.",
        "- Equal sampled-hand budgets do not mean equal decision counts or wall time; realized trajectory lengths and refinement costs differ.",
        "- Latent split and latent fixed have the same information input. Raw observes current features; latent sees event history.",
        "- Encoder/centroids are frozen and originally fitted on low-fold data. Split monitors use evolving training trajectories.",
        "- Learned split boundaries generalize inside a parent; they are not exact information-set separation or perfect-recall recovery.",
        "- Split effects include local regret/average reset; no reset-only control was run.",
        "- Commands and literature: [protocol](../../REFINEMENT_EXPERIMENT.md).",
        "",
        "## Local BR-gap",
        "",
        "Ante units; lower is better. Current and average strategies are separate.",
        "",
        "| Hands | Policy | Arm | Gap | Normal 95% interval | Crossfit deviation |",
        "|---:|---|---|---:|---|---:|",
    ]
    pairs = []
    no_splits = all(row["splits"] == 0 for row in rows)
    if no_splits:
        lines[2:2] = [
            "**No split passed the predeclared gate. The split arm remains the fixed policy; this run tests the gate, NOT the effect of accepted refinement. Do not interpret identical curves as evidence that useful splitting cannot work.**",
            "",
        ]
    if initial:
        lines[2:2] = [
            f"Matched untrained uniform reference: gap={initial['mean']:.4f}, crossfit={initial['crossfit']['mean']:.4f}. Reused the first {evaluation['roots']} pre-update uniform rows from the per-hand audit; this is not an MA100 value.",
            "",
        ]
    for hand in hands:
        for policy in ("current", "average"):
            for arm in ARMS:
                key = (arm, hand, "checkpoint", policy)
                values = summaries[key]
                gap = values["local_br_gap"]
                lo, hi = gap["normal95"]
                lines.append(
                    f"| {hand} | {policy} | {arm} | {gap['mean']:.4f} | [{lo:.4f}, {hi:.4f}] | {values['crossfit_deviation']['mean']:.4f} |"
                )
            split, fixed = (
                np.array(
                    [
                        v["local_br_gap"]
                        for v in groups[(arm, hand, "checkpoint", policy)]
                    ]
                )
                for arm in ("split", "fixed")
            )
            pairs.append(dict(hands=hand, policy=policy, **interval(split - fixed)))
    lines += [
        "",
        "## Paired Split Minus Fixed Latent",
        "",
        "Negative favors splitting on this diagnostic, not proven strategic superiority.",
        "",
        "| Hands | Policy | Difference | Normal 95% interval |",
        "|---:|---|---:|---|",
    ]
    for row in pairs:
        lo, hi = row["normal95"]
        lines.append(
            f"| {row['hands']} | {row['policy']} | {row['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |"
        )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, policy in enumerate(("current", "average")):
        if initial and initial["mean"] > 0:
            axes[0, column].axhline(
                initial["mean"],
                color="#777777",
                linestyle="--",
                label="Untrained uniform (same panel)",
            )
        for arm in ARMS:
            values = [
                summaries[(arm, hand, "checkpoint", policy)]["local_br_gap"]
                for hand in hands
            ]
            y, se = np.array([v["mean"] for v in values]), np.array(
                [v["se"] for v in values]
            )
            x = np.array(hands)
            positive = y > 0
            axes[0, column].plot(
                x[positive], y[positive], "o-", color=COLORS[arm], label=LABELS[arm]
            )
            visible = y - se > 0
            axes[0, column].errorbar(
                x[visible],
                y[visible],
                yerr=se[visible],
                fmt="none",
                color=COLORS[arm],
                capsize=3,
            )
            raw = np.array(
                [
                    summaries[("raw", hand, "checkpoint", policy)]["local_br_gap"][
                        "mean"
                    ]
                    for hand in hands
                ]
            )
            defined = raw > 0
            if arm != "raw":
                axes[1, column].plot(
                    x[defined],
                    y[defined] / raw[defined],
                    "o-",
                    color=COLORS[arm],
                    label=f"{LABELS[arm]} / raw",
                )
        axes[0, column].set_title(f"Local BR-gap: {policy}")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Ante (log scale)")
        axes[1, column].set_title(f"Ratio to matched raw k-means: {policy}")
        axes[1, column].axhline(1, color="#555555", linewidth=0.8)
    for panel in axes.flat:
        panel.set_xscale("log")
        panel.set_xlabel("Training hands")
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8)
    figure.suptitle(
        "7-stud: fixed vs local refinement (no split accepted)"
        if no_splits
        else "7-stud: fixed vs parent-local latent refinement"
    )
    figure.text(
        0.5,
        0.015,
        f"{evaluation['roots']} shared original roots; {evaluation['particles']} particles. Bars: 1 SE. Not exploitability/LBR.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(run / "local_br_gap.png", dpi=150)
    plt.close(figure)
    lines += [
        "",
        "![Local BR-gap](local_br_gap.png)",
        "",
        "Finite-sample max can inflate gaps. Crossfit chooses an action on one half and evaluates on the other; it is signed and is not an unbiased exact gap. Lines connect measured budgets only.",
        "",
        "## Before/After a Split",
        "",
        f"Same {evaluation['split_roots']} roots before/after; not mixed with the larger checkpoint panel. Includes immediate effects of reinitialization.",
        "",
        "| Hands | Policy | Before | After | Paired change |",
        "|---:|---|---:|---:|---:|",
    ]
    split_hands = sorted({key[1] for key in groups if key[2] == "before"})
    if not split_hands:
        lines += [
            "",
            "No split passed the predeclared gate; there are no before/after split measurements.",
        ]
    for hand in split_hands:
        for policy in ("current", "average"):
            before, after = (
                summaries[("split", hand, phase, policy)]["local_br_gap"]["mean"]
                for phase in ("before", "after")
            )
            lines.append(
                f"| {hand} | {policy} | {before:.4f} | {after:.4f} | {after-before:.4f} |"
            )
    progress = {arm: read_rows(run / arm / "progress.jsonl") for arm in ARMS}
    reviews = read_rows(run / "split/reviews.jsonl")
    reasons = Counter(row["reason"] for row in reviews)
    confirmed = [row for row in reviews if "validation_gain" in row]
    if confirmed:
        lines += [
            "",
            f"Of {len(confirmed)} supported confirmation panels, conflicting action directions reproduced in {sum(r['conflict_reproduced'] for r in confirmed)} and estimated gain was positive in {sum(r['validation_gain'] > 0 for r in confirmed)}. Positive lower gates: {sum(r['lower_gate'] > 0 for r in confirmed)}. These are sampled-regret diagnostics, not Local BR-gap measurements.",
        ]
    lines += [
        "",
        "## Refinement and Cost",
        "",
        f"Completed reviews: {len(reviews)}. Gate outcomes: `{dict(reasons)}`.",
        "",
        "| Arm | Splits | Decisions | Occupied keys | Train+review seconds | Table MiB | Peak monitor payload MiB | Final model MiB | Resume MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, history in progress.items():
        row = history[-1]
        model = run / arm / f"checkpoint_{config['hands']:07d}.npz"
        lines.append(
            f"| {arm} | {row['splits']} | {row['decisions']} | {row['occupied_keys']} | {row['seconds']:.1f} | {row['allocated_table_bytes']/2**20:.3f} | {max(v['monitor_array_bytes'] for v in history)/2**20:.3f} | {model.stat().st_size/2**20:.3f} | {row['checkpoint_bytes']/2**20:.3f} |"
        )
    lines += [
        "",
        "Table bytes are allocated regret/average/visit arrays; monitor payload excludes Python-object overhead, encoder and worker RSS. Equal reserved capacity is not equal active leaf count. Final model MiB counts only the policy-table/routing NPZ, not the shared frozen encoder/atlases, which must be retained. Model snapshots omit monitor replay; last.npz retains it for interruption resume.",
        f"Evaluation totals: {sum(r['rollouts'] for r in rows):,} rollouts; {sum(r['policy_queries'] for r in rows):,} policy queries; {sum(r['missing'] for r in rows)} missing partitions.",
        "",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm, history in progress.items():
        axes[0].step(
            [v["hands"] for v in history],
            [v["splits"] for v in history],
            where="post",
            label=LABELS[arm],
            color=COLORS[arm],
        )
        axes[1].plot(
            [v["hands"] for v in history],
            [v["monitor_array_bytes"] / 2**20 for v in history],
            label=LABELS[arm],
            color=COLORS[arm],
        )
    for panel, title, ylabel in zip(
        axes,
        ("Accepted binary splits", "Monitor array payload (not RSS)"),
        ("Splits", "MiB"),
    ):
        panel.set_title(title)
        panel.set_xlabel("Training hands")
        panel.set_ylabel(ylabel)
        panel.grid(alpha=0.2)
        panel.legend(fontsize=8)
    axes[0].set_ylim(
        0, max(1, max(v["splits"] for history in progress.values() for v in history))
    )
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(run / "refinement_cost.png", dpi=150)
    plt.close(figure)
    lines += ["![Refinement cost](refinement_cost.png)", ""]
    (run / "summary.json").write_text(
        json.dumps(
            dict(
                groups=[
                    dict(arm=k[0], hands=k[1], phase=k[2], policy=k[3], **v)
                    for k, v in summaries.items()
                ],
                paired=pairs,
                review_reasons=dict(reasons),
                initial_uniform=initial,
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run / "RESULTS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initial-run", type=Path)
    args = parser.parse_args()
    report(args.run, args.initial_run)
