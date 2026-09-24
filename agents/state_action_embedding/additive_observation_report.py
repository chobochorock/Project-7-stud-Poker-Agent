"""Plot completed cumulative-action experiments without importing PyTorch.

Usage: python -m agents.state_action_embedding.additive_observation_report --out-dir RUN
Writes comparison.png and summary.json; source training metrics remain untouched.
"""

import argparse
import json

from .action_path_report import Path, np, plt, read


MODES = ("sum", "sum_pe", "rotary_sum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(args.out_dir.glob("seed*"))
    if not runs:
        raise ValueError("No seed directories")
    results = [
        {mode: read(run / f"{mode}_metrics.json") for mode in MODES} for run in runs
    ]
    audits = [read(run / "algebra_audit.json") for run in runs]
    baselines = [read(run / "probe_baselines.json") for run in runs]

    def stats(values):
        return dict(
            mean=float(np.mean(values)),
            sd=float(np.std(values, ddof=int(len(values) > 1))),
            values=values,
        )

    summary = dict(seeds=[int(run.name[4:]) for run in runs], modes={}, baselines={})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ("#187c98", "#b13b60", "#7e8731")
    labels = ("Sum", "Sum + PE", "Rotated sum")
    for index, (mode, label, color) in enumerate(zip(MODES, labels, colors)):
        records = [result[mode] for result in results]
        entry = {
            "val_mse": stats([r["final"]["val"]["standardized_mse"] for r in records]),
            "improvement_over_mean": stats(
                [r["final"]["val"]["improvement_over_mean"] for r in records]
            ),
            "transition_mse": stats(
                [r["final"]["val_transition_relative_mse"] for r in records]
            ),
            "endpoint_relative_l2": stats(
                [r[mode]["same_endpoint_relative_l2_mean"] for r in audits]
            ),
            "endpoint_equal_fraction": stats(
                [r[mode]["same_endpoint_equal_fraction"] for r in audits]
            ),
            "reversal_relative_l2": stats(
                [r[mode]["reversal_relative_l2_mean"] for r in audits]
            ),
        }
        for probe in ("sum_probe", "student_probe"):
            entry[probe] = {
                metric: stats([r[probe][metric] for r in records])
                for metric in ("mean_card_recall", "chip_mae")
            }
        summary["modes"][mode] = entry
        steps = [row["step"] for row in records[0]["curve"]]
        curves = np.array(
            [[row["val"]["standardized_mse"] for row in r["curve"]] for r in records]
        )
        mean, sd = curves.mean(0), curves.std(0, ddof=int(len(runs) > 1))
        axes[0].plot(steps, mean, color=color, label=label)
        axes[0].fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.15)
        for j, (probe, name) in enumerate(
            (
                ("sum_probe", "Action-prefix vector"),
                ("student_probe", "Current-observation MLP"),
            )
        ):
            score = entry[probe]["mean_card_recall"]
            axes[1].bar(
                index + (j - 0.5) * 0.35,
                score["mean"] * 100,
                0.35,
                yerr=score["sd"] * 100,
                color=colors[j],
                label=name if index == 0 else None,
            )
        score = entry["endpoint_relative_l2"]
        axes[2].bar(index, score["mean"], yerr=score["sd"], color=color)
    for name, color, label in (
        ("random_sum", "#333333", "Random action sum"),
        ("street_length", "#888888", "Street + prefix length"),
    ):
        summary["baselines"][name] = {
            metric: stats([r[name][metric] for r in baselines])
            for metric in ("mean_card_recall", "chip_mae")
        }
        axes[1].axhline(
            summary["baselines"][name]["mean_card_recall"]["mean"] * 100,
            color=color,
            linestyle="--",
            label=label,
        )
    axes[0].set(
        title="A. Fit to cumulative pseudo-targets",
        xlabel="MLP optimizer updates",
        ylabel="Validation standardized MSE (lower)",
    )
    axes[0].legend(fontsize=9)
    axes[0].text(
        0.04,
        0.95,
        "Different targets: not a quality ranking",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].set(
        title="B. Visible-card linear probe",
        ylabel="Recall @ known card count (%)",
        ylim=(0, 60),
    )
    axes[1].set_xticks(range(3), labels)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].set(
        title="C. Same endpoints, unequal paths",
        ylabel="Mean relative L2 difference (zero required)",
        ylim=(0, 0.7),
    )
    axes[2].set_xticks(range(3), labels)
    axes[2].text(
        0.03,
        0.96,
        "8,192 legal pairs per seed\nReused diagnostic, not a fresh test",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Frozen 32D action SGNS -> observation sums | seeds 11, 22, 33; bars/bands = seed SD",
        fontsize=12,
    )
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)
    summary["resources"] = [read(run / "resources.json") for run in runs]
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
