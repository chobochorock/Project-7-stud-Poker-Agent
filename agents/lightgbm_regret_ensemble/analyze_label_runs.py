"""Compare two completed fresh 7-stud label runs, not offline teacher fits.

Usage: python agents/lightgbm_regret_ensemble/analyze_label_runs.py RUN_PAIR_DIR
Input: refined/ and cross_entropy/ runs, each with independent_audit.csv.
Output: comparison.png, comparison_log.png, comparison.json in RUN_PAIR_DIR.
Metric: first-5th-infoset rollout Local BR-gap in antes, NOT exploitability.
Intervals cover held-out root sampling only, not training-seed or particle bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_epoch_ensemble import analyze
from run_epoch_ensemble import np, plt


def estimate(values: np.ndarray | list[float]) -> dict[str, float | list[float]]:
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, se=se, ci95=[mean - 1.96 * se, mean + 1.96 * se])


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    assert estimate([1, 3])["mean"] == 2 and estimate([1, 3])["se"] == 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--roots", type=int, default=512)
    args = parser.parse_args()
    directory = args.directory.resolve()
    modes = ["refined", "cross_entropy"]
    data, config, reports, independent = {}, {}, {}, {}
    for mode in modes:
        run = directory / mode
        analyze(run)
        config[mode] = json.loads((run / "config.json").read_text(encoding="utf-8"))
        reports[mode] = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
        assert reports[mode]["complete"] and config[mode]["label_mode"] == mode
        data[mode] = np.atleast_1d(
            np.genfromtxt(run / "per_hand.csv", delimiter=",", names=True)
        )
        independent[mode] = np.atleast_1d(
            np.genfromtxt(run / "independent_audit.csv", delimiter=",", names=True)
        )
        audit = independent[mode]
        assert len(audit) == args.roots and args.roots > 1
        assert np.array_equal(audit["root"], np.arange(args.roots))
        assert all(np.isfinite(audit[field]).all() for field in audit.dtype.names)
        assert len(np.unique(audit["seed"])) == len(np.unique(audit["particles"])) == 1
    a, b = modes
    for key in (
        "hands",
        "epoch_hands",
        "particles",
        "seed",
        "tree_budget",
        "atlas",
        "binary_sha256",
        "source_sha256",
    ):
        assert config[a][key] == config[b][key], key
    for key in (
        "hard256_root_local_gap_ante",
        "hard256_training_nodes",
        "hard256_buckets",
    ):
        assert np.array_equal(data[a][key], data[b][key]), key
    for key in ("seed", "particles", "hard256_current", "hard256_average", "uniform"):
        assert np.array_equal(independent[a][key], independent[b][key]), key
    first_epoch = config[a]["epoch_hands"]
    first_hash = {
        mode: file_hash(directory / mode / f"rows_{first_epoch}.bin") for mode in modes
    }
    assert (
        first_hash[a] == first_hash[b]
    ), "initial data must match before first abstraction"
    assert np.array_equal(
        data[a]["ensemble_root_local_gap_ante"][:first_epoch],
        data[b]["ensemble_root_local_gap_ante"][:first_epoch],
    )
    names = [
        "Refined current",
        "Soft CE current",
        "Hard-256 current",
        "Hard-256 average",
        "Uniform",
    ]
    samples = [independent[a]["ensemble_current"], independent[b]["ensemble_current"]]
    samples += [
        independent[a][k] for k in ("hard256_current", "hard256_average", "uniform")
    ]
    result = dict(
        complete=True,
        hands_per_method=config[a]["hands"],
        epoch_hands=first_epoch,
        training_seed=config[a]["seed"],
        evaluation_seed=int(independent[a]["seed"][0]),
        roots=args.roots,
        particles=int(independent[a]["particles"][0]),
        metric="current-policy first-5th-infoset rollout gap in antes; not exploitability or LBR",
        interval_scope="pointwise 95% root-sampling normal approximation; one training seed; excludes particle/max bias",
        analysis_sha256=file_hash(Path(__file__).resolve()),
        evaluation_binary_sha256=file_hash(
            Path(__file__).resolve().parent / "bin/evaluate_epoch_ensemble.exe"
        ),
        first_epoch_data_sha256=first_hash[a],
        baseline_identical=True,
        coverage={
            mode: {
                "roots_without_any_bucket": int(
                    np.sum(independent[mode]["matched_epoch_models"] == 0)
                ),
                "roots_matching_all_epochs": int(
                    np.sum(
                        independent[mode]["matched_epoch_models"]
                        == config[mode]["hands"] // first_epoch
                    )
                ),
            }
            for mode in modes
        },
        heldout={name: estimate(values) for name, values in zip(names, samples)},
        paired_soft_minus_refined=estimate(samples[1] - samples[0]),
        resources={
            mode: {
                key: reports[mode][key]
                for key in (
                    "ensemble_model_bytes",
                    "ensemble_buckets_total",
                    "saved_row_bytes",
                    "combined_cpp_peak_rss_gib",
                    "training_nodes",
                    "fit_seconds",
                )
            }
            for mode in modes
        },
        training_config=config,
    )
    for mode in modes:
        coverage_path = directory / mode / "coverage.csv"
        if coverage_path.exists():
            coverage = np.atleast_1d(
                np.genfromtxt(coverage_path, delimiter=",", names=True)
            )
            assert np.array_equal(coverage["root"], independent[mode]["root"])
            assert np.array_equal(
                coverage["matched_models"], independent[mode]["matched_epoch_models"]
            )
            assert np.array_equal(
                coverage["policy_found"], coverage["matched_models"] > 0
            )
            result["coverage"][mode]["by_actor"] = {
                str(actor): {
                    "roots": int(np.sum(coverage["actor"] == actor)),
                    "unmatched": int(
                        np.sum(
                            (coverage["actor"] == actor)
                            & (coverage["policy_found"] == 0)
                        )
                    ),
                }
                for actor in (0, 1)
            }
        result["resources"][mode]["hard256_model_bytes"] = (
            (directory / mode / f"hard256_{config[mode]['hands']}.bin").stat().st_size
        )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    colors = ["#167e87", "#b63c58", "#626c78"]
    block = min(500, config[a]["hands"])
    for index, (mode, prefix, label) in enumerate(
        [
            (a, "ensemble", "Refined"),
            (b, "ensemble", "Soft CE"),
            (a, "hard256", "Hard-256 current"),
        ]
    ):
        rows = data[mode]
        count = len(rows) // block
        gap = (
            rows[f"{prefix}_root_local_gap_ante"][: count * block]
            .reshape(count, block)
            .mean(axis=1)
        )
        x = rows["hand"][: count * block].reshape(count, block).mean(axis=1)
        axes[0, 0].plot(x, gap, label=label, color=colors[index])
        post = [
            r
            for r in reports[mode]["boundaries"]
            if r["hand"] == r["models"] * first_epoch
        ]
        axes[0, 1].plot(
            [r["hand"] for r in post],
            [r[prefix] for r in post],
            "o-",
            label=label,
            color=colors[index],
        )
    axes[0, 0].set_title(
        f"Every-hand evaluation ({block}-hand means; {config[a]['particles']} particles)"
    )
    axes[0, 1].set_title("Fixed 128-root audit after abstraction; 128 particles")
    for ax in axes[0]:
        ax.set(xlabel="Training hands", ylabel="Estimated one-step gap (ante)")
        ax.grid(alpha=0.2)
        ax.legend()
    values = [result["heldout"][name] for name in names]
    axes[1, 0].errorbar(
        [r["mean"] for r in values],
        range(len(names)),
        xerr=[1.96 * r["se"] for r in values],
        fmt="o",
        capsize=4,
        color="#167e87",
    )
    axes[1, 0].set_yticks(range(len(names)), names)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(
        title=f"Held-out {args.roots} roots: mean and 95% interval",
        xlabel="Estimated one-step gap (ante)",
    )
    axes[1, 0].grid(axis="x", alpha=0.2)
    sizes = [
        result["resources"][mode]["ensemble_model_bytes"] / 2**20 for mode in modes
    ]
    sizes.append(result["resources"][a]["hard256_model_bytes"] / 2**20)
    axes[1, 1].bar(["Refined", "Soft CE", "Hard-256"], sizes, color=colors)
    axes[1, 1].set(title="Final inference exports, all epochs (not RAM)", ylabel="MiB")
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Fresh 7-stud self-play: {config[a]['hands']:,} hands each, {first_epoch:,}-hand epochs, seed {config[a]['seed']}\n"
        "Caution: about half of held-out roots have no stored ensemble bucket; not a game-strength ranking",
        fontsize=12,
    )
    fig.savefig(directory / "comparison.png", dpi=160)
    for ax in axes[0]:
        ax.set_yscale("log", nonpositive="mask")
        ax.set_ylabel("Estimated one-step gap (ante, log scale)")
    fig.savefig(directory / "comparison_log.png", dpi=160)
    plt.close(fig)
    (directory / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("training_config", "resources")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
