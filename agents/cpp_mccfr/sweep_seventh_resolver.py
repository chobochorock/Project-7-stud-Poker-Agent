import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_final_json(output: str) -> dict:
    marker = '\n{\n  "agent_a"'
    start = output.rfind(marker)
    if start < 0:
        raise RuntimeError("final evaluation JSON was not found")
    return json.loads(output[start + 1 :])


def save_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if not rows:
        return
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], output_dir: Path) -> None:
    baseline = next(row for row in rows if row["iterations"] == 0)
    resolved = [row for row in rows if row["iterations"] > 0]
    iterations = sorted({row["iterations"] for row in resolved})
    priors = sorted({row["prior"] for row in resolved})

    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_curve.axhspan(
        baseline["ci95_low"], baseline["ci95_high"], color="0.8", alpha=0.5
    )
    ax_curve.axhline(
        baseline["lbr"],
        color="black",
        linestyle="--",
        label=f'30M baseline ({baseline["lbr"]:.3f})',
    )
    for prior in priors:
        points = sorted(
            (row for row in resolved if row["prior"] == prior),
            key=lambda row: row["iterations"],
        )
        x = [row["iterations"] for row in points]
        y = [row["lbr"] for row in points]
        yerr = [1.96 * row["se"] for row in points]
        ax_curve.errorbar(
            x, y, yerr=yerr, marker="o", capsize=3, label=f"prior={prior}"
        )
    ax_curve.set_title("7th-street resolver sweep")
    ax_curve.set_xlabel("CFR+ iterations per reached subgame")
    ax_curve.set_ylabel("LBR profit (ante/hand, lower is better)")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend()

    matrix = np.full((len(priors), len(iterations)), np.nan)
    for row in resolved:
        matrix[priors.index(row["prior"]), iterations.index(row["iterations"])] = row[
            "lbr"
        ]
    image = ax_heat.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
    ax_heat.set_xticks(range(len(iterations)), iterations)
    ax_heat.set_yticks(range(len(priors)), priors)
    ax_heat.set_xlabel("Iterations")
    ax_heat.set_ylabel("Prior strength")
    ax_heat.set_title("LBR lower bound")
    for y in range(len(priors)):
        for x in range(len(iterations)):
            ax_heat.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax_heat, label="ante/hand")
    fig.tight_layout()
    fig.savefig(output_dir / "seventh_resolver_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--particles", type=int, default=240)
    parser.add_argument("--seed", type=int, default=46103)
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[32, 64, 100, 160, 256]
    )
    parser.add_argument("--priors", type=float, nargs="+", default=[10, 100, 1000])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agents/cpp_mccfr/data/results/seventh_resolver_sweep_10k240"),
    )
    args = parser.parse_args()

    executable = Path("agents/cpp_mccfr/bin/stud_mccfr_seventh_resolver.exe")
    common = [
        str(executable),
        "--bucket",
        "power-memory16",
        "--load-atlas",
        "agents/cpp_mccfr/data/power512_epsheur20_memory16_v1.bin",
        "--start-street",
        "5",
        "--algorithm",
        "mccfr",
        "--ante",
        "1000",
        "--load",
        "agents/cpp_mccfr/data/made_call_r1000_k512_epsheur20_memory16_30m.bin",
        "--seventh-hand-history",
        "--hands",
        str(args.hands),
        "--iterations",
        "0",
        "--opponent",
        "policy-lbr",
        "--belief-particles",
        str(args.particles),
        "--seed",
        str(args.seed),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    rows = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else []
    )
    completed = {(row["iterations"], float(row["prior"])) for row in rows}
    jobs = [(0, 0.0)] + [
        (iteration, prior) for prior in args.priors for iteration in args.iterations
    ]

    for index, (iteration, prior) in enumerate(jobs, 1):
        if (iteration, float(prior)) in completed:
            print(
                f"skip {index}/{len(jobs)}: iterations={iteration}, prior={prior}",
                flush=True,
            )
            continue
        command = common.copy()
        if iteration > 0:
            command += [
                "--seventh-resolve-iterations",
                str(iteration),
                "--seventh-resolve-prior",
                str(prior),
            ]
        print(
            f"run {index}/{len(jobs)}: iterations={iteration}, prior={prior}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if "evaluation-heartbeat" in line:
                print(line.rstrip(), flush=True)
        if process.wait() != 0:
            raise RuntimeError(
                f"evaluation failed: iterations={iteration}, prior={prior}"
            )
        output = "".join(lines)
        raw_name = f"iter{iteration}_prior{prior:g}.log"
        (args.output_dir / raw_name).write_text(output, encoding="utf-8")
        result = parse_final_json(output)
        rows.append(
            {
                "iterations": iteration,
                "prior": float(prior),
                "lbr": result["average_profit_ante_for_lbr"],
                "se": result["paired_standard_error_ante"],
                "ci95_low": result["ci95_ante_for_lbr"][0],
                "ci95_high": result["ci95_ante_for_lbr"][1],
                "elapsed_seconds": result["elapsed_seconds"],
                "subgames": result.get("seventh_resolver", {}).get("subgames", 0),
                "node_visits": result.get("seventh_resolver", {}).get("node_visits", 0),
                "blueprint_fallbacks": result.get("seventh_resolver", {}).get(
                    "blueprint_fallbacks", 0
                ),
            }
        )
        rows.sort(key=lambda row: (row["prior"], row["iterations"]))
        save_results(rows, args.output_dir)
        print(f'result: LBR={result["average_profit_ante_for_lbr"]:.6f}', flush=True)

    plot_results(rows, args.output_dir)
    print(args.output_dir / "seventh_resolver_sweep.png")


if __name__ == "__main__":
    main()
