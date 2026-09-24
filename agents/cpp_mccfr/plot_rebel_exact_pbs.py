import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("metrics CSV is empty")

    iterations = [int(row["iterations"]) for row in rows]
    regret = [float(row["regret_bound_proxy"]) for row in rows]
    average = [float(row["average_positive_regret_mean"]) for row in rows]
    infosets = [int(row["infosets"]) for row in rows]

    def log_slope(values: list[float]) -> float:
        valid = [(x, y) for x, y in zip(iterations, values) if x > 0 and y > 0]
        if len(valid) < 2:
            return float("nan")
        xs = [math.log(x) for x, _ in valid]
        ys = [math.log(y) for _, y in valid]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        return (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
            if denominator
            else float("nan")
        )

    slope = log_slope(regret)
    average_slope = log_slope(average)
    revisited = [float(row.get("revisited_infoset_fraction", 0)) for row in rows]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(iterations, regret, marker="o", label="external-regret bound proxy")
    top.plot(iterations, average, marker=".", label="mean positive regret / T")
    top.set_yscale("log")
    top.set_ylabel("ante / iteration (log)")
    top.grid(alpha=0.25)
    top.legend()
    if math.isfinite(slope) and math.isfinite(average_slope):
        top.set_title(
            f"Exact-PBS local CFR: decomposition {slope:.3f}, mean {average_slope:.3f}"
        )
    else:
        top.set_title("Exact-PBS local CFR")

    bottom.plot(iterations, infosets, color="#c44e52", marker="o", label="infosets")
    bottom.set_xlabel("completed CFR iterations")
    bottom.set_ylabel("exact information sets")
    bottom.grid(alpha=0.25)
    if any(revisited):
        right = bottom.twinx()
        right.plot(
            iterations,
            revisited,
            color="#55a868",
            marker=".",
            label="revisited fraction",
        )
        right.set_ylabel("revisited infoset fraction")
        right.set_ylim(0, max(revisited) * 1.15)
    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(
        f'{{"output":"{args.output}","decomposition_slope":{slope},'
        f'"mean_positive_regret_slope":{average_slope}}}'
    )


if __name__ == "__main__":
    main()
