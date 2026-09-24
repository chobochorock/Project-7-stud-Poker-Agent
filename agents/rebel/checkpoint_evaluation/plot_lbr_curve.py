from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: row["training_hands"])
    x = [row["training_hands"] for row in rows]
    y = [row["lbr_ante_per_hand"] for row in rows]
    lower = [row["ci95"][0] for row in rows]
    upper = [row["ci95"][1] for row in rows]
    mae = [row["validation_mae_ante"] for row in rows]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(x, y, marker="o", color="#c44e52", label="LBR lower bound")
    top.fill_between(x, lower, upper, color="#c44e52", alpha=0.18, label="95% CI")
    top.set_ylabel("LBR profit (ante / hand)")
    top.set_title("7-Stud partial recursive ReBeL checkpoint evaluation")
    top.grid(alpha=0.25)
    top.legend()

    bottom.plot(x, mae, marker="o", color="#4c72b0")
    bottom.set_xlabel("self-play attempts used for V7 data")
    bottom.set_ylabel("held-out V7 MAE (ante)")
    bottom.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    print(json.dumps({"output": str(args.output.resolve()), "points": len(rows)}))


if __name__ == "__main__":
    main()
