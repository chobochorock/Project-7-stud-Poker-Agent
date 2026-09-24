"""Evaluate saved Deep CFR checkpoints and draw a compact training graph."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)


def evaluate(model: Path, args: argparse.Namespace, opponent: str) -> dict:
    command = [
        sys.executable,
        "-B",
        str(ROOT / "agents/deep_cfr/evaluate_deep_cfr_7th.py"),
        "--model",
        str(model),
        "--start-street",
        "5",
        "--hands",
        str(args.lbr_hands if opponent == "policy-lbr" else args.hands),
        "--opponent",
        opponent,
        "--belief-particles",
        str(args.particles),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--port",
        str(args.port),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def trajectory_gap(entry: dict) -> tuple[float | None, list[float] | None]:
    means = [player.get("trajectory_br_gap_mean_ante") for player in entry["players"]]
    errors = [
        player.get("trajectory_br_gap_standard_error_ante")
        for player in entry["players"]
    ]
    if any(value is None for value in means + errors):
        return None, None
    mean = 0.5 * sum(means)
    standard_error = 0.5 * math.sqrt(sum(value * value for value in errors))
    return mean, [
        max(0.0, mean - 1.96 * standard_error),
        mean + 1.96 * standard_error,
    ]


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box,
    rows,
    key,
    title,
    color,
    *,
    log_y=False,
    ci_key=None,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="#9aa0a6", width=1)
    draw.text((left + 8, top + 6), title, fill="#202124")
    points = [(row["iteration"], row[key]) for row in rows if row.get(key) is not None]
    if not points:
        draw.text((left + 8, top + 28), "no data", fill="#777777")
        return
    xs, raw_ys = zip(*points)
    ys = tuple(math.log10(max(1e-9, value)) for value in raw_ys) if log_y else raw_ys
    bounds = list(ys)
    if ci_key:
        for row in rows:
            if row.get(key) is None or not row.get(ci_key):
                continue
            bounds.extend(
                math.log10(max(1e-9, value)) if log_y else value
                for value in row[ci_key]
            )
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(bounds), max(bounds)
    if y0 == y1:
        y0 -= 0.5
        y1 += 0.5
    pad_x, pad_y = 42, 28
    plot = (left + pad_x, top + pad_y, right - 12, bottom - 24)
    px0, py0, px1, py1 = plot
    draw.line((px0, py1, px1, py1), fill="#666666")
    draw.line((px0, py0, px0, py1), fill="#666666")
    mapped = []
    for (x, _), y in zip(points, ys):
        px = (px0 + px1) / 2 if x0 == x1 else px0 + (x - x0) / (x1 - x0) * (px1 - px0)
        py = py1 - (y - y0) / (y1 - y0) * (py1 - py0)
        mapped.append((px, py))
    if len(mapped) > 1:
        draw.line(mapped, fill=color, width=3)
    for ((_, raw_y), point, row) in zip(
        points, mapped, [r for r in rows if r.get(key) is not None]
    ):
        x, y = point
        if ci_key and row.get(ci_key):
            low, high = row[ci_key]
            converted_low = math.log10(max(1e-9, low)) if log_y else low
            converted_high = math.log10(max(1e-9, high)) if log_y else high
            error_top = py1 - (converted_high - y0) / (y1 - y0) * (py1 - py0)
            error_bottom = py1 - (converted_low - y0) / (y1 - y0) * (py1 - py0)
            draw.line((x, error_top, x, error_bottom), fill="#777777", width=1)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text((px0, py1 + 5), str(x0), fill="#555555")
    draw.text((px1 - 25, py1 + 5), str(x1), fill="#555555")
    label_y1 = 10**y1 if log_y else y1
    label_y0 = 10**y0 if log_y else y0
    draw.text((left + 3, py0), f"{label_y1:.3g}", fill="#555555")
    draw.text((left + 3, py1 - 10), f"{label_y0:.3g}", fill="#555555")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=0)
    parser.add_argument("--lbr-hands", type=int, default=200)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--lbr-every", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=37001)
    parser.add_argument("--port", type=int, default=28731)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    history = {}
    history_path = args.run_dir / "history.jsonl"
    for line in history_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        history[int(entry["iteration"])] = trajectory_gap(entry)
    checkpoints = []
    for path in args.run_dir.glob("policy_i*.pt"):
        match = re.fullmatch(r"policy_i(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise RuntimeError("no policy_i*.pt checkpoints found")

    output_path = args.run_dir / "evaluations.json"
    rows = (
        []
        if args.force or not output_path.exists()
        else json.loads(output_path.read_text(encoding="utf-8"))
    )
    cached = {row["iteration"]: row for row in rows}
    final_iteration = checkpoints[-1][0]
    for iteration, model in checkpoints:
        row = cached.get(iteration, {"iteration": iteration})
        gap, gap_ci = history.get(iteration, (None, None))
        row["trajectory_br_gap"] = gap
        row["trajectory_br_gap_ci95"] = gap_ci
        if args.hands > 0 and (args.force or row.get("heuristic_profit") is None):
            result = evaluate(model, args, "heuristic")
            row["heuristic_profit"] = result["average_profit_ante_for_a"]
            row["heuristic_ci95"] = result["ci95_ante_for_a"]
        if args.lbr_every > 0 and (
            iteration % args.lbr_every == 0 or iteration == final_iteration
        ):
            if args.force or row.get("lbr_lower_bound") is None:
                result = evaluate(model, args, "policy-lbr")
                row["lbr_lower_bound"] = result[
                    "approx_exploitability_lower_bound_ante"
                ]
                row["lbr_ci95"] = result["ci95_ante_for_lbr"]
        cached[iteration] = row
        output_path.write_text(
            json.dumps(
                sorted(cached.values(), key=lambda item: item["iteration"]), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)
    rows = sorted(cached.values(), key=lambda item: item["iteration"])

    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Corrected Deep CFR training history", fill="#111111")
    draw_panel(
        draw,
        (20, 45, 1180, 310),
        rows,
        "heuristic_profit",
        "Heuristic profit (higher is better)",
        "#167d50",
        ci_key="heuristic_ci95",
    )
    draw_panel(
        draw,
        (20, 325, 1180, 590),
        rows,
        "lbr_lower_bound",
        "Policy-LBR lower bound (lower is better; 95% CI bars)",
        "#b3261e",
        ci_key="lbr_ci95",
    )
    draw_panel(
        draw,
        (20, 605, 1180, 870),
        rows,
        "trajectory_br_gap",
        "Trajectory-relaxed BR-gap proxy, log scale (lower is better)",
        "#1a73e8",
        log_y=True,
        ci_key="trajectory_br_gap_ci95",
    )
    graph = args.run_dir / "training_history.png"
    image.save(graph)
    print(json.dumps({"evaluations": str(output_path), "graph": str(graph)}))


if __name__ == "__main__":
    main()
