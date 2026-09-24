"""Compare longer imitation, longer PPO, and a larger event Transformer.

Usage (project root, existing Python 3.12 runtime for training):
    python -m agents.state_action_embedding.bc_ppo_scaling run --out-dir RUN
    python -m agents.state_action_embedding.bc_ppo_scaling run --out-dir SMOKE --smoke
    python -m agents.state_action_embedding.bc_ppo_scaling report --out-dir RUN

Run starts at most three worker processes, with hidden windows on Windows.
Workers reuse bc_ppo; report needs only NumPy/Matplotlib (ordinary Python works).
Input: unchanged pilot demonstrations and pilot checkpoints. Output: NEW run only.
Small-model PPO at 128/512 updates shares one training path; snapshots preserve
the best-so-far policy at each budget. Final test decks are new, not pilot tests.
This measures profit against one fixed heuristic, NOT exploitability/general skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "agents/state_action_embedding/data/bc_ppo_20260919"
TEST_BASE = 620_000_000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def worker(args):
    from . import bc_ppo as experiment

    plan = read_json(args.out_dir / "plan.json")
    directory = args.out_dir / args.size
    dim, layers = (64, 2) if args.size == "small" else (128, 4)
    experiment.main(
        [
            "train",
            "--out-dir",
            str(directory),
            "--seed",
            str(args.seed),
            "--demonstrations",
            str(PILOT / "demonstrations.npz"),
            "--dim",
            str(dim),
            "--layers",
            str(layers),
            "--bc-steps",
            str(plan["bc_steps"]),
            "--bc-eval-every",
            str(plan["bc_eval_every"]),
            "--ppo-updates",
            str(plan["ppo_updates"]),
            "--ppo-eval-every",
            str(plan["ppo_eval_every"]),
            "--ppo-snapshot-updates",
            str(plan["short_update"]),
            "--rollout-hands",
            str(plan["rollout_hands"]),
            "--threads",
            str(plan["threads_per_worker"]),
            "--validation-pairs",
            str(plan["validation_pairs"]),
        ]
    )
    experiment.torch.set_num_threads(1)
    directory = directory / f"seed{args.seed}"
    short = plan["short_update"]
    policies = {
        "bc": directory / "bc.pt",
        "short_best": directory / f"ppo_{short}_best.pt",
        "short_last": directory / f"ppo_{short}_last.pt",
        "long_best": directory / "ppo_best.pt",
        "long_last": directory / "ppo_last.pt",
    }
    if args.size == "small":
        policies.update(
            pilot_bc=PILOT / f"seed{args.seed}/bc.pt",
            pilot_ppo=PILOT / f"seed{args.seed}/ppo_best.pt",
            heuristic=None,
        )
    started = time.perf_counter()
    arrays, metrics = {}, {}
    for name, path in policies.items():
        model, metadata = experiment.load_policy(path) if path else (None, {})
        values = experiment.evaluate_policy(
            model, plan["test_pairs"], TEST_BASE, 830_000_000 + args.seed * 100_000
        )
        arrays[name] = values
        metrics[name] = {
            **experiment.profit_metrics(values),
            "checkpoint": str(path) if path else None,
            "selected_update": metadata.get("selected_update"),
            "selected_step": metadata.get("selected_step"),
        }
        print(f"TEST {args.size} seed={args.seed} {name}: {metrics[name]}", flush=True)
    if "heuristic" in arrays:
        experiment.np.testing.assert_array_equal(arrays["heuristic"].sum(1), 0)
    experiment.np.savez_compressed(directory / "scaling_evaluation.npz", **arrays)
    write_json(
        directory / "scaling_evaluation.json",
        {
            "seed": args.seed,
            "size": args.size,
            "deal_seed_base": TEST_BASE,
            "metrics": metrics,
            "seconds": time.perf_counter() - started,
            "source_sha256": experiment.source_hashes(),
            "decoding": "stochastic categorical for ALL learned policies",
        },
    )


def run(args):
    if args.out_dir.exists():
        raise FileExistsError("Use a new output directory; existing runs are immutable")
    if not (PILOT / "demonstrations.npz").is_file():
        raise FileNotFoundError(PILOT / "demonstrations.npz")
    args.out_dir.mkdir(parents=True)
    plan = {
        "seeds": [11] if args.smoke else [11, 22, 33],
        "bc_steps": 2 if args.smoke else 20000,
        "bc_eval_every": 1 if args.smoke else 500,
        "ppo_updates": 2 if args.smoke else 512,
        "short_update": 1 if args.smoke else 128,
        "ppo_eval_every": 1 if args.smoke else 16,
        "rollout_hands": 8 if args.smoke else 128,
        "validation_pairs": 4 if args.smoke else 256,
        "test_pairs": 4 if args.smoke else 4000,
        "threads_per_worker": args.threads,
        "workers": args.workers,
        "test_deal_base": TEST_BASE,
        "demonstrations_sha256": hashlib.sha256(
            (PILOT / "demonstrations.npz").read_bytes()
        ).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Fixed heuristic, paired held-out decks, validation selection only",
        "smoke": args.smoke,
    }
    write_json(args.out_dir / "plan.json", plan)
    for size in ("small", "large"):
        (args.out_dir / size).mkdir()
    # Interleave sizes so long-running large models do not all wait for small runs.
    pending = [(size, seed) for seed in plan["seeds"] for size in ("large", "small")]
    active, completed = [], []
    started, last_status = time.perf_counter(), 0.0
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                size, seed = pending.pop(0)
                log = (args.out_dir / f"{size}_seed{seed}.log").open(
                    "x", encoding="utf-8"
                )
                command = [
                    sys.executable,
                    "-m",
                    "agents.state_action_embedding.bc_ppo_scaling",
                    "worker",
                    "--out-dir",
                    str(args.out_dir),
                    "--size",
                    size,
                    "--seed",
                    str(seed),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                active.append((size, seed, process, log))
                print(f"Started {size} seed={seed} pid={process.pid}", flush=True)
            for item in active[:]:
                size, seed, process, log = item
                if process.poll() is not None:
                    log.close()
                    if process.returncode:
                        raise RuntimeError(f"{size} seed={seed} failed; see its log")
                    active.remove(item)
                    completed.append([size, seed])
                    print(f"Completed {size} seed={seed}", flush=True)
            now = time.perf_counter()
            if now - last_status >= 30:
                status = []
                for size, seed, process, log in active:
                    path = args.out_dir / size / f"seed{seed}/progress.json"
                    try:
                        point = read_json(path)["curve"][-1]
                        status.append(
                            f"{size}/{seed} {point['stage']} {point.get('step', point.get('update'))}"
                        )
                    except (FileNotFoundError, json.JSONDecodeError):
                        status.append(f"{size}/{seed} starting")
                print(
                    f"{now-started:.0f}s | done {len(completed)} | "
                    + "; ".join(status),
                    flush=True,
                )
                last_status = now
            if pending or active:
                time.sleep(2)
    finally:
        for _, _, process, log in active:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
    write_json(
        args.out_dir / "completed.json",
        {"runs": completed, "seconds": time.perf_counter() - started},
    )


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .bc_ppo_report import audit_demonstrations, summarize

    plan = read_json(args.out_dir / "plan.json")
    raw, training, configs = {}, {}, {}
    for size in ("small", "large"):
        training[size], configs[size] = [], []
        arrays = []
        for seed in plan["seeds"]:
            directory = args.out_dir / size / f"seed{seed}"
            evaluation = read_json(directory / "scaling_evaluation.json")
            assert evaluation["deal_seed_base"] == TEST_BASE
            training[size].append(read_json(directory / "training.json"))
            config = read_json(directory / "config.json")
            assert config["demonstrations_sha256"] == plan["demonstrations_sha256"]
            assert config["bc_steps"] == plan["bc_steps"]
            assert config["ppo_updates"] == plan["ppo_updates"]
            configs[size].append(config)
            with np.load(
                directory / "scaling_evaluation.npz", allow_pickle=False
            ) as archive:
                arrays.append(dict(archive))
        for policy in arrays[0]:
            raw[f"{size}_{policy}"] = np.stack([a[policy] for a in arrays])
    for config in configs["small"] + configs["large"]:
        assert config["source_sha256"] == configs["small"][0]["source_sha256"]
    np.testing.assert_array_equal(raw["small_heuristic"].sum(-1), 0)
    contrasts = {
        "long_BC_minus_pilot_BC": ("small_bc", "small_pilot_bc"),
        "A_minus_pilot_PPO": ("small_short_best", "small_pilot_ppo"),
        "B_minus_A": ("small_long_best", "small_short_best"),
        "C_minus_B": ("large_long_best", "small_long_best"),
        "B_minus_A_last": ("small_long_last", "small_short_last"),
        "C_minus_B_last": ("large_long_last", "small_long_last"),
    }
    summary = {
        "plan": plan,
        "policies": {k: summarize(v) for k, v in raw.items()},
        "contrasts": {k: summarize(raw[a] - raw[b]) for k, (a, b) in contrasts.items()},
        "models": {
            size: {
                "parameters": configs[size][0]["parameters"],
                "bc_selected_steps": [t["bc_selected_step"] for t in training[size]],
                "ppo_selected_updates": [
                    t["ppo_selected_update"] for t in training[size]
                ],
                "bc_test": {
                    stage: {
                        metric: float(
                            np.mean(
                                [t["bc_test"][stage][metric] for t in training[size]]
                            )
                        )
                        for metric in ("nll", "accuracy")
                    }
                    for stage in ("all", "bet", "discard_reveal")
                },
                "resources": {
                    phase: {
                        metric: [t[f"{phase}_memory"][metric] for t in training[size]]
                        for metric in ("seconds", "rss_sampled_peak_mib")
                    }
                    for phase in ("bc", "ppo")
                },
                "learner_decisions": [t["learner_decisions"] for t in training[size]],
            }
            for size in training
        },
        "audit": audit_demonstrations(PILOT / "demonstrations.npz"),
        "uncertainty": "Paired-deal bootstrap conditional on these trained models, not all training randomness. Seed means shown separately. No multiplicity adjustment; exploratory comparisons.",
        "runtime_caveat": "Concurrent CPU runs contend for resources; wall times are not isolated speed benchmarks.",
    }
    write_json(args.out_dir / "summary.json", summary)
    labels = ["Pilot PPO", "A: longer BC", "B: longer PPO", "C: larger model"]
    best = ["small_pilot_ppo", "small_short_best", "small_long_best", "large_long_best"]
    last = [None, "small_short_last", "small_long_last", "large_long_last"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for i, key in enumerate(best):
        m = summary["policies"][key]
        mean = m["mean_chips_per_hand"]
        lo, hi = m["paired_deal_bootstrap_95pct"]
        axes[0].errorbar(
            i - 0.08,
            mean,
            yerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
            fmt="o",
            color="#187b69",
            capsize=4,
            label="Validation-selected" if i == 0 else None,
        )
        axes[0].scatter(
            np.full(len(plan["seeds"]), i - 0.08),
            m["per_seed_chips_per_hand"],
            s=12,
            color="#333333",
            zorder=4,
        )
        if last[i]:
            m = summary["policies"][last[i]]
            mean = m["mean_chips_per_hand"]
            lo, hi = m["paired_deal_bootstrap_95pct"]
            axes[0].errorbar(
                i + 0.08,
                mean,
                yerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
                fmt="s",
                color="#b35176",
                capsize=4,
                label="Final budget" if i == 1 else None,
            )
    axes[0].set_xticks(range(4), labels, rotation=15)
    axes[0].set_title("Test profit vs fixed heuristic")
    axes[0].set_ylabel("Net chips / hand")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=8)
    for i, (size, color) in enumerate((("small", "#187b69"), ("large", "#b35176"))):
        for j, t in enumerate(training[size]):
            curve = [p for p in t["curve"] if p["stage"] == "bc"]
            axes[1].plot(
                [p["step"] for p in curve],
                [p["all"]["nll"] for p in curve],
                color=color,
                alpha=0.65,
                label=f"{size} validation" if j == 0 else None,
            )
            axes[1].plot(
                [p["step"] for p in curve],
                [p["train_probe"]["all"]["nll"] for p in curve],
                "--",
                color=color,
                alpha=0.4,
                label=f"{size} train probe" if j == 0 else None,
            )
            ppo = [p for p in t["curve"] if p["stage"] == "ppo" and "validation" in p]
            axes[2].plot(
                [p["train_hands"] for p in ppo],
                [p["validation"]["chips_per_hand"] for p in ppo],
                color=color,
                alpha=0.6,
                label=size if j == 0 else None,
            )
    axes[1].set_title("BC: train / validation NLL")
    axes[1].set_xlabel("BC updates")
    axes[1].legend(fontsize=8)
    axes[2].set_title("PPO: validation profit")
    axes[2].set_xlabel("Fresh training hands")
    axes[2].axvline(
        plan["short_update"] * plan["rollout_hands"], color="#777777", linestyle="--"
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"BC / PPO scaling | {len(plan['seeds'])} seeds | {plan['test_pairs']*2} new seat-paired test hands per policy/seed\n95% paired-deal bootstrap; black dots = training seeds; lower NLL is better",
        fontsize=12,
    )
    fig.savefig(args.out_dir / "comparison.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "policies": {k: summary["policies"][k] for k in best},
                "contrasts": summary["contrasts"],
                "models": summary["models"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "report"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--size", choices=("small", "large"), default="small")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.workers < 1 or args.workers > 6 or args.threads < 1:
        parser.error("Need 1..6 workers and positive threads")
    {"run": run, "worker": worker, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
