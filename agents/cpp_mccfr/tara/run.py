"""TARA (tree-accumulated regret abstraction) on C++ 7-stud v3 with a co-trained hard256 k-means arm.

Usage (project root):
    python -m agents.cpp_mccfr.tara.run build --check
    python -m agents.cpp_mccfr.tara.run train --run agents/cpp_mccfr/data/tara/<new> \
        --config agents/cpp_mccfr/tara/configs/tara.json [--hands 127000 --seed 222 ...]
    python -m agents.cpp_mccfr.tara.run lbr --run <run> --output <run>/<new lbr dir> --seeds A B [--avg sum]
Every option can be set in a JSON config (configs/*.json) and overridden on the command line.
Training alternates: C++ trains one interval (writes buffer_<t>.bin at the boundary and exits 3),
Python fits forest t+1 on that interval's buffer (fit.py), C++ resumes with it, and so on.
Outputs: checkpoint.bin, models/model_<hand>.bin at every boundary and the end, per_hand.csv /
panel.csv (one-step Local BR-gap of the first 5th-street decision, same evaluator as
dynamic_tree_validated), forests.csv (per-tree leaves / regret mass), probes.csv (policy at fixed
self-play states 100 hands before, at, and 100 hands after each new tree), fits.jsonl, config.json.
Local BR-gap and Policy-LBR are in ante and are not exploitability. See README.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from .. import hand_recall as legacy
from ..dynamic_tree_validated import run as validated
from . import fit

HERE = Path(__file__).resolve().parent
DATA = legacy.FAMILY / "data" / "tara"
SOURCES = tuple(validated.SOURCES) + (HERE / "solver.cpp", HERE / "lbr.cpp", HERE / "fit.py", Path(__file__))
DEFAULTS = dict(
    hands=127000, seed=222, eval_seed=307, particles=32, panel_every=10000, panel_roots=128,
    checkpoint_every=5000, mode="tara", key_mode="summary", weighting="pow2", prev_scope="all",
    latest_only=False, avg_weight="linear", schedule="doubling", n0=1000, leaf_budget=2048,
    min_samples_leaf=50, partition="learned", reservoir=4096, ehs_samples=64, probe_delta=100, probe_episodes=200,
    clusters=0, b2_buffer="", avg_fallback="current", keep_buffers=False, max_kmeans_states=500000, reserve_gib=10.0,
)
# tara: fitted forests at every boundary. kmeans_sched (B0): the same schedule, weights and per-forest RM+,
# but every forest is the co-trained k-means arm's partition. kmeans_partition (B0-single): one such forest,
# plain RM+ (no boundary discount). fixed (B2): one centroid forest on the TARA features.
FOREST_MODES = ("tara", "kmeans_sched")
CHOICES = dict(mode=("tara", "kmeans_sched", "kmeans_partition", "fixed"), key_mode=("summary", "exact"),
               weighting=("pow2", "halving", "uniform", "index", "index_frozen"), prev_scope=("all", "last"),
               avg_weight=("linear", "same_as_regret"), schedule=("doubling", "fixed"),
               partition=("learned", "random"), avg_fallback=("current", "uniform"))
CODES = dict(key_mode={"exact": 1, "summary": 2},
             weighting={"pow2": 0, "halving": 1, "uniform": 2, "index": 3, "index_frozen": 4},
             avg_weight={"linear": 0, "same_as_regret": 1}, avg_fallback={"uniform": 0, "current": 1})
INTERRUPTED = False


def hashes() -> dict:
    return {str(p.relative_to(legacy.PROJECT)): legacy.sha256(p) for p in SOURCES}


def build(check: bool = False, lbr: bool = False) -> Path:
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    kind = "lbr" if lbr else "solver"
    output = legacy.FAMILY / "bin" / f"tara_{kind}_{validated.identity(hashes())[:16]}.exe"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_suffix(".building.exe")
        subprocess.run([compiler, "-O3", "-std=c++17", "-static-libstdc++", "-static-libgcc",
                        str(HERE / f"{kind}.cpp"), "-o", str(temporary), "-lpsapi"],
                       cwd=legacy.PROJECT, check=True)
        os.replace(temporary, output)
    if check and not lbr:
        subprocess.run([str(output), "--self-test", str(legacy.ATLAS)], check=True)
    return output


def boundaries(cfg: dict) -> list[int]:
    """Forest additions (TARA and B0-sched only); see schedule()."""
    return schedule(cfg) if cfg["mode"] in FOREST_MODES else []


def schedule(cfg: dict) -> list[int]:
    """Interval ends up to hands: doubling N_t = n0 2^(t-1) -> n0 (2^t - 1); fixed -> n0 t.

    Every mode stops (and saves models/model_<hand>.bin) at these hands. A final hand on a boundary also
    prepares the next forest, so a later --hands extension continues the schedule."""
    result, edge, t = [], 0, 1
    while True:
        edge += cfg["n0"] * 2 ** (t - 1) if cfg["schedule"] == "doubling" else cfg["n0"]
        if edge > cfg["hands"]:
            return result
        result.append(edge)
        t += 1


def manifest_for(cfg: dict) -> dict:
    options = {k: cfg[k] for k in DEFAULTS if k not in ("hands", "keep_buffers", "max_kmeans_states", "reserve_gib")}
    atlas = "the co-trained k-means arm's partition: betting summary + seat + hard256 atlas cluster"
    partition_rule = {
        "tara": ("learned: per public node sklearn DecisionTreeRegressor (squared_error) on sampled regret; "
                 "leaf budget by regret mass (D'Hondt, unrealizable leaves reallocated)" if cfg["partition"] == "learned"
                 else "random control: same rows/leaf allocation, random feature and admissible threshold per split"),
        "kmeans_sched": f"every forest (tree 1 included) is {atlas}; added at the TARA boundaries; no fitting",
        "kmeans_partition": f"one forest, {atlas}; no boundaries, no fitting",
        "fixed": "one forest of per-street standardized k-means centroids on the TARA features, fit on b2_buffer",
    }[cfg["mode"]]
    config = dict(
        schema="tara_7stud_1", **options, source_sha256=hashes(), atlas_sha256=legacy.ATLAS_SHA,
        algorithm=("RM+ external-sampling MCCFR; regret keys = leaves of the forests; "
                   "R_j <- max(0, R_j + w_j r) per forest and action; action RM(sum_j R_j); "
                   "average S_j += w_avg sigma at sampled opponent nodes; policy frozen per bucket tuple per traversal; "
                   f"zero average mass -> {cfg['avg_fallback']} policy"),
        budget="one fresh fifth-street deal and two traversals per model per hand",
        rules="C++ stud v3 HU, heuristic H4, ante1000, stack1000ante",
        public_node={"summary": "hard256 baseline betting summary + seat (card cluster removed)",
                     "exact": "seat, street, legal mask and the exact betting history"}[cfg["key_mode"]],
        features="power vector 0-17, equity vs uniform opponent (MC, ehs_samples), dead flush/straight/pair cards",
        partition_rule=partition_rule,
        tree_data=("visits of interval t only (uniform reservoir per public node, one row per visit, "
                   "label = unweighted sampled regret over legal actions); forest t+1 used from interval t+1"
                   if cfg["mode"] == "tara" else "none (no fitting)"),
        baseline="fresh existing hard256 betting-summary partition plus explicit seat (dynamic_tree_validated arm)",
        initialization="zero regrets and average sums; " + (
            "tree 1 is one leaf per public node (card-blind)" if cfg["mode"] == "tara" else "partition fixed from hand 1"),
        metric="post-update fifth-street one-step Local BR-gap, ante, NOT exploitability/LBR",
    )
    return dict(identity=validated.identity(config), config=config)


def load_config(args) -> dict:
    cfg = dict(DEFAULTS)
    if args.resume:  # the saved options win; only --hands (extension) may change
        cfg.update(json.loads((args.run / "options.json").read_text(encoding="utf-8")))
        if args.hands is not None:
            cfg["hands"] = max(cfg["hands"], args.hands)
        return cfg
    if args.config:
        extra = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = set(extra) - set(DEFAULTS) - {"description"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update({k: v for k, v in extra.items() if k != "description"})
    for name in DEFAULTS:
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    for name, allowed in CHOICES.items():
        if cfg[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    if cfg["hands"] < 1 or cfg["n0"] < 1 or cfg["leaf_budget"] < 1 or cfg["min_samples_leaf"] < 1:
        raise ValueError("hands, n0, leaf_budget and min_samples_leaf must be positive")
    if cfg["mode"] == "fixed" and (cfg["clusters"] < 1 or not cfg["b2_buffer"]):
        raise ValueError("mode fixed (B2) needs clusters and b2_buffer")
    if cfg["mode"].startswith("kmeans") and cfg["key_mode"] != "summary":
        raise ValueError("B0 modes reproduce the k-means arm only with key_mode summary")
    if len(boundaries(cfg)) + 1 > 64:
        raise ValueError("at most 64 forests (solver lookup capacity)")
    return cfg


def call(binary: Path, run: Path, manifest: dict, cfg: dict, stop: int, boundary: bool, next_b: int,
         prev_b: int, add: str, initial: str, resume: bool) -> int:
    command = [str(binary), str(run), manifest["identity"], str(stop), str(int(boundary)), str(next_b), str(prev_b),
               add, initial, str(int(resume)), str(cfg["seed"]), str(cfg["eval_seed"]), str(cfg["particles"]),
               str(cfg["panel_every"]), str(cfg["panel_roots"]), str(cfg["checkpoint_every"]),
               str(CODES["key_mode"][cfg["key_mode"]]), str(CODES["weighting"][cfg["weighting"]]),
               str(int(cfg["prev_scope"] == "last")), str(int(cfg["latest_only"])),
               str(CODES["avg_weight"][cfg["avg_weight"]]), str(int(cfg["mode"] == "tara")),
               str(cfg["reservoir"]), str(cfg["ehs_samples"]), str(cfg["probe_delta"]), str(cfg["probe_episodes"]),
               str(cfg["max_kmeans_states"]), str(int(cfg["reserve_gib"] * 2**30)),
               str(CODES["avg_fallback"][cfg["avg_fallback"]])]
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), command=command,
                                  exe_sha256=legacy.sha256(binary))) + "\n")
    with (run / "console.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=legacy.PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                global INTERRUPTED
                INTERRUPTED = True
                print("Finishing the hand and saving. Please wait.", flush=True)
                process.send_signal(signal.CTRL_BREAK_EVENT)
                remaining, _ = process.communicate()
                print(remaining, end="")
                log.write(remaining)
                # A stop during the segment's last hand still completes it (exit 0/3): train() records that.
                return process.returncode


def train(args) -> int:
    cfg = load_config(args)
    run = args.run.resolve()
    if not run.is_relative_to(DATA.resolve()):
        raise ValueError(f"Use a new directory under {DATA}")
    if legacy.sha256(legacy.ATLAS) != legacy.ATLAS_SHA:
        raise ValueError("Frozen atlas changed")
    manifest = manifest_for(cfg)
    state_path = run / "progress.json"
    if args.resume:
        if json.loads((run / "config.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume configuration/source mismatch")
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Nonempty directory: use another run or --resume")
    binary = build()
    run.mkdir(parents=True, exist_ok=True)
    with validated.writer_lock(run):
        if not args.resume:
            shutil.copy2(legacy.ATLAS, run / "hand_atlas.bin")
            shutil.copy2(legacy.ATLAS.with_suffix(".json"), run / "atlas_provenance.json")
            for source in SOURCES:
                target = run / "source" / source.relative_to(legacy.PROJECT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (run / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            if cfg["mode"].startswith("kmeans"):
                fit.write_forest(run / "forest_1.bin", [], kind=2)
                (run / "fits.jsonl").write_text(json.dumps(dict(
                    forest=1, kind="hard256 atlas cluster x public summary key (B0)")) + "\n", encoding="utf-8")
            if cfg["mode"] == "fixed":
                source = Path(cfg["b2_buffer"]).resolve()
                slots = fit.read_buffer(source)
                fit.write_forest(run / "forest_1.bin", [], fit.fit_centroids(slots, cfg["clusters"], cfg["seed"]))
                (run / "fits.jsonl").write_text(json.dumps(dict(
                    forest=1, kind="centroids", clusters=cfg["clusters"], source=str(source),
                    source_sha256=legacy.sha256(source))) + "\n", encoding="utf-8")
            save_state(state_path, dict(hand=0, interval=1, forest=None, pending_fit=None))
        (run / "options.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        adds = cfg["mode"] in FOREST_MODES
        edges = schedule(cfg)
        while True:
            if state["pending_fit"]:  # the C++ side stopped at a boundary (and wrote buffer_<t>.bin for TARA)
                t = state["pending_fit"]
                buffer = run / f"buffer_{t}.bin"
                forest = run / f"forest_{t + 1}.bin"
                started = datetime.now(timezone.utc)
                if cfg["mode"] == "kmeans_sched":
                    records, stats, kind = [], dict(kind="hard256 atlas cluster x public summary key (B0)"), 2
                else:
                    records, stats = fit.fit_forest(fit.read_buffer(buffer), cfg["leaf_budget"],
                                                    cfg["min_samples_leaf"], cfg["seed"] * 1000 + t, cfg["partition"])
                    kind = 0
                fit.write_forest(forest.with_suffix(".tmp"), records, kind=kind)
                os.replace(forest.with_suffix(".tmp"), forest)
                stats["fit_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
                with (run / "fits.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(dict(forest=t + 1, boundary=state["hand"], **stats)) + "\n")
                print(f"FIT forest={t + 1} " + json.dumps(stats), flush=True)
                state.update(interval=t + 1, forest=str(forest), pending_fit=None)
                save_state(state_path, state)
                # buffer_1 is the B2 k-means source; drop the others only after the state is saved
                if buffer.exists() and (t > 1 or cfg["mode"] != "tara") and not cfg["keep_buffers"]:
                    buffer.unlink()
            if state["hand"] >= cfg["hands"] or INTERRUPTED:
                return 2 if INTERRUPTED and state["hand"] < cfg["hands"] else 0
            upcoming = [b for b in edges if b > state["hand"]]
            past = [b for b in edges if b <= state["hand"]]
            stop = upcoming[0] if upcoming else cfg["hands"]
            boundary = adds and bool(upcoming)
            started = (run / "checkpoint.bin").exists()
            initial = str(run / "forest_1.bin") if cfg["mode"] != "tara" and not started else "-"
            code = call(binary, run, manifest, cfg, stop, boundary, upcoming[0] if boundary else 0,
                        past[-1] if adds and past else 0, state["forest"] or "-", initial, started)
            if code not in (0, 3):
                return code
            state.update(hand=stop, forest=None, pending_fit=state["interval"] if code == 3 else None)
            save_state(state_path, state)


def save_state(path: Path, state: dict) -> None:
    path.with_suffix(".tmp").write_text(json.dumps(state), encoding="utf-8")
    os.replace(path.with_suffix(".tmp"), path)


def evaluate_lbr(args) -> None:
    run, output = args.run.resolve(), args.output.resolve()
    manifest = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if manifest["config"]["schema"] != "tara_7stud_1" or manifest["config"]["source_sha256"] != hashes():
        raise ValueError("Model configuration/source mismatch")
    models = sorted((run / "models").glob("model_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    if args.hand:
        models = [m for m in models if int(m.stem.split("_")[1]) == args.hand]
    elif not args.all_models:
        models = models[-1:]
    if not models or output.exists() or not output.is_relative_to(DATA.resolve()):
        raise ValueError("Need an existing model and a new output under the family data folder")
    binary = build(lbr=True)
    output.mkdir(parents=True)
    from ..abstraction_lbr.run import estimate
    import csv
    import numpy as np

    names = ("model_current", "baseline_current", "model_average", "baseline_average", "uniform")
    kind = {"sum": "1", "latest": "2"}[args.avg]
    summary = dict(run=str(run), seeds=args.seeds, pairs_per_seed=args.pairs, particles=args.particles,
                   average_output=args.avg, metric="LBR attacker ante/hand, not exploitability",
                   deal_stream="abstraction_lbr convention (deal seed = CLI seed)", models={})
    with validated.writer_lock(run):
        subprocess.run([str(binary), "--self-test", str(models[-1]), manifest["identity"], kind], check=True)
        for model in models:
            hand = int(model.stem.split("_")[1])
            samples = {name: [] for name in names}
            for seed in args.seeds:
                destination = output / f"hand{hand}" / str(seed)
                destination.parent.mkdir(parents=True, exist_ok=True)
                command = [str(binary), str(model), str(destination), str(args.pairs), str(args.particles), str(seed),
                           manifest["identity"], kind]
                with (output / f"hand{hand}_{seed}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                with (destination / "paired_payoffs.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != args.pairs:
                    raise ValueError("Incomplete LBR output")
                for name in names:
                    samples[name].append([(float(r[name + "_seat0"]) + float(r[name + "_seat1"])) / 2 for r in rows])
                print(f"LBR hand={hand} seed={seed} output={destination}", flush=True)
            arrays = {name: np.asarray(v) for name, v in samples.items()}
            summary["models"][str(hand)] = dict(
                policies={name: estimate(v) for name, v in arrays.items()},
                paired_model_minus_baseline={p: estimate(arrays["model_" + p] - arrays["baseline_" + p])
                                             for p in ("current", "average")})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--check", action="store_true")
    training = sub.add_parser("train")
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", action="store_true")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            training.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            training.add_argument(flag, type=type(default), default=None)
    evaluation = sub.add_parser("lbr")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--pairs", type=int, default=250)
    evaluation.add_argument("--particles", type=int, default=64)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[2026100721, 2026100722],
                            help="LBR deal seeds (default: those of the dynamic_tree_validated 1M LBR)")
    evaluation.add_argument("--avg", choices=("sum", "latest"), default="sum")
    evaluation.add_argument("--hand", type=int, help="evaluate models/model_<hand>.bin (default: the last)")
    evaluation.add_argument("--all-models", action="store_true", help="evaluate every boundary snapshot")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        print(build(args.check))
    elif args.command == "lbr":
        evaluate_lbr(args)
    else:
        raise SystemExit(train(args))


if __name__ == "__main__":
    main()
