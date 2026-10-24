"""Distil a frozen C++ MCCFR teacher into Transformer/MLP students and score them with Policy-LBR.

Usage (project root; g++ on PATH; TORCH_PY = Torch runtime with Toy .deep_cfr_deps):
    python -m agents.cpp_mccfr.neural_distill.run build   --run agents/cpp_mccfr/data/neural_distill_RUN
    python -m agents.cpp_mccfr.neural_distill.run export  --run ... [--hands 400000 --seed 20261009]
    python -m agents.cpp_mccfr.neural_distill.run train   --run ... [--models transformer:11 mlp:11 ...]
    python -m agents.cpp_mccfr.neural_distill.run check   --run ...   # parity + privacy self-test
    python -m agents.cpp_mccfr.neural_distill.run lbr     --run ... [--seeds 2026100401 2026100402]
    python -m agents.cpp_mccfr.neural_distill.run analyze --run ...
Teacher (--teacher at build, stored in RUN/teacher.json): "dt=RUN_DIR" (matched hard256 betting-summary
    + seat k-means stored in a dynamic-tree checkpoint; default agents/cpp_mccfr/data/dynamic_tree/seed111,
    1M hands) or "mccfr=ATLAS,MODEL,BUCKET,HANDS" (e.g. the K512 power-memory16 10M table); read only.
Metric: Policy-LBR attacker ante/hand (approximate best response; higher = worse), current and average
    policies of student and teacher on identical deals plus uniform. Not exploitability.
See README.md for the training objective, data distribution, budgets and limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PROJECT = FAMILY.parents[1]
DEFAULT_TEACHER = "dt=" + str(FAMILY / "data/dynamic_tree/seed111")
TORCH_PY = os.environ.get(
    "TORCH_PY", "C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
)
NAMES = ("model_current", "baseline_current", "model_average", "baseline_average", "uniform")
SOURCES = ("tokens.hpp", "net.hpp", "teacher.hpp", "export.cpp", "evaluate.cpp", "train.py", "run.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_spec(run: Path) -> str:
    return json.loads((run / "teacher.json").read_text(encoding="utf-8"))["spec"]


def teacher_files(spec: str) -> list[Path]:
    if spec.startswith("dt="):
        root = Path(spec[3:])
        return [root / "checkpoint.bin", root / "hand_atlas.bin"]
    atlas, model = spec[6:].split(",")[:2]
    return [Path(atlas), Path(model)]


def build(run: Path, spec: str) -> None:
    directory = run / "build"
    directory.mkdir(parents=True)
    source_copy = run / "source"
    source_copy.mkdir()
    for name in SOURCES:
        shutil.copy2(HERE / name, source_copy / name)
    text = (FAMILY / "dynamic_tree_unlimited/solver.cpp").read_text(encoding="utf-8")
    signature, include = "int main(int argc, char** argv)", '#include "../hand_recall.cpp"'
    if text.count(signature) != 1 or text.count(include) != 1:
        raise ValueError("Engine entrypoint/include changed; inspect before rebuilding")
    text = text.replace(signature, "int unused_tree_training_main(int argc, char** argv)")
    text = text.replace(include, f'#include "{(FAMILY / "hand_recall.cpp").as_posix()}"')
    (directory / "tree_engine.inc").write_text(text, encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ not found")
    for name in ("export", "evaluate"):
        command = [compiler, "-O3", "-std=c++17", "-DNOMINMAX", "-static-libstdc++", "-static-libgcc",
                   "-I", str(directory), str(HERE / f"{name}.cpp"), "-o", str(directory / f"{name}.exe"), "-lpsapi"]
        with (directory / f"{name}_build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = dict(
        sources={name: sha256(HERE / name) for name in SOURCES},
        engine={str(p.relative_to(PROJECT)): sha256(p) for p in (
            FAMILY / "dynamic_tree_unlimited/solver.cpp", FAMILY / "hand_recall.cpp", FAMILY / "stud_mccfr.cpp",
            FAMILY / "stud_rules.hpp", FAMILY / "abstraction_lbr/evaluate.hpp")},
        teacher={str(p): sha256(p) for p in teacher_files(spec)},
        compiler=subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
    )
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run / "teacher.json").write_text(json.dumps(dict(spec=spec), indent=2), encoding="utf-8")
    print("BUILD done", flush=True)


def export_keys(run: Path) -> None:
    """Replay the dataset's hands (same seed/hands, deterministic) and write the teacher keys sidecar."""
    meta = json.loads((run / "export.json").read_text(encoding="utf-8"))
    with (run / "export_keys.log").open("w") as log:
        subprocess.run([str(run / "build/export.exe"), teacher_spec(run), str(run / "keys.bin"), str(meta["hands"]),
                        str(meta["seed"]), "--keys-only"], stdout=log, stderr=subprocess.STDOUT, check=True)
    print((run / "export_keys.log").read_text().splitlines()[-1], flush=True)


def export(run: Path, hands: int, seed: int) -> None:
    with (run / "export.log").open("w") as log:
        subprocess.run([str(run / "build/export.exe"), teacher_spec(run), str(run / "dataset.bin"), str(hands), str(seed)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    (run / "export.json").write_text(json.dumps(dict(hands=hands, seed=seed, teacher=teacher_spec(run))), encoding="utf-8")
    print((run / "export.log").read_text().splitlines()[-1], flush=True)


def parse_models(values: list[str]) -> list[tuple[str, int]]:
    return [(item.split(":")[0], int(item.split(":")[1])) for item in values]


def train(run: Path, models: list[tuple[str, int]], steps: int, threads: int, parallel: int) -> None:
    pending = []
    for architecture, seed in models:
        output = run / f"{architecture}_seed{seed}"
        command = [TORCH_PY, "-m", "agents.cpp_mccfr.neural_distill.train", "--dataset", str(run / "dataset.bin"),
                   "--architecture", architecture, "--seed", str(seed), "--steps", str(steps),
                   "--threads", str(threads), "--output", str(output)]
        if architecture == "mlp_key":
            command += ["--keys", str(run / "keys.bin")]
        log = (run / f"train_{architecture}_seed{seed}.log").open("w")
        pending.append((subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT), log, output))
        while len([p for p in pending if p[0].poll() is None]) >= parallel:
            next(p for p in pending if p[0].poll() is None)[0].wait()
    for process, log, output in pending:
        if process.wait() != 0:
            raise RuntimeError(f"training failed: {output}")
        log.close()
    print("TRAIN done", flush=True)


def students(run: Path) -> list[Path]:
    return sorted(p for p in run.iterdir() if p.is_dir() and (p / "weights.ndnet").is_file())


def check(run: Path, evaluator: str = "evaluate.exe", only: list[str] | None = None) -> None:
    """Parity + privacy self-test per student; merges into checks.json and records the binary used."""
    exe = run / "build" / evaluator
    path = run / "checks.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for student in students(run):
        if only and student.name not in only:
            continue
        parity_command = [str(exe), "--parity", str(student / "weights.ndnet"), str(run / "dataset.bin"),
                          str(student / "parity_reference.bin")]
        if student.name.startswith("mlp_key"):
            parity_command.append(str(run / "keys.bin"))
        parity = subprocess.run(parity_command, capture_output=True, text=True)
        privacy = subprocess.run([str(exe), "--self-test", teacher_spec(run), str(student / "weights.ndnet")],
                                 capture_output=True, text=True)
        results[student.name] = dict(parity=parity.stdout.strip(), parity_ok=parity.returncode == 0,
                                     self_test=privacy.stdout.strip(), self_test_ok=privacy.returncode == 0,
                                     binary=evaluator, sha256=sha256(exe))
        if parity.returncode or privacy.returncode:
            raise RuntimeError(f"check failed for {student.name}: {parity.stdout} {parity.stderr} {privacy.stderr}")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


def policies_to_score(run: Path, controls: list[str]) -> list[tuple[str, str]]:
    """(label, STUDENT argument for evaluate.exe): trained students plus smoothing controls."""
    items = [(student.name, str(student / "weights.ndnet")) for student in students(run)]
    items += [(control.replace(":", "_"), control) for control in controls]
    return items


def lbr(run: Path, seeds: list[int], pairs: int, particles: int, parallel: int, tag: str = "lbr",
        controls: list[str] | None = None, only: list[str] | None = None, evaluator: str = "evaluate.exe") -> None:
    exe = run / "build" / evaluator
    items = [item for item in policies_to_score(run, controls or []) if not only or item[0] in only]
    running = []
    for label, argument in items:
        for seed in seeds:
            output = run / tag / f"{label}_{seed}"
            if (output / "evaluation.json").is_file():
                continue
            output.parent.mkdir(exist_ok=True)
            log = (run / tag / f"{label}_{seed}.log").open("w")
            command = [str(exe), teacher_spec(run), argument, str(output), str(pairs), str(particles), str(seed)]
            running.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, output))
            while len([r for r in running if r[0].poll() is None]) >= parallel:
                next(r for r in running if r[0].poll() is None)[0].wait()
    for process, log, output in running:
        if process.wait() != 0:
            raise RuntimeError(f"LBR failed: {output}")
        log.close()
    (run / tag).mkdir(exist_ok=True)
    protocol_path = run / tag / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    if protocol and (protocol["seeds"] != seeds or protocol["pairs"] != pairs or protocol["particles"] != particles):
        raise ValueError("protocol differs from earlier runs under this tag")
    evaluators = protocol.get("evaluators", {})
    for label, _ in items:
        evaluators[label] = dict(binary=evaluator, sha256=sha256(exe))
    protocol.update(seeds=seeds, pairs=pairs, particles=particles, evaluators=evaluators)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("LBR done", flush=True)


def estimate(values: np.ndarray) -> dict:
    """Seed-stratified deal-pair estimate (same as abstraction_lbr.run.estimate)."""
    sys.path.insert(0, str(PROJECT))
    from agents.cpp_mccfr.abstraction_lbr.run import estimate as base_estimate

    return base_estimate(values)


def holm(p_values: dict) -> dict:
    order = sorted(p_values, key=p_values.get)
    adjusted, running_max = {}, 0.0
    for rank, key in enumerate(order):
        running_max = max(running_max, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running_max
    return adjusted


def two_sided_p(est: dict) -> float:
    from math import erf, sqrt

    z = est["mean"] / est["se"] if est["se"] > 0 else 0.0
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def load_matrix(run: Path, tag: str, label: str, seeds: list[int], pairs: int):
    import csv

    columns = {name: [] for name in NAMES}
    deals, diagnostics = [], []
    for seed in seeds:
        directory = run / tag / f"{label}_{seed}"
        with (directory / "paired_payoffs.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != pairs:
            raise ValueError(f"{directory}: {len(rows)} pairs, expected {pairs}")
        engine = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        for name in NAMES:
            values = [(float(r[f"{name}_seat0"]) + float(r[f"{name}_seat1"])) / 2 for r in rows]
            if any(abs(float(r[f"{name}_seat{s}"])) > 1000 for r in rows for s in (0, 1)):
                raise ValueError(f"{directory}: payoff out of bounds")
            if abs(np.mean(values) - engine["policies"][name]["mean"]) > 1e-9:
                raise ValueError(f"{directory}: CSV mean differs from engine mean for {name}")
            columns[name].append(values)
        deals.append([r["deal_seed"] for r in rows])
        diag = json.loads((directory / "student_diagnostics.json").read_text(encoding="utf-8"))
        diag["engine"] = engine["policies"]
        diagnostics.append(diag)
    return {name: np.array(values) for name, values in columns.items()}, deals, diagnostics


def attack_fit(diagnostics: list) -> dict:
    fit = {}
    for key in ("current_actual", "current_hypothetical", "average_actual", "average_hypothetical"):
        queries = [d[key]["queries"] for d in diagnostics]
        total = sum(queries)
        fit[key] = dict(queries=int(total))
        for k in ("mean_tv", "best_action_agreement"):
            fit[key][k] = float(sum(d[key][k] * q for d, q in zip(diagnostics, queries)) / total) if total else None
    return fit


def engine_totals(diagnostics: list) -> dict:
    out = {}
    for name in NAMES:
        out[name] = {k: float(sum(d["engine"][name][k] for d in diagnostics)) for k in ("seconds", "queries", "decisions")}
        out[name]["mean_ess"] = float(np.mean([d["engine"][name]["mean_ess"] for d in diagnostics]))
    return out


def analyze(run: Path, seeds: list[int], tag: str = "lbr", controls: list[str] | None = None) -> None:
    protocol = json.loads((run / tag / "protocol.json").read_text(encoding="utf-8"))
    pairs = protocol["pairs"]
    result = {"tag": tag, "seeds": seeds, "protocol": protocol, "policies": {}}
    shared, deal_stream, matrices = {}, None, {}
    for label, _ in policies_to_score(run, controls or []):
        matrix, deals, diagnostics = load_matrix(run, tag, label, seeds, pairs)
        if deal_stream is None:
            deal_stream = deals
        elif deals != deal_stream:
            raise ValueError(f"{label}: deal stream differs from the other policies")
        for name in ("baseline_current", "baseline_average", "uniform"):
            if name in shared and not np.array_equal(shared[name], matrix[name]):
                raise ValueError(f"{label}: {name} differs; teacher or deals not identical across runs")
            shared[name] = matrix[name]
        matrices[label] = matrix
        entry = dict(
            lbr={name: estimate(matrix[name]) for name in NAMES},
            minus_teacher={policy: estimate(matrix[f"model_{policy}"] - matrix[f"baseline_{policy}"])
                           for policy in ("current", "average")},
            attack_fit=attack_fit(diagnostics),
            engine=engine_totals(diagnostics),
        )
        metrics_path = run / label / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            final = metrics["final_validation"]
            entry["validation_fit"] = {k: final.get(k) for k in ("tv_current", "tv_average", "agree_current", "agree_average", "kl_average")}
            entry["training"] = dict(parameters=metrics["parameters"], train_seconds=metrics["train_seconds"],
                                     objective=metrics.get("objective"), steps=metrics["args"].get("steps"),
                                     target=metrics["args"].get("target", "maxabs"),
                                     train_sha256=metrics.get("source_sha256"))
        result["policies"][label] = entry
    groups = {}
    for label in matrices:
        if "_seed" in label:
            groups.setdefault(label.split("_seed")[0], []).append(label)
    pooled, p_values = {}, {}
    for architecture, labels in groups.items():
        objectives = {result["policies"][l].get("training", {}).get("objective") for l in labels}
        if len(objectives) != 1:
            raise ValueError(f"{architecture}: students trained with different objectives {objectives}")
        pooled[architecture] = {"students": labels}
        for policy in ("current", "average"):
            difference = np.mean([matrices[l][f"model_{policy}"] - matrices[l][f"baseline_{policy}"] for l in labels], axis=0)
            pooled[architecture][f"minus_teacher_{policy}"] = estimate(difference)
            pooled[architecture][f"absolute_{policy}"] = estimate(np.mean([matrices[l][f"model_{policy}"] for l in labels], axis=0))
            p_values[f"{architecture}_minus_teacher_{policy}"] = two_sided_p(pooled[architecture][f"minus_teacher_{policy}"])
    if {"transformer", "mlp"} <= set(groups):
        for policy in ("current", "average"):
            t = np.mean([matrices[l][f"model_{policy}"] for l in groups["transformer"]], axis=0)
            m = np.mean([matrices[l][f"model_{policy}"] for l in groups["mlp"]], axis=0)
            pooled[f"transformer_minus_mlp_{policy}"] = estimate(t - m)
            p_values[f"transformer_minus_mlp_{policy}"] = two_sided_p(pooled[f"transformer_minus_mlp_{policy}"])
    result["pooled"] = pooled
    result["p_values_normal"] = p_values
    result["holm_adjusted"] = holm(p_values) if p_values else {}
    result["teacher"] = {name: estimate(shared[name]) for name in ("baseline_current", "baseline_average", "uniform")}
    (run / f"summary_{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(run, result, tag)
    print(json.dumps({k: {p: round(v["lbr"][p]["mean"], 4) for p in NAMES} for k, v in result["policies"].items()}, indent=2))


def plot(run: Path, result: dict, tag: str) -> None:
    """Linear axis (keeps negative/zero values), bootstrap 95% intervals over deal pairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, means, low, high, colors = [], [], [], [], []
    teacher = result["teacher"]
    for name, color in (("baseline_current", "#4c78a8"), ("baseline_average", "#72b7b2")):
        labels.append("teacher " + name.split("_")[1])
        means.append(teacher[name]["mean"])
        low.append(teacher[name]["bootstrap_ci95"][0])
        high.append(teacher[name]["bootstrap_ci95"][1])
        colors.append(color)
    for student, entry in result["policies"].items():
        for policy, color in (("current", "#e45756"), ("average", "#f58518")):
            estimate_ = entry["lbr"][f"model_{policy}"]
            labels.append(f"{student} {policy}")
            means.append(estimate_["mean"])
            low.append(estimate_["bootstrap_ci95"][0])
            high.append(estimate_["bootstrap_ci95"][1])
            colors.append(color)
    figure, axis = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    x = np.arange(len(labels))
    axis.bar(x, means, color=colors)
    axis.errorbar(x, means, yerr=[np.array(means) - low, np.array(high) - means], fmt="none", color="black", capsize=3)
    axis.axhline(teacher["uniform"]["mean"], color="gray", linestyle="--", linewidth=1, label="uniform")
    axis.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="always fold (1 ante)")
    axis.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    axis.set_ylabel("Policy-LBR attacker winnings (ante/hand)")
    axis.set_title("Teacher vs distilled students, identical deals; bars = deal-pair bootstrap 95%")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run / f"lbr_comparison_{tag}.png", dpi=130)
    plt.close(figure)


def reproduce(run: Path, seeds: list[int], reference: list[Path], tag: str = "lbr") -> None:
    """Compare teacher rows with an earlier per-pair CSV (columns current_seat0.. average_seat1) on the shared prefix."""
    import csv

    report = {}
    for seed, path in zip(seeds, reference):
        with path.open(encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
        for student in students(run):
            directory = run / tag / f"{student.name}_{seed}"
            if not (directory / "paired_payoffs.csv").is_file():
                continue
            with (directory / "paired_payoffs.csv").open(encoding="utf-8") as handle:
                new = list(csv.DictReader(handle))
            shared = min(len(old), len(new))
            mismatches = 0
            for a, b in zip(old[:shared], new[:shared]):
                if a["deal_seed"] != b["deal_seed"]:
                    mismatches += 1
                    continue
                for policy in ("current", "average"):
                    for seat in (0, 1):
                        if float(a[f"{policy}_seat{seat}"]) != float(b[f"baseline_{policy}_seat{seat}"]):
                            mismatches += 1
            report[f"{student.name}_{seed}"] = dict(reference=str(path), shared_pairs=shared, mismatching_values=mismatches)
            break
    (run / "teacher_reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("build", "export", "export-keys", "train", "check", "lbr", "analyze", "reproduce"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help="teacher spec, build step only")
    parser.add_argument("--hands", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--models", nargs="+", default=["transformer:11", "transformer:22", "transformer:33",
                                                        "mlp:11", "mlp:22", "mlp:33"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026100401, 2026100402])
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--tag", default="lbr", help="output subfolder for lbr/analyze/reproduce")
    parser.add_argument("--controls", nargs="*", default=[], help="e.g. smooth:0.1 smooth:0.3 (teacher mixed with uniform)")
    parser.add_argument("--only", nargs="*", default=[], help="lbr/check: restrict to these policy labels")
    parser.add_argument("--evaluator", default="evaluate.exe", help="lbr/check: binary name inside RUN/build")
    parser.add_argument("--reference", type=Path, nargs="*", default=[], help="reproduce: earlier per-pair CSVs, one per seed")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else PROJECT / args.run
    if args.step == "build":
        if run.exists():
            raise ValueError("run directory exists; use a new one")
        build(run, args.teacher)
    elif args.step == "export":
        export(run, args.hands, args.seed)
    elif args.step == "export-keys":
        export_keys(run)
    elif args.step == "train":
        train(run, parse_models(args.models), args.steps, args.threads, args.parallel)
    elif args.step == "check":
        check(run, args.evaluator, args.only)
    elif args.step == "lbr":
        lbr(run, args.seeds, args.pairs, args.particles, args.parallel, args.tag, args.controls, args.only, args.evaluator)
    elif args.step == "reproduce":
        reproduce(run, args.seeds, args.reference, args.tag)
    else:
        analyze(run, args.seeds, args.tag, args.controls)


if __name__ == "__main__":
    main()
