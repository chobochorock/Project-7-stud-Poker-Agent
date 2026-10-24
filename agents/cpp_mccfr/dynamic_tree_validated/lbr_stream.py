"""Evaluate frozen DT/KM checkpoints without retaining split-monitor samples.

Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.lbr_stream --help
See LBR_STREAM.md. Source/model files stay unchanged; build adapters and results
live under the selected NEW family data output. No training or exports on import.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import log_gain, schedule, run as base
from ..abstraction_lbr.run import NAMES, estimate
from ..dynamic_tree.review import checkpoint_header


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Source shape changed; inspect stream adapter: " + old[:80])
    return source.replace(old, new)


def build(output: Path) -> Path:
    directory = output / "build"
    directory.mkdir()
    source = (base.HERE / "solver.cpp").read_text(encoding="utf-8")
    first = "    void load(std::istream& in) {"
    end = "\n};\n\nstruct Target {"
    if source.count(first) != 1 or source.count(end) != 1:
        raise ValueError("Loader boundaries changed")
    original = source[source.index(first):source.index(end)]
    header = """    void load(std::istream& in) {
        const auto begin = in.tellg();
        in.seekg(0, std::ios::end);
        const auto end = in.tellg();
        hand_recall::require(begin >= 0 && end >= begin, "non-seekable checkpoint");
        in.seekg(begin);
"""
    modified = replace_once(original, first + "\n", header)
    loop = """            for (uint32_t i = 0; i < size; ++i) {
                Sample sample;
                sample.x = read<Features>(in); sample.regret = read<Values>(in); sample.hand = read<uint64_t>(in);
                for (auto value : sample.x) { require(std::isfinite(value), "invalid feature"); }
                for (auto value : sample.regret) { require(std::isfinite(value), "invalid monitor regret"); }
                node.samples.push_back(sample);
            }"""
    skip = """            if (size) {
                static_assert(sizeof(Features) + sizeof(Values) + sizeof(uint64_t) == 208);
                static_assert(sizeof(std::streamoff) >= 8);
                const auto bytes = std::streamoff(size) * 208;
                const auto position = in.tellg();
                require(position >= 0 && end - position >= bytes, "truncated monitor payload");
                in.seekg(bytes, std::ios::cur);
                require(bool(in), "monitor seek failed");
            }"""
    modified = replace_once(modified, loop, skip)
    modified = replace_once(modified, "        tree.clear(); roots.clear(); pending.clear(); frozen.clear();",
                            "        tree.clear(); roots.clear(); pending.clear(); frozen.clear();\n        tree.reserve(count);")
    source = replace_once(source, original, modified + original.replace("void load(", "void load_reference(", 1))
    source = replace_once(source, '#include "../hand_recall.cpp"',
                          f'#include "{(base.legacy.FAMILY / "hand_recall.cpp").as_posix()}"')
    (directory / "stream_solver.inc").write_text(source, encoding="utf-8")
    adapter = (base.HERE / "lbr.cpp").read_text(encoding="utf-8")
    if adapter.count("\nint main(") != 1:
        raise ValueError("Adapter main changed")
    adapter = adapter[:adapter.index("\nint main(")]
    adapter = replace_once(adapter, '#include "solver.cpp"', '#include "stream_solver.inc"')
    adapter = replace_once(adapter, '#include "../abstraction_lbr/evaluate.hpp"',
                           f'#include "{(base.legacy.FAMILY / "abstraction_lbr/evaluate.hpp").as_posix()}"')
    (directory / "stream_adapter.inc").write_text(adapter, encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    binary = directory / "lbr_stream.exe"
    command = [compiler, "-O3", "-std=c++17", "-static-libstdc++", "-static-libgcc",
               "-I", str(directory), str(base.HERE / "lbr_stream.cpp"), "-o", str(binary), "-lpsapi"]
    with (directory / "build.log").open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return binary


def execute(args, binary: Path, output: Path) -> dict:
    run = args.run.resolve()
    with base.writer_lock(run):
        config = json.loads((run / "config.json").read_text())["config"]
        reader = log_gain.manifest_at if "gain_schedule" in config else schedule.manifest_at
        manifest = reader(run)
        hands = checkpoint_header(run, manifest["identity"])
        watched = [run / name for name in ("checkpoint.bin", "hand_atlas.bin", "config.json", "panel.csv", "splits.csv", "per_hand.csv")]
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
        command = [str(binary), str(run), manifest["identity"], str(output),
                   str(args.pairs), str(args.particles), *map(str, args.seeds)]
        with (output / "console.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"LBR failed; inspect {output / 'console.log'}")
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
        if after != before:
            raise ValueError("Input files changed during evaluation")
    samples = {name: [] for name in NAMES}
    details = []
    for seed in args.seeds:
        directory = output / str(seed)
        metadata = json.loads((directory / "evaluation.json").read_text())
        assert (metadata["training_hands"], metadata["pairs"], metadata["particles"], metadata["seed"]) == (hands,args.pairs,args.particles,seed)
        details.append(metadata)
        with (directory / "paired_payoffs.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == args.pairs and [int(r["pair"]) for r in rows] == list(range(args.pairs))
        assert len({r["deal_seed"] for r in rows}) == args.pairs
        for name in NAMES:
            values = np.array([[float(r[name+"_seat0"]),float(r[name+"_seat1"])] for r in rows])
            assert np.isfinite(values).all() and (np.abs(values) <= 1000).all()
            assert np.isclose(values.mean(),metadata["policies"][name]["mean"])
            samples[name].append(values.mean(axis=1))
    arrays = {name: np.asarray(values) for name,values in samples.items()}
    summary = dict(run=str(run), training_hands=hands, seeds=args.seeds, pairs_per_seed=args.pairs,
                   particles=args.particles, metric="LBR attacker ante/hand, not exact exploitability",
                   policies={name: estimate(values) for name,values in arrays.items()},
                   paired_model_minus_baseline={p: estimate(arrays["model_"+p]-arrays["baseline_"+p]) for p in ("current","average")},
                   input_unchanged=True, input_stat=before, model_identity=manifest["identity"],
                   exe_sha256=base.legacy.sha256(binary), command=command,
                   loader=json.loads((output / "loader.json").read_text()), evaluations=details)
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pairs",type=int,default=250)
    parser.add_argument("--particles",type=int,default=64)
    parser.add_argument("--seeds",type=int,nargs="+",default=[2026100901,2026100902])
    parser.add_argument("--reuse-build",type=Path,help="Verified prior lbr_stream output; avoids recompilation")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(base.DATA.resolve()):
        parser.error("Use a NEW output under dynamic_tree_validated data")
    if args.pairs < 2 or args.particles < 1 or len(set(args.seeds)) != len(args.seeds) or any(s<=0 or s>=2**64 for s in args.seeds):
        parser.error("Invalid budgets or seeds")
    config = json.loads((args.run / "config.json").read_text())["config"]
    (log_gain.manifest_at if "gain_schedule" in config else schedule.manifest_at)(args.run.resolve())
    hashes = dict(base.hashes())
    for path in (Path(__file__), base.HERE / "lbr_stream.cpp"):
        hashes[str(path.relative_to(base.legacy.PROJECT))] = base.legacy.sha256(path)
    output.mkdir(parents=True)
    if args.reuse_build:
        origin = args.reuse_build.resolve()
        saved = json.loads((origin / "build_manifest.json").read_text())
        binary = Path(saved["binary"])
        if saved["source_sha256"] != hashes or base.legacy.sha256(binary) != saved["binary_sha256"]:
            raise ValueError("Build/source hash mismatch")
    else:
        binary = build(output)
    record = dict(source_sha256=hashes,binary=str(binary),binary_sha256=base.legacy.sha256(binary))
    (output/"build_manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    result = execute(args,binary,output)
    print(json.dumps({"policies":result["policies"],"paired":result["paired_model_minus_baseline"],"loader":result["loader"]},indent=2))


if __name__ == "__main__":
    main()
