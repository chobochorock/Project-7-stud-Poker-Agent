"""Audit old/fixed and log-gain drivers without changing trained model sources.

Usage (project root): python -m agents.cpp_mccfr.dynamic_tree_validated.audit_split
Runs two 2048-hand controls at identical gain=.05, plus a synthetic split test.
Checks all stored solver state, not just scores. Output stays in family data/.
This is an implementation regression test, not a convergence experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from . import log_gain, run
from .test_dynamic_tree import rows
from .test_schedule import solver_payload


def audit(output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to(run.DATA.resolve()):
        raise ValueError("Audit output must be under the model family's data folder")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    old = run.DATA / "confirm_centered_seed222_50k"
    new = run.DATA / "log_gain_seed222_1m"
    manifests = [json.loads((p / "config.json").read_text()) for p in (old, new)]
    old_config, new_config = (m["config"] for m in manifests)
    if old_config["source_sha256"] != new_config["source_sha256"]:
        raise AssertionError("Completed runs used different common sources")
    verified_sources = []
    for relative, expected in old_config["source_sha256"].items():
        for path in (
            run.legacy.PROJECT / relative,
            old / "source" / relative,
            new / "source" / relative,
        ):
            if run.legacy.sha256(path) != expected:
                raise AssertionError(f"Source changed: {path}")
        verified_sources.append(relative)
    for relative, expected in new_config["log_gain_source_sha256"].items():
        for path in (run.legacy.PROJECT / relative, new / "source" / relative):
            if run.legacy.sha256(path) != expected:
                raise AssertionError(f"Log-gain source changed: {path}")

    compiler = shutil.which("g++")
    if not compiler:
        raise ValueError("g++ is required")
    binary = output / "audit_split.exe"
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            "-static-libstdc++",
            "-static-libgcc",
            str(run.HERE / "audit_split.cpp"),
            "-o",
            str(binary),
            "-lpsapi",
        ],
        check=True,
    )
    synthetic = json.loads(
        subprocess.check_output([str(binary), str(run.legacy.ATLAS)], text=True)
    )
    common = [
        "--hands",
        "2048",
        "--seed",
        "222",
        "--eval-seed",
        "307",
        "--particles",
        "32",
        "--panel-roots",
        "128",
        "--panel-every",
        "1024",
        "--checkpoint-every",
        "1024",
        "--window",
        "256",
        "--min-child",
        "32",
        "--min-rmse",
        "1",
        "--reserve-gib",
        "0",
    ]
    original, fixed = output / "original", output / "fixed_log_driver"
    original_args = [
        "train",
        "--run",
        str(original),
        *common,
        "--gain",
        "0.05",
        "--split-rule",
        "centered",
    ]
    fixed_args = ["train", "--run", str(fixed), *common, "--fixed-gain", "0.05"]
    if run.train(run.parser().parse_args(original_args)) != 0:
        raise AssertionError("Original driver failed")
    if log_gain.train(log_gain.parser().parse_args(fixed_args)) != 0:
        raise AssertionError("Log-gain driver failed")
    old_state, new_state = solver_payload(original), solver_payload(fixed)
    if old_state != new_state:
        raise AssertionError("Same-threshold drivers produced different solver states")
    old_splits, new_splits = rows(original / "splits.csv"), rows(fixed / "splits.csv")
    if old_splits != [
        {k: v for k, v in row.items() if k != "gain_threshold"} for row in new_splits
    ]:
        raise AssertionError("Split decisions differ")
    fixed_panel = [
        {k: v for k, v in row.items() if k != "gain_threshold"}
        for row in rows(fixed / "panel.csv")
    ]
    if rows(original / "panel.csv") != fixed_panel:
        raise AssertionError("Fixed-panel metrics differ")
    accepted = sum(int(row["accepted"]) for row in old_splits)
    if accepted == 0:
        raise AssertionError("Parity test did not exercise any splits")
    config_diff = {
        k: {"old": old_config.get(k), "new": new_config.get(k)}
        for k in sorted(old_config.keys() | new_config.keys())
        if old_config.get(k) != new_config.get(k)
    }
    result = dict(
        completed_run_identities=[m["identity"] for m in manifests],
        verified_common_sources=verified_sources,
        completed_config_differences=config_diff,
        synthetic=synthetic,
        control=dict(
            hands_per_driver=2048,
            seed=222,
            gain=0.05,
            window=256,
            min_child=32,
            min_rmse=1,
            particles=32,
            panel_roots=128,
            panel_every=1024,
            split_checks=len(old_splits),
            accepted_splits=accepted,
            tree_leaves=len(old_state[0][3]),
            tree_visits=old_state[0][1][0],
            logical_solver_state_equal=True,
            split_log_equal=True,
            panel_equal=True,
            compared="both solvers: regrets, average sums, touches, tree, monitor samples, RNGs, visit counts",
            original_arguments=original_args,
            fixed_driver_arguments=fixed_args,
        ),
        limit="Small exact-parity test, not a causal explanation of the 1M LBR difference",
    )
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=run.DATA / "split_audit_20261008"
    )
    audit(parser.parse_args().output)
