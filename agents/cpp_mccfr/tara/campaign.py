"""Run a TARA arm x seed matrix in parallel (each job is an ordinary run.py train or lbr call).

Usage (project root):
    python -m agents.cpp_mccfr.tara.campaign --name main_127k_20261008 \
        --arms tara b3_latest b4_uniform --seeds 111 222 333 444 555 --hands 127000 --jobs 8
    python -m agents.cpp_mccfr.tara.campaign --name main_127k_20261008 --arms tara ... --seeds ... \
        --hands 127000 --lbr sum latest [--all-models]
Runs go to data/tara/<name>/<arm>_seed<seed>/ with configs/<arm>.json; an unfinished run is resumed,
a finished one is skipped. --extra passes further run.py train options to every job (e.g. "--clusters 8").
--lbr evaluates finished runs instead: <name>/lbr/<arm>_seed<seed>_<avg>_s<deal seeds>_h<hands>/ per average output, run
one after another per run (they share the run's writer lock), at the final model or every boundary
snapshot (--all-models). A new job starts only while both free RAM and remaining commit (RAM + page
file) exceed --min-free-gib; a TARA job peaks at about 0.4 GiB (C++ with a full 4096-row reservoir,
then the Python fit), so keep the margin when other jobs share the machine.
"""

from __future__ import annotations

import os

# numpy's BLAS reserves ~0.5 GiB of commit per process with one buffer per hardware thread; the jobs
# need no BLAS threading (sklearn trees and the C++ solver are single-threaded). Set before numpy loads;
# child jobs inherit it.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from . import run  # noqa: E402

HERE = Path(__file__).resolve().parent


def free_gib() -> float:
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    # Smaller of free physical memory and remaining commit (RAM + page file): other users' jobs share both.
    return min(status.available, status.page_available) / 2**30


def done(path: Path, hands: int) -> bool:
    state = path / "progress.json"
    if not state.exists():
        return False
    progress = json.loads(state.read_text())
    return progress["hand"] >= hands and not progress["pending_fit"]


def jobs(args) -> list[tuple[str, list[list[str]]]]:
    root = run.DATA / args.name
    module = [sys.executable, "-m", "agents.cpp_mccfr.tara.run"]
    result = []
    for arm in args.arms:
        config = HERE / "configs" / f"{arm}.json"
        if not config.exists():
            raise SystemExit(f"missing {config}")
        for seed in args.seeds:
            path = root / f"{arm}_seed{seed}"
            if args.lbr:
                if not done(path, args.hands):
                    print(f"SKIP unfinished {path.name}", flush=True)
                    continue
                commands = []
                seeds = args.lbr_seeds or [2026100721, 2026100722]  # run.py lbr default (DT 1M LBR seeds)
                for avg in args.lbr:
                    output = root / "lbr" / f"{path.name}_{avg}_s{'-'.join(map(str, seeds))}_h{args.hands}"
                    if output.exists():
                        state = "complete" if (output / "summary.json").exists() else "INCOMPLETE (remove to redo)"
                        print(f"SKIP existing {state} {output.name}", flush=True)
                        continue
                    commands.append(module + ["lbr", "--run", str(path), "--output", str(output), "--avg", avg,
                                              *(["--all-models"] if args.all_models else []),
                                              "--seeds", *map(str, seeds)])
                if commands:
                    result.append((f"{path.name}_lbr", commands))
                continue
            if done(path, args.hands):
                print(f"SKIP finished {path.name}", flush=True)
                continue
            command = module + ["train", "--run", str(path), "--hands", str(args.hands)]
            if (path / "progress.json").exists():
                command += ["--resume"]
            else:
                command += ["--config", str(config), "--seed", str(seed),
                            *args.extra.replace("{seed}", str(seed)).split()]
            result.append((path.name, [command]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--hands", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=2.5)
    parser.add_argument("--extra", default="", help="extra run.py train options for every job; {seed} is replaced")
    parser.add_argument("--lbr", nargs="+", choices=("sum", "latest"), help="evaluate finished runs instead")
    parser.add_argument("--all-models", action="store_true", help="with --lbr: every boundary snapshot")
    parser.add_argument("--lbr-seeds", type=int, nargs="+", help="with --lbr: deal seeds (default: run.py lbr default)")
    args = parser.parse_args()
    root = run.DATA / args.name
    root.mkdir(parents=True, exist_ok=True)
    queue = jobs(args)
    run.build(lbr=bool(args.lbr))  # build once before the jobs start
    running, failed = [], []
    while queue or running:
        for job in list(running):
            name, commands, process, log = job
            if process.poll() is None:
                continue
            if process.returncode == 0 and commands:
                job[2] = subprocess.Popen(commands.pop(0), cwd=run.legacy.PROJECT, stdout=log,
                                          stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                continue
            log.close()
            running.remove(job)
            if process.returncode:
                failed.append(name)
            print(f"{'DONE' if process.returncode == 0 else f'FAILED code={process.returncode}'} {name} "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
        while queue and len(running) < args.jobs and free_gib() > args.min_free_gib:
            name, commands = queue.pop(0)
            log = (root / f"{name}.campaign.log").open("a", encoding="utf-8")
            process = subprocess.Popen(commands.pop(0), cwd=run.legacy.PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            running.append([name, commands, process, log])
            print(f"START {name} {time.strftime('%H:%M:%S')} free_gib={free_gib():.1f}", flush=True)
            time.sleep(2)
        time.sleep(10)
    print("FAILED " + " ".join(failed) if failed else "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
