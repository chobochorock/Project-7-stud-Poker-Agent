"""Training CPU seconds of a 7-stud run (the clock of the equal-CPU-time comparison).

Usage: python train_clock.py RUN_DIR [RUN_DIR ...]
Prints, per run, the training CPU seconds at its last curve.csv row (and in hours): the value to pass as
-BudgetCpuSeconds to the network / ReBeL arms so they get the same training CPU time as that run.
- Network and ReBeL runs record it (curve.csv / snapshots.csv column train_cpu_seconds).
- Tabular runs: curve.csv's process CPU time (cpu_seconds, restarting at every resumption) minus the online LBR's time
  (online_lbr.csv online_seconds, single-threaded); the first row after a restart adds its training wall seconds
  instead, because the CPU time between the last row before a stop and the stop is in no row.
"""
import csv, os, sys
import numpy as np


def cpu_clock(d, cv, snap):
    """(curve-row train CPU seconds, {snapshot file: train CPU seconds}) or None when the run has no CPU column."""
    if cv and "train_cpu_seconds" in cv[0]:
        return (np.array([0.0] + [float(r["train_cpu_seconds"]) for r in cv]),
                {r["file"]: float(r["train_cpu_seconds"]) for r in snap if r.get("train_cpu_seconds")})
    if not cv or "cpu_seconds" not in cv[0]:
        return None
    online = [], []
    op = os.path.join(d, "online_lbr.csv")
    if os.path.exists(op):
        for r in csv.DictReader(open(op)): online[0].append(float(r["roots"])); online[1].append(float(r["online_seconds"]))
    o_at = lambda x: online[1][np.searchsorted(online[0], x, side="right") - 1] if online[0] and x >= online[0][0] else 0.0
    # process CPU time restarts at every resumption: the row after a restart adds its training wall seconds instead (the
    # CPU time between the last row before the stop and the stop is not in any row; the trainers are single-threaded)
    t, prev = [0.0], (0.0, 0.0, 0.0)  # previous row's cpu, online seconds, training seconds
    for r in cv:
        c, o, w = float(r["cpu_seconds"]), o_at(float(r["roots"])), float(r["seconds"])
        t.append(t[-1] + ((c - prev[0]) - (o - prev[1]) if c >= prev[0] else w - prev[2]))
        prev = (c, o, w)
    t = np.array(t)
    rr = np.array([0.0] + [float(r["roots"]) for r in cv])
    return t, {r["file"]: float(np.interp(float(r["roots"]), rr, t)) for r in snap}


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    for d in sys.argv[1:]:
        cv = list(csv.DictReader(open(os.path.join(d, "curve.csv"))))
        snap = list(csv.DictReader(open(os.path.join(d, "snapshots.csv"))))
        c = cpu_clock(d, cv, snap)
        if c is None or not cv:
            print(f"{d}: no CPU column (older run); training wall seconds {cv[-1]['seconds'] if cv else 0}")
            continue
        t = float(c[0][-1])
        print(f"{d}: training CPU {t:.0f} s ({t / 3600:.2f} h) at root {cv[-1]['roots']}; training wall {float(cv[-1]['seconds']):.0f} s")
