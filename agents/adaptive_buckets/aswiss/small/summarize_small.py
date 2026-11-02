"""Summary tables of run_small.sh output (and the matched k-means runs of kmeans_matched.sh, if present).

Usage: python summarize_small.py OUT > summary.md
Per game: final exact exploitability (NashConv/2, chips) of the bucket average and the current policy, median and
range over the random abstractions; buckets; for ASWISS the iteration of the last split, whether the run ended in a
stop state (no split in the last judgments), and how many final buckets mix different own pasts (imperfect recall).
D/2 is the bucket deviation gain of the final average policy in its partition (onpath.py; 0 at an agent-form equilibrium
of the abstract game); for ASWISS it is shown only when the last judgment split nothing (the final map is then the
partition that policy was trained on).
"""
import csv, glob, os, re, sys
import numpy as np
from onpath import split

OUT = sys.argv[1]


def rows(d, name):
    p = f"{d}/{name}"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def final(d):
    r = rows(d, "exploitability.csv")
    return (float(r[-1]["expl_bucket_avg"]), float(r[-1]["expl_current"]), int(r[-1]["buckets_after"]),
            float(r[-1]["avg_rhoP_half"]), int(r[-1]["iters"])) if r else None


def mixing(tree_dir, bmap):
    prev = np.fromfile(f"{tree_dir}/prev.bin", np.int32); act = np.fromfile(f"{tree_dir}/prev_action.bin", np.int32)
    b = np.fromfile(bmap, np.int32); n = len(b)
    depth = np.zeros(n, int)
    for I in range(n):
        J = I
        while prev[J] >= 0: depth[I] += 1; J = prev[J]
    past, sig = {}, {}
    for I in np.argsort(depth, kind="stable"):
        past[I] = () if prev[I] < 0 else past[prev[I]] + ((int(b[prev[I]]), int(act[I])),)
        sig.setdefault(int(b[I]), set()).add(past[I])
    return sum(len(s) > 1 for s in sig.values()), len(sig)


def fmt(v):
    v = np.array(v)
    if len(v) == 1: return f"{v[0]:.2e}"
    return f"{np.median(v):.2e} [{v.min():.1e}, {v.max():.1e}]"


for game in ("kuhn", "leduc"):
    R = f"{OUT}/runs/{game}"
    if not os.path.isdir(R): continue
    arms = sorted(os.listdir(R))
    print(f"\n## {game}\n")
    ref = final(f"{R}/ref")
    print(f"unabstracted CFR+ after {ref[4]:,} iterations: average {ref[0]:.2e}, current {ref[1]:.2e}, certificate rho/2 {ref[3]:.4f}\n")
    print("| arm | runs | buckets | average policy | current policy | rho_P/2 of the average | D/2 of the average |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    groups = {}
    for a in arms:
        if a == "ref": continue
        key = re.sub(r"_s\d+$", "", a)
        f = final(f"{R}/{a}")
        if f:
            o = rows(f"{R}/{a}", "onpath.csv")
            groups.setdefault(key, []).append(f + (float(o[-1]["gain_half"]) if o else None,))
    for key, fs in groups.items():
        if key.startswith("aswiss"): continue
        gains = [x[5] for x in fs if x[5] is not None]
        print(f"| {key} | {len(fs)} | {fmt([x[2] for x in fs])} | {fmt([x[0] for x in fs])} | {fmt([x[1] for x in fs])} | "
              f"{fmt([x[3] for x in fs])} | {fmt(gains) if gains else '-'} |")
    print("\n| ASWISS arm | runs | final buckets | average policy | current policy | last split (iterations) | stop state | final buckets mixing own pasts | D/2 (stop-state runs) | matched k-means average |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, fs in groups.items():
        if not key.startswith("aswiss"): continue
        runs = [a for a in arms if re.sub(r"_s\d+$", "", a) == key]
        last, stop, mix, km, gain = [], 0, [], [], []
        for a in runs:
            j = rows(f"{R}/{a}", "judgments.csv"); c = rows(f"{R}/{a}", "closure.csv")
            split_iters = [int(x["iters"]) for x in j if int(x["splits"]) > 0] + [int(x["iters"]) for x in c if int(x["divided"]) > 0]
            last.append(max(split_iters) if split_iters else 0)
            stopped = bool(j) and int(j[-1]["splits"]) == 0 and float(j[-1]["G_half"]) <= 1e-12 and float(j[-1]["HK_half"]) <= 1e-12
            stop += stopped
            if stopped and (not c or int(c[-1]["divided"]) == 0):
                k_last = len(rows(f"{R}/{a}", "log.csv")) - 1
                gain.append(split(f"{OUT}/{game}", np.fromfile(f"{R}/{a}/final_bucket_map.bin", np.int32),
                                  np.fromfile(f"{R}/{a}/policy_bucket_{k_last}.bin"))[4])
            m, nb = mixing(f"{OUT}/{game}", f"{R}/{a}/final_bucket_map.bin"); mix.append(m / nb)
            k = final(f"{OUT}/runs_km/{game}/{a}")
            if k: km.append(k[0])
        print(f"| {key} | {len(fs)} | {fmt([x[2] for x in fs])} | {fmt([x[0] for x in fs])} | {fmt([x[1] for x in fs])} | "
              f"{int(np.median(last)):,} [{min(last):,}, {max(last):,}] | {stop}/{len(runs)} | {np.median(mix):.0%} [{min(mix):.0%}, {max(mix):.0%}] | "
              f"{fmt(gain) if gain else '-'} | "
              f"{fmt(km) if km else 'missing'} |")
