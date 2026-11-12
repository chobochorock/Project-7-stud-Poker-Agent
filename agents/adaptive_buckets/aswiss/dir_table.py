# Tables of PRIOR_POSTERIOR_DESIGN_KO.md section 17 (CPU 900 s and equal iterations) from the runs of run_pp_dir.sh:
#   python3 dir_table.py OUT/runs > dir_table.md   (memory of the iterates arms from the seed-1 reruns memref_mix_s1, memref_pure_s1)
import csv, json, os, statistics as st, sys
import numpy as np
R = sys.argv[1]
arms = [("pp mix, iterates, root (reference)", "ppeq_a_r4_rnlab"), ("pp mix, iterates, parent", "ppmix_it_parent"),
        ("pp mix, avg, root", "ppmix_avg_root"), ("pp mix, avg, parent", "ppmix_avg_parent"),
        ("pure a, iterates", "pure_a"), ("pure a, avg", "pure_a_avg"),
        ("k-means (rule a counts)", "kmeans_a"), ("MCCFR (no abstraction)", "mccfr")]
def rows(n, s):
    with open(f"{R}/{n}_s{s}/exploitability.csv") as f: return list(csv.DictReader(f))
def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None
def fmt(x, p=4):
    return "" if x is None else f"{x:.{p}g}"
out = ["| arm | iterations (median) | average policy, median [min, max] | current policy, median | posterior buckets in use (median) | prior leaves (median) | misassigned (visit-weighted) | routing tables MB (median) | peak RSS MB (median) |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
eq = {}
marks = [8192000, 32768000, 65536000, 131072000]
for name, n in arms:
    L = [rows(n, s)[-1] for s in (1, 2, 3)]
    g = lambda k: [float(r[k]) if r.get(k, "") not in ("", "nan") else None for r in L]
    avg = g("expl_bucket_avg"); cur = g("expl_current")
    mis = []
    for s in (1, 2, 3):
        p = f"{R}/{n}_s{s}/analysis.json"
        if os.path.exists(p):
            a = json.load(open(p)); mis.append((a.get("share"), a.get("visit_weighted_share")))
    ms = ""
    if mis and all(m[0] is not None for m in mis):
        ms = f"{100*med([m[0] for m in mis]):.1f}% ({100*med([m[1] for m in mis]):.1f}%)"
    used = [len(np.unique(np.fromfile(f"{R}/{n}_s{s}/final_bucket_map.bin", np.int32))) for s in (1, 2, 3)]
    rmb, rss = fmt(med(g('routing_mb')), 3), fmt(med(g('peak_rss_mb')), 3)
    ref = {"ppeq_a_r4_rnlab": "memref_mix", "pure_a": "memref_pure"}.get(n)
    if ref and os.path.exists(f"{R}/{ref}_s1/log.csv"):
        with open(f"{R}/{ref}_s1/log.csv") as f: last = list(csv.DictReader(f))[-1]
        if float(last["train_seconds"]) >= 899.9:
            rmb, rss = fmt(float(last["routing_mb"]), 3) + " (seed 1 rerun)", fmt(float(last["peak_rss_mb"]), 3) + " (seed 1 rerun)"
        else: rmb = rss = "pending (seed 1 rerun still running)"
    pl = int(med(g('prior_leaves')) or 0)
    if not rmb: rmb = rss = "not measured (run predates the column)"
    if not ms: ms = "not computable (parent start frees the old tables)" if "parent" in name else "n/a (no prior)"
    out.append(f"| {name} | {int(med(g('iters'))):,} | {fmt(med(avg))} [{fmt(min(avg))}, {fmt(max(avg))}] | {fmt(med(cur))} | "
               f"{int(med(used)):,} | {pl:,} | {ms} | {rmb} | {rss} |")
    for t in marks:
        v = []
        for s in (1, 2, 3):
            m = [r for r in rows(n, s) if int(r["iters"]) == t]
            v.append(float(m[0]["expl_bucket_avg"]) if m else None)
        eq[(name, t)] = v
print("\n".join(out))
print("\n| arm | " + " | ".join(f"{t:,}" for t in marks) + " |\n|---|" + "---:|" * len(marks))
km = {t: med(eq[("k-means (rule a counts)", t)]) for t in marks}
for name, _ in arms:
    cells = []
    for t in marks:
        v = eq[(name, t)]
        if any(x is None for x in v): r = [x for x in v if x is not None]; cells.append("not reached" if not r else "%s (%d of 3 seeds reached)" % (", ".join(f"{x:.4g}" for x in r), len(r))); continue
        m = med(v); cells.append(f"{m:.4g} [{min(v):.4g}, {max(v):.4g}] ({m/km[t]:.2f})" if km[t] else f"{m:.4g}")
    print(f"| {name} | " + " | ".join(cells) + " |")
