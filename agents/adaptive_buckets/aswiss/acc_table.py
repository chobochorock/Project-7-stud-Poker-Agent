"""Final-row table of the accumulated-regret arms on mini-stud (design sections 18-19).

Usage: python acc_table.py RUNS_DIR > acc_final_table.md
RUNS_DIR holds <ID>_s<seed>/ folders with log.csv and exploitability.csv (eval.py). Every arm below that has runs is
listed, in this order; an arm without runs is listed as "not run" (never left out). Ratios are to the k-means baseline
(kmeans_a), whose bucket counts follow rule a's run, not these arms': an unmatched reference.
"""
import csv, os, statistics, sys

R = sys.argv[1]
ARMS = [
    ("tree_mse", "tree, accumulated MSE, restart (split every 2x)"),
    ("tree_mse_g4", "tree, accumulated MSE, restart, every 4x"),
    ("tree_mse_g10", "tree, accumulated MSE, restart, every 10x"),
    ("tree_mse_frz", "tree, accumulated MSE, restart, frozen after 16.4M"),
    ("tree_mse_gate", "tree, accumulated MSE, restart, split when window regret <= 0.1"),
    ("tree_mse_inh", "tree, accumulated MSE, full inheritance (2x)"),
    ("tree_mse_inh_g10", "tree, accumulated MSE, full inheritance, every 10x"),
    ("tree_mse_inh_frz", "tree, accumulated MSE, full inheritance, frozen after 16.4M"),
    ("tree_dir", "tree, accumulated direction, restart (2x)"),
    ("tree_dir_inh", "tree, accumulated direction, full inheritance (2x)"),
    ("ppacc_dir_inh", "ppacc, direction, full inheritance"),
    ("ppacc_mse_inh", "ppacc, MSE, full inheritance"),
    ("ppacc_gate", "ppacc, MSE, tables restart, split when window regret <= 0.1"),
    ("ppacc_dir_gate", "ppacc, direction, tables restart, split when window regret <= 0.1"),
    ("ppacc_rt", "ppacc, MSE, posterior splits in real time, full inheritance (user's alternative 1)"),
    ("ppacc_rt_restart", "ppacc, MSE, posterior splits in real time, tables restart"),
    ("ppspec_gate", "ppspec (user's specification), alternating judgments, tables restart"),
    ("ppspec_rt", "ppspec, posterior splits in real time, full inheritance (alternative 1)"),
    ("ppspec_leaves", "ppspec, tables are the sums of the leaves' records (alternative 2)"),
    ("pure_a", "pure a, iterates (reference)"),
    ("ppmix_avg_root", "pp mix, avg, root (reference)"),
    ("kmeans_a", "k-means, rule a counts (baseline)"),
    ("mccfr", "MCCFR, no abstraction (reference)"),
]
AT = 16_384_000


def runs(arm):
    out = []
    for s in range(1, 10):
        d = os.path.join(R, f"{arm}_s{s}")
        e = os.path.join(d, "exploitability.csv")
        if os.path.exists(e):
            ex = list(csv.DictReader(open(e)))
            lg = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
            if ex: out.append((ex, lg))
    return out


def med(x): return statistics.median(x) if x else None


def rng(x, f): return f"{f(med(x))} [{f(min(x))}, {f(max(x))}]"


base = runs("kmeans_a")
bav = med([float(ex[-1]["expl_bucket_avg"]) for ex, _ in base]) if base else None
bcu = med([float(ex[-1]["expl_current"]) for ex, _ in base]) if base else None
print("# mini-stud small, CPU 900 s, seeds 1-3: exact exploitability (NashConv/2) at the last row\n")
print("Median over seeds [min, max]. Ratio: to the k-means baseline's median (its bucket counts follow rule a's run, not "
      "these arms': an unmatched reference). Tables: buckets some information set uses at the end. At 16.4M: the row at "
      "16,384,000 iterations (before that judgment). Window regret: the abstract game's average regret over the window "
      "since the last judgment, sum over buckets of max(0, max_a regret gain) / iterations, at the last row (arms that "
      "wait for it only; a judgment happens when it is <= 0.1). Exploitability is not the abstract game's regret.\n")
print("| ID | arm | seeds | iterations (median) | average policy | ratio to k-means | current policy | ratio | tables in use "
      "| average policy at 16.4M | window regret at the end |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for arm, label in ARMS:
    rs = runs(arm)
    if not rs:
        print(f"| `{arm}` | {label} | 0 | not run | | | | | | | |"); continue
    av = [float(ex[-1]["expl_bucket_avg"]) for ex, _ in rs]
    cu = [float(ex[-1]["expl_current"]) for ex, _ in rs]
    tb = [int(float(ex[-1]["buckets_used"] if ex[-1].get("buckets_used") not in (None, "", "nan") else ex[-1]["buckets"]))
          for ex, _ in rs]  # buckets some information set uses (older logs without that column: all buckets)
    it = med([int(ex[-1]["iters"]) for ex, _ in rs])
    at = [float(r["expl_bucket_avg"]) for ex, _ in rs for r in ex if int(r["iters"]) == AT]
    wr = [float(lg[-1]["eps_stat"]) for _, lg in rs if lg[-1].get("eps_stat") not in ("nan", "", None)]
    ra = f"{med(av) / bav:.2f}" if bav else "missing baseline"
    rc = f"{med(cu) / bcu:.2f}" if bcu else "missing baseline"
    print(f"| `{arm}` | {label} | {len(rs)} | {it / 1e6:.0f}M | {rng(av, lambda v: f'{v:.4f}')} | {ra} | "
          f"{rng(cu, lambda v: f'{v:.3f}')} | {rc} | {min(tb):,}–{max(tb):,} | "
          f"{f'{med(at):.4f}' if at else 'no row'} | {f'{med(wr):.3f} [{min(wr):.3f}, {max(wr):.3f}]' if wr else '–'} |")
