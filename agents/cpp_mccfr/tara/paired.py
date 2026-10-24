"""Seed-paired arm-vs-arm statistics for TARA campaigns (same training seed = same deals and k-means arm).

Usage (project root):
    python -m agents.cpp_mccfr.tara.paired --campaign agents/cpp_mccfr/data/tara/<c1> [<c2> ...] --hand 127000 \
        --pairs tara:b2_k64 tara:random tara:kmeans ... --output agents/cpp_mccfr/data/tara/<review>/paired_127000.md
For each pair A:B ("kmeans" = the co-trained k-means arm of A's runs) and each metric, reports the mean ± sample sd
of the per-seed difference A - B, its 95% t interval, the number of seeds with A > B, a two-sided one-sample
t-test p-value and Benjamini-Hochberg / Holm adjusted values over all tests in the file (5 seeds: small
sample; the tests share the k-means comparator and are correlated; "n.s." means inconclusive). Metrics:
  LBR at --hand (deal seeds --lbr-seeds): current, sum_j S_j average, newest-tree average (single-forest arms: = sum)
  Local BR-gap crossfit, mean over the last --window hands up to --hand (current, average)
Positive = A worse (higher LBR / gap). Local BR-gap and LBR are not exploitability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .report import discover, lbr_models, read


def metrics(run: Path, hand: int, window: int, seeds_tag: str, kmeans: bool) -> dict:
    out = {}
    models = {avg: lbr_models(run, avg, seeds_tag) for avg in ("sum", "latest")}
    single = not models["latest"]
    for key, avg, name in (("current", "sum", "lbr_current"), ("average", "sum", "lbr_average_sum"),
                           ("average", "latest", "lbr_average_newest")):
        if kmeans:  # the co-trained k-means policies are the same in every file: compare every output with them
            source = next((m for m in models.values() if str(hand) in m), {})
        else:
            source = models["sum" if single and avg == "latest" else avg] or (models["sum"] if key == "current" else {})
        if str(hand) in source:
            policy = ("baseline_" if kmeans else "model_") + key
            out[name] = source[str(hand)]["policies"][policy]["mean"]
    frame = read(run / "per_hand.csv", ["hands", "tree_current_crossfit", "kmeans_current_crossfit",
                                         "tree_average_crossfit", "kmeans_average_crossfit"])
    frame = frame[frame.hands <= hand].tail(window)
    if len(frame) == window and int(frame.hands.iloc[-1]) == hand:
        for p in ("current", "average"):
            out[f"crossfit_{p}"] = float(frame[f"{'kmeans' if kmeans else 'tree'}_{p}_crossfit"].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, nargs="+", required=True)
    parser.add_argument("--hand", type=int, required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--lbr-seeds", type=int, nargs="+", default=[2026100721, 2026100722])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    runs = discover(args.campaign, None)
    tag = "-".join(map(str, args.lbr_seeds))
    lines = [f"# Seed-paired differences at {args.hand} hands", "",
             f"A - B per training seed (positive = A worse). LBR deal seeds {tag}; crossfit Local BR-gap over the last "
             f"{args.window} hands. Two-sided one-sample t-test on the seed differences; q (BH) and Holm adjust over all "
             "tests below (correlated: shared k-means comparator). With 5 seeds, n.s. means inconclusive.", "",
             "| A | B | metric | mean ± sd | 95% CI | A > B | p | q (BH) | Holm | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    results = []
    for pair in args.pairs:
        a, b = pair.split(":")
        seeds = sorted(set(runs[a]) & (set(runs[a]) if b == "kmeans" else set(runs[b])))
        ma = {s: metrics(runs[a][s], args.hand, args.window, tag, False) for s in seeds}
        mb = {s: metrics(runs[a][s] if b == "kmeans" else runs[b][s], args.hand, args.window, tag, b == "kmeans")
              for s in seeds}
        for name in ("lbr_current", "lbr_average_sum", "lbr_average_newest", "crossfit_current", "crossfit_average"):
            common = [s for s in seeds if name in ma[s] and name in mb[s]]
            if len(common) < 2:
                continue
            d = np.array([ma[s][name] - mb[s][name] for s in common])
            p = float(stats.ttest_1samp(d, 0).pvalue)
            half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
            results.append(dict(a=a, b=b, metric=name, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                ci95=[float(d.mean()) - half, float(d.mean()) + half],
                                a_worse=int((d > 0).sum()), n=len(common), p=p, seeds=common))
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i]["p"])
    bh, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):  # Benjamini-Hochberg step-up
        i = order[rank - 1]
        running = min(running, results[i]["p"] * m / rank)
        bh[i] = running
    holm, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):  # Holm step-down
        running = max(running, min(1.0, results[i]["p"] * (m - rank)))
        holm[i] = running
    for r, q, h in zip(results, bh, holm):
        r["q_bh"], r["p_holm"] = q, h
        lines.append(f"| {r['a']} | {r['b']} | {r['metric']} | {r['mean']:+.3f} ± {r['sd']:.3f} | "
                     f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['a_worse']}/{r['n']} | {r['p']:.3f} | {q:.3f} | "
                     f"{h:.3f} | {','.join(map(str, r['seeds']))} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
