"""Summarise rule_compare.py output: per rule, the same games (paired by line order within each seed file)."""
import json, sys, glob
from collections import Counter
EPS = 1e-7
prefixes = sys.argv[1:]
for rule in ("maxgain", "all"):
    rows = [json.loads(l) for p in prefixes for l in open(f"{p}_{rule}.jsonl")]
    steps = inc = cons = chains_inc = ended0 = 0
    overshoot = []; macro = Counter(); nsplits = []
    for r in rows:
        s = r["seq"]; steps += len(s) - 1; any_inc = False
        for i in range(1, len(s)):
            if s[i] > s[i - 1] + EPS:
                inc += 1; any_inc = True
                if i >= 2 and s[i - 1] > s[i - 2] + EPS: cons += 1
        chains_inc += any_inc; ended0 += s[-1] <= EPS
        overshoot.append(max(s) - s[0])
        nsplits.append(sum(r["nsplit"]))
        j = 0
        while j < len(s) - 1:
            k = j + 1
            while k < len(s) and s[k] > s[j] + EPS: k += 1
            if k == len(s): macro["unfinished"] += 1; break
            macro[k - j] += 1; j = k
    n = len(rows)
    print(f"{rule:8s} games {n}  split steps {steps} (mean {steps / n:.2f}/game, buckets split {sum(nsplits) / n:.2f}/game)")
    print(f"         steps where the bound rose {inc} ({100 * inc / steps:.1f}% of steps), games with any rise "
          f"{chains_inc} ({100 * chains_inc / n:.1f}%), two rises in a row {cons}")
    print(f"         ended at bound 0: {ended0} ({100 * ended0 / n:.1f}%), mean largest rise above the start "
          f"{sum(max(0, o) for o in overshoot) / n:.4f}, macro-step lengths {dict(sorted(macro.items(), key=str))}")

# per generator / start: games with any rise, mean split steps
from collections import defaultdict
for rule in ("maxgain", "all"):
    agg = defaultdict(lambda: [0, 0, 0])
    for p in prefixes:
        for l in open(f"{p}_{rule}.jsonl"):
            r = json.loads(l); q = r["seq"]; a = agg[(r["kind"], r["start"])]
            a[0] += 1; a[1] += any(q[i] > q[i - 1] + EPS for i in range(1, len(q))); a[2] += len(q) - 1
    print(rule, "  ".join(f"{k}/{s}: rise {v[1]}/{v[0]}, {v[2] / v[0]:.2f} steps" for (k, s), v in sorted(agg.items())))
