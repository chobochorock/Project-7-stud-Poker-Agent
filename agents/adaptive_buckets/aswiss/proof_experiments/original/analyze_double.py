import json, sys, glob
from collections import defaultdict
EPS = 1e-7
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
agg = defaultdict(lambda: dict(chains=0, inc=0, cons=0, full=0, partial=0, last=0, macro_max=0, macro_hist=defaultdict(int)))
examples = defaultdict(list)
for r in rows:
    s = r["seq"]; key = (r["kind"], r["start"]); a = agg[key]; a["chains"] += 1
    for i in range(1, len(s)):
        if s[i] > s[i-1] + EPS:
            a["inc"] += 1
            if i >= 2 and s[i-1] > s[i-2] + EPS:
                a["cons"] += 1; examples["consecutive"].append(r)
            if i + 1 < len(s):
                if s[i+1] <= s[i-1] + EPS: a["full"] += 1
                else: a["partial"] += 1; examples["partial"].append(r)
            else:
                a["last"] += 1
    # adaptive macro-steps
    j = 0
    while j < len(s) - 1:
        k = j + 1
        while k < len(s) and s[k] > s[j] + EPS: k += 1
        if k == len(s): a["macro_hist"]["unfinished"] += 1; break
        L = k - j; a["macro_hist"][L] += 1; a["macro_max"] = max(a["macro_max"], L); j = k
tot = defaultdict(int)
print(f"{'generator/start':22s} chains  incr  consec  full-rec  partial  macro-step length histogram")
for key in sorted(agg):
    a = agg[key]
    for f in ("chains", "inc", "cons", "full", "partial"): tot[f] += a[f]
    hist = dict(sorted(((str(k), v) for k, v in a["macro_hist"].items())))
    print(f"{key[0]+'/'+key[1]:22s} {a['chains']:6d} {a['inc']:5d} {a['cons']:7d} {a['full']:9d} {a['partial']:8d}  {hist}")
print("TOTAL", dict(tot))
for name in ("consecutive", "partial"):
    if examples[name]:
        r = examples[name][0]
        print(f"\nfirst '{name}' example ({r['kind']}/{r['start']}): seq={[round(x,4) for x in r['seq']]} who={r['who']}")
        for k, A in enumerate(r["As"]): print(f"  A{k} = {A}")
