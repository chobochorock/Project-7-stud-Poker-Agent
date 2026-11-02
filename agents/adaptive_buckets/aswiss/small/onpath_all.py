"""onpath.py for every epoch checkpoint of the fixed-abstraction runs (and the unabstracted run) of run_small.sh.

Usage: python onpath_all.py OUT GAME     writes OUT/runs/GAME/<arm>/onpath.csv (iters, rho/2 and rho_P/2 on/off path)
"""
import csv, os, sys
import numpy as np
from onpath import split

OUT, GAME = sys.argv[1], sys.argv[2]
R = f"{OUT}/runs/{GAME}"
for arm in sorted(os.listdir(R)):
    if not (arm == "ref" or arm.startswith("fixed_")): continue
    d = f"{R}/{arm}"
    if os.path.exists(f"{d}/onpath.csv") or not os.path.exists(f"{d}/exploitability.csv"): continue
    bmap = np.fromfile(f"{d}/final_bucket_map.bin", np.int32)  # fixed runs never split
    rows = list(csv.DictReader(open(f"{d}/log.csv")))
    with open(f"{d}/onpath.csv", "w") as f:
        f.write("iters,rho_on_half,rho_off_half,rhoP_on_half,rhoP_off_half,gain_half\n")
        for k, r in enumerate(rows):
            v = split(f"{OUT}/{GAME}", bmap, np.fromfile(f"{d}/policy_bucket_{k}.bin"))
            f.write(r["iters"] + "," + ",".join(f"{x:.9g}" for x in v) + "\n")
    print(arm, "done", flush=True)
