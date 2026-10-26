"""Export a mini-stud preset tree, the time-step (coarsest) bucket map and card features for the C++ runner."""
import sys, os, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from environments.mini_stud.game import PRESETS, build_tree

preset = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
t = build_tree(PRESETS[preset])
arr = lambda a, dt: np.frombuffer(a, dtype=dt)
np.asarray(arr(t.players, np.int32)).tofile(f"{out}/players.bin")
np.asarray(arr(t.information, np.int32)).tofile(f"{out}/info.bin")
np.asarray(arr(t.edge_start, np.int64)).tofile(f"{out}/start.bin")
np.asarray(arr(t.degree, np.int32)).tofile(f"{out}/degree.bin")
np.asarray(arr(t.edges, np.int32)).tofile(f"{out}/edges.bin")
np.asarray(arr(t.edge_probability, np.float64)).tofile(f"{out}/prob.bin")
np.asarray(arr(t.utility, np.float64)).tofile(f"{out}/util.bin")
counts = np.array([len(a) for a in t.actions], dtype=np.int32); counts.tofile(f"{out}/counts.bin")
np.array(t.owners, dtype=np.int32).tofile(f"{out}/owner.bin")

def time_step(i):
    parts = t.keys[i].split("|")
    street = sum(p.startswith("s") for p in parts) - 1
    bets = parts[-1][1:]
    now = "".join(bets[k + 2] for k in range(0, len(bets), 3) if int(bets[k]) == street)
    prev = "".join(bets[k + 2] for k in range(0, len(bets), 3) if int(bets[k]) < street)
    return (t.owners[i], street, now, tuple(t.actions[i])), prev
groups, prevs = {}, {}
steps = [time_step(i) for i in range(len(t.keys))]
coarse = np.array([groups.setdefault(s, len(groups)) for s, _ in steps], dtype=np.int32)
coarse.tofile(f"{out}/bucket_timestep.bin")
for i in range(len(t.keys)):
    assert counts[i] == counts[np.flatnonzero(coarse == coarse[i])[0]]
# card features for the k-means baseline: own known ranks, own public ranks, opponent public ranks, previous-street betting
ranks = "QKA"
def rank_counts(text):
    return [text.count(r) for r in ranks]
feats = []
prev_ids = {}
for i, key in enumerate(t.keys):
    parts = key.split("|"); d = {p[0]: p[1:] for p in parts if p[0] in "hdro"}
    own_hidden = d["h"]
    if "d" in d: own_hidden = own_hidden.replace(d["d"], "", 1)
    own_pub = d.get("r", ""); opp_pub = d.get("o", ""); own_all = own_hidden
    for p in parts:
        if p.startswith("s"):
            mine, _, theirs = p.split(":", 1)[1].partition("/")
            own_all += mine.rstrip("*")
            if not mine.endswith("*"): own_pub += mine
            if theirs and theirs != "?": opp_pub += theirs
    prev_onehot = [0] * 10
    prev_onehot[prev_ids.setdefault(steps[i][1], len(prev_ids)) % 10] = 1
    feats.append(rank_counts(own_all) + rank_counts(own_pub) + rank_counts(opp_pub) + prev_onehot)
np.array(feats, dtype=np.float64).tofile(f"{out}/features.bin")
meta = dict(preset=preset, nodes=len(t.players), edges=len(t.edges), infosets=len(t.keys), width=t.max_children,
            timestep_buckets=len(groups), feature_dim=len(feats[0]), previous_histories=len(prev_ids))
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1); print(meta)
