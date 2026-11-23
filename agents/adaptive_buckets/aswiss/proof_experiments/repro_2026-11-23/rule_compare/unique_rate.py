"""Share of games (same generators as double_split2.py) whose all-tied start has a non-unique abstract equilibrium
(chain_search.stage returns None), i.e. games the uploaded experiments leave out at the first stage."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from chain_search import stage
rng = np.random.default_rng(int(sys.argv[2])); n = int(sys.argv[3])
for kind in ("rand", "near", "rand3"):
    bad = 0
    for _ in range(n):
        if kind == "rand": K = int(rng.choice([4, 5])); As = [rng.integers(-3, 4, (2, 2)).astype(float) for _ in range(K)]
        elif kind == "near":
            K = int(rng.choice([3, 4, 5])); base = rng.integers(-3, 4, (2, 2))
            As = [np.clip(base + rng.choice([-2, -1, 0, 0, 0, 0, 1, 2], size=(2, 2)), -4, 4).astype(float) for _ in range(K)]
        else: K = 3; As = [rng.integers(-3, 4, (3, 3)).astype(float) for _ in range(3)]
        bad += stage(As, [1 / K] * K, [0] * K, [0] * K) is None
    print(f"{kind}: non-unique at the all-tied start {bad}/{n} ({100 * bad / n:.1f}%)", flush=True)
