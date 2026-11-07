# Run: OPENBLAS_NUM_THREADS=1 python bench_mlp_step_blas.py IN H1 H2 OUT BATCH STEPS
"""Same minibatch Adam step as bench_ours.cpp with numpy + OpenBLAS (one thread): embedding gather, two GEMM layers."""
import sys, time
import numpy as np
inn, h1, h2, out, B, steps = map(int, sys.argv[1:7])
rng = np.random.default_rng(2); f = np.float32
P = [rng.normal(0, np.sqrt(2 / 64), (inn, h1)).astype(f), np.zeros(h1, f), rng.normal(0, np.sqrt(2 / h1), (h1, h2)).astype(f),
     np.zeros(h2, f), (0.1 * rng.normal(0, np.sqrt(2 / h2), (h2, out))).astype(f), np.zeros(out, f)]
M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
def step(t):
    idx = rng.integers(0, inn, B)
    a1 = np.maximum(P[0][idx] + P[1], 0); a2 = np.maximum(a1 @ P[2] + P[3], 0); y = a2 @ P[4] + P[5]
    dy = (y - f(0.1)) / B
    g4 = a2.T @ dy; g5 = dy.sum(0); d2 = (dy @ P[4].T) * (a2 > 0)
    g2 = a1.T @ d2; g3 = d2.sum(0); d1 = (d2 @ P[2].T) * (a1 > 0)
    rows, inv = np.unique(idx, return_inverse=True); g0 = np.zeros((len(rows), h1), f); np.add.at(g0, inv, d1); g1 = d1.sum(0)
    c1 = 1 - 0.9 ** t; c2 = 1 - 0.999 ** t
    for k, g in ((1, g1), (2, g2), (3, g3), (4, g4), (5, g5)):
        M[k] = 0.9 * M[k] + 0.1 * g; V[k] = 0.999 * V[k] + 0.001 * g * g; P[k] -= 1e-3 * (M[k] / c1) / (np.sqrt(V[k] / c2) + 1e-8)
    m = 0.9 * M[0][rows] + 0.1 * g0; v = 0.999 * V[0][rows] + 0.001 * g0 * g0; M[0][rows] = m; V[0][rows] = v
    P[0][rows] -= 1e-3 * (m / c1) / (np.sqrt(v / c2) + 1e-8)
for t in (1, 2): step(t)
t0 = time.perf_counter()
for t in range(3, 3 + steps): step(t)
print(f"numpy+OpenBLAS(1 thread) in={inn} {h1}x{h2} out={out} batch={B}: {1e3 * (time.perf_counter() - t0) / steps:.3f} ms per step")
