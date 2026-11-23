"""
Does a small step in the split direction reduce the split player's exploitability IN G?
Compare two label definitions at the pre-split (both-tied) abstract NE:
  avg-label : argmax_a advantage vs opponent's AVERAGE strategy (ASWISS spec, section 4)
  br-label  : argmax_a advantage vs opponent's BEST RESPONSE (in G) to my current strategy
Split step: x_k' = (1-alpha) xbar + alpha e_{label_k}  (children start at parent strategy, move slightly)
"""
import numpy as np
from search_pathology import maximin, unique_strategy, W

def f_P1(xs, As):   # P1 worst-case value in G (higher = less exploitable)
    return sum(W[k] * np.min(xs[k] @ As[k]) for k in range(2))

rng = np.random.default_rng(7)
stats = {"avg": [0, 0], "br": [0, 0]}   # [n_split_triggered, n_exploitability_increased]
alpha = 1e-3
example = None
for it in range(20000):
    m = n = 2 if it % 2 == 0 else 3
    A1 = rng.integers(-3, 4, (m, n)).astype(float); A2 = rng.integers(-3, 4, (m, n)).astype(float)
    As = (A1, A2); Abar = W[0] * A1 + W[1] * A2
    v, x = maximin(Abar); vy, y = maximin(-Abar.T)
    if not (unique_strategy(Abar, v) and unique_strategy(-Abar.T, vy)):
        continue
    # unique BR of P2 in each copy (Danskin needs uniqueness)
    cols = [x @ As[k] for k in range(2)]
    if any(np.sort(c)[1] - np.sort(c)[0] < 1e-6 for c in cols):
        continue
    br = [int(np.argmin(c)) for c in cols]
    labels = {
        "avg": [int(np.argmax(As[k] @ y - x @ As[k] @ y)) for k in range(2)],
        "br":  [int(np.argmax(As[k][:, br[k]] - x @ As[k][:, br[k]])) for k in range(2)],
    }
    adv_ok = {
        "avg": all((As[k] @ y - x @ As[k] @ y).max() > 1e-9 for k in range(2)),
        "br":  all((As[k][:, br[k]] - x @ As[k][:, br[k]]).max() > 1e-9 for k in range(2)),
    }
    base = f_P1((x, x), As)
    for key in ("avg", "br"):
        lab = labels[key]
        if lab[0] == lab[1] or not adv_ok[key]:
            continue                      # no split triggered under this label rule
        xs = [(1 - alpha) * x + alpha * np.eye(m)[lab[k]] for k in range(2)]
        stats[key][0] += 1
        if f_P1(xs, As) < base - 1e-12:
            stats[key][1] += 1
    if example is None and m == 2 and np.allclose(A1, [[3, -3], [-3, 2]]) and np.allclose(A2, [[2, -3], [-3, 2]]):
        example = (labels, br)

for key, (n_split, n_bad) in stats.items():
    print(f"{key}-label: splits triggered {n_split}, P1 exploitability INCREASED after small step in {n_bad} "
          f"({100 * n_bad / max(n_split, 1):.1f}%)")

# counterexample B explicitly
A1 = np.array([[3, -3], [-3, 2]], float); A2 = np.array([[2, -3], [-3, 2]], float); As = (A1, A2)
Abar = W[0] * A1 + W[1] * A2; _, x = maximin(Abar); _, y = maximin(-Abar.T)
br = [int(np.argmin(x @ As[k])) for k in range(2)]
print("\nCounterexample B: xbar =", np.round(x, 4), " P2 BR per copy =", br)
print("  avg-labels:", [int(np.argmax(As[k] @ y)) for k in range(2)],
      " br-labels:", [int(np.argmax(As[k][:, br[k]])) for k in range(2)])
for key, lab in (("avg", [int(np.argmax(As[k] @ y)) for k in range(2)]),
                 ("br", [int(np.argmax(As[k][:, br[k]])) for k in range(2)])):
    for a in (1e-3, 1e-2, 3e-2):
        xs = [(1 - a) * x + a * np.eye(2)[lab[k]] for k in range(2)]
        print(f"  {key}-label step alpha={a}: P1 worst-case value {f_P1((x, x), As):.5f} -> {f_P1(xs, As):.5f}")
