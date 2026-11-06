"""Reference run of the official ESCHER code (github.com/Sandholm-Lab/ESCHER) on an OpenSpiel game, to check our C++
ESCHER (../stud7/sota/escher.hpp through nn_tree.cpp) against it under the same hyperparameters.

Usage: python escher_official_ref.py --escher DIR_WITH_ESCHER_PY --out PREFIX [--game leduc_poker] [--iterations 100]
       [--seed 1] [--check-every 10] [--regret-steps 500] [--value-steps 500] [--avg-steps 10000]
Needs open_spiel and tensorflow. Writes PREFIX.csv: iteration, hands (value + regret traversals so far), nash_conv,
exploitability (= nash_conv / 2, the unit of our eval.py), seconds.
Settings: the paper's ESCHER estimator (importance_sampling=False; the code's default True is the paper's reach-weighting
ablation), value exploration 0.01 (paper p.18, code default), uniform traverser sampling (expl=1.0), 1,000 regret and
1,000 value traversals per iteration and batch 2048 (paper Tables 3-4), default training steps of Table 4 (500 / 500 /
10,000), the code's 256 x 128 networks, learning rate 1e-3 and memories of 1e5, average-policy batch 2048.
"""
import argparse, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--escher", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--game", default="leduc_poker")
ap.add_argument("--iterations", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--check-every", type=int, default=10)
ap.add_argument("--traversals", type=int, default=1000)
ap.add_argument("--value-traversals", type=int, default=1000)
ap.add_argument("--regret-steps", type=int, default=500)
ap.add_argument("--value-steps", type=int, default=500)
ap.add_argument("--avg-steps", type=int, default=10000)
ap.add_argument("--batch", type=int, default=2048)
a = ap.parse_args()

import numpy as np
import tensorflow as tf
import pyspiel
random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
sys.path.insert(0, a.escher)
import ESCHER

game = pyspiel.load_game(a.game)
solver = ESCHER.ESCHERSolver(
    game, policy_network_layers=(256, 128), regret_network_layers=(256, 128), value_network_layers=(256, 128),
    num_iterations=a.iterations, num_traversals=a.traversals, num_val_fn_traversals=a.value_traversals,
    learning_rate=1e-3, batch_size_regret=a.batch, batch_size_value=a.batch, batch_size_average_policy=a.batch,
    memory_capacity=int(1e5), policy_network_train_steps=a.avg_steps, regret_network_train_steps=a.regret_steps,
    value_network_train_steps=a.value_steps, check_exploitability_every=a.check_every, importance_sampling=False,
    val_expl=0.01, expl=1.0, save_policy_weights=False, train_device="cpu", infer_device="cpu")
t0 = time.time()
_, _, convs, nodes = solver.solve(save_path_convs=a.out)
# solve() checks at loop indices 0, k, 2k, ... (after iteration i + 1); hands per iteration: value + 2 x regret traversals
per_iter = a.value_traversals + 2 * a.traversals
with open(a.out + ".csv", "w") as f:
    f.write("iteration,hands,nash_conv,exploitability,nodes,seconds_total\n")
    for j, (c, n) in enumerate(zip(convs, nodes)):
        it = j * a.check_every + 1
        f.write(f"{it},{it * per_iter},{c:.6g},{c / 2:.6g},{n},{time.time() - t0:.1f}\n")
print("saved", a.out + ".csv")
