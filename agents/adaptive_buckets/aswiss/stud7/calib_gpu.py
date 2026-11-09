"""GPU calibration of the neural baselines' networks at their 7-stud sizes (runs_fair settings).

Usage (on the machine of the comparison):  python calib_gpu.py [--out calib_gpu.json] [--steps 50]
For each network it times, with PyTorch, (1) one Adam training step at the baseline's batch size and (2) one forward
pass at batch 1 (a traversal's per-node query), on the CPU with 1 thread, on the CPU with all threads, and on CUDA.
Inputs are dense float32 vectors of the feature size (our C++ code uses a sparse first layer; a PyTorch port, like the
official ESCHER and MMD code, would feed dense tensors). Results go to the screen and to --out (JSON), to be combined
with each baseline's share of training time spent in network work (runs_fair logs): ESCHER and MMD spend almost all
of it in network training, ReBeL about 17 % (the rest is subgame solving on the CPU).
"""
import argparse, json, os, platform, time

import torch
from torch import nn

FEATURES = 2 + 3 + 12 * 53 + 24 * 49 + 7 + 8  # kDeepCfrTensorSize (stud_mccfr.cpp): 1832
NETS = {  # name: (inputs, hidden1, hidden2, outputs, training batch)
    "MMD policy 512x512": (FEATURES, 512, 512, 8, 128),
    "ESCHER regret 256x128": (FEATURES, 256, 128, 8, 2048),
    "ESCHER value 256x128": (2 * FEATURES, 256, 128, 1, 2048),
    "ReBeL value 128x128": (110 + 2 * 1326, 128, 128, 2 * 1326, 32),
}


def mlp(i, h1, h2, o):
    return nn.Sequential(nn.Linear(i, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, o))


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize()


def time_net(spec, device, steps):
    i, h1, h2, o, batch = spec
    torch.manual_seed(1)
    net = mlp(i, h1, h2, o).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(batch, i, device=device); y = torch.randn(batch, o, device=device)
    for _ in range(5):
        opt.zero_grad(); nn.functional.mse_loss(net(x), y).backward(); opt.step()
    sync(device); t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); nn.functional.mse_loss(net(x), y).backward(); opt.step()
    sync(device); train_ms = 1e3 * (time.perf_counter() - t0) / steps
    x1 = torch.randn(1, i, device=device)
    with torch.no_grad():
        for _ in range(20): net(x1)
        sync(device); t0 = time.perf_counter()
        for _ in range(steps * 10):
            net(x1); sync(device)  # one query at a time, as a traversal asks
        infer_ms = 1e3 * (time.perf_counter() - t0) / (steps * 10)
    return train_ms, infer_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_gpu.json")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cpu": platform.processor(), "cpu_threads": os.cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps(info))
    setups = [("cpu, 1 thread", torch.device("cpu"), 1), ("cpu, all threads", torch.device("cpu"), os.cpu_count())]
    if torch.cuda.is_available(): setups.append(("cuda", torch.device("cuda"), None))
    results = []
    print(f"{'network':24s} {'device':18s} {'train step ms':>14s} {'batch-1 forward ms':>19s}")
    for name, spec in NETS.items():
        for label, device, threads in setups:
            if threads: torch.set_num_threads(threads)
            train_ms, infer_ms = time_net(spec, device, a.steps)
            results.append({"network": name, "device": label, "train_step_ms": train_ms, "forward_batch1_ms": infer_ms,
                             "inputs": spec[0], "hidden": [spec[1], spec[2]], "outputs": spec[3], "batch": spec[4]})
            print(f"{name:24s} {label:18s} {train_ms:14.3f} {infer_ms:19.4f}")
    with open(a.out, "w") as f: json.dump({"info": info, "results": results}, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
