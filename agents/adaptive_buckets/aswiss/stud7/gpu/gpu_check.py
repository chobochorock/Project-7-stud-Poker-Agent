"""Can this machine run the GPU versions of the 7-stud baselines? Run it once on the laptop and keep the JSON.

Usage: python agents/adaptive_buckets/aswiss/stud7/gpu/gpu_check.py [--out gpu_check.json] [--quick]
Checks, in order, and stops with an explanation at the first failure:
1. PyTorch imports (a Windows application-control block shows up here as WinError 4551);
2. it is a CUDA build that sees the GPU, and the GPU's architecture is in the build (an RTX 50-series laptop GPU is
   Blackwell, compute capability 12.0 = sm_120, which needs a PyTorch build for CUDA 12.8 or newer);
3. a CUDA kernel actually runs;
then times the network shapes the baselines use, on one CPU thread and on the GPU (milliseconds per call):
- policy / regret networks (MMD, ESCHER): input 1,832 (the engine's Deep CFR tensor), output 8; hidden 128x64 (the CPU
  baselines) and 512x512 (a GPU-sized version); a training step of batch 512 and a forward of batch 4,096;
- ReBeL value networks: input 2,762, output 2,652; hidden 128x128 and 512x512; a forward of the 124 leaves of a 6th-street
  CFR iteration and a training step of batch 32;
- ReBeL's vectorised CFR: one regret-matching + reach update over 373 action rows x 1,035 hole pairs.
The numbers decide which parts are worth moving to the GPU (small networks often are not: launch overhead dominates).
Nothing here touches the training runs; it uses one CPU thread and the GPU for about a minute (--quick: a few seconds).
"""
import argparse, json, os, platform, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="gpu_check.json")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
report = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(), "ok": False}


def done(message):
    report["message"] = message
    with open(a.out, "w") as f: json.dump(report, f, indent=1)
    print("\n" + message + f"\n(saved {a.out})")
    sys.exit(0)


try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20)
    report["nvidia_smi"] = out.stdout.strip() or out.stderr.strip()
except Exception as e:  # no driver tool on PATH
    report["nvidia_smi"] = f"not available: {e}"
print("nvidia-smi:", report["nvidia_smi"])

try:
    import torch
except OSError as e:
    report["torch_error"] = str(e)
    done("PyTorch could not load one of its DLLs. 'WinError 4551' means Windows application control (for example Smart App "
         "Control) blocked it. Options: run the GPU experiments in WSL2 (Ubuntu) with a Linux PyTorch, where the policy does "
         "not apply, or allow the Python installation in the Windows security settings (your decision).")
except ImportError as e:
    report["torch_error"] = str(e)
    done("PyTorch is not installed for this Python. Install a CUDA 12.8+ build (needed for an RTX 50-series GPU), e.g. "
         "python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 (check pytorch.org for the current "
         "command), then rerun.")
report["torch"] = torch.__version__
report["torch_cuda"] = torch.version.cuda
report["cuda_available"] = torch.cuda.is_available()
print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    done("This PyTorch does not see a GPU: " + ("it is a CPU-only build (torch.version.cuda is None); install a CUDA 12.8+ build."
         if torch.version.cuda is None else "CUDA build, but no device: check the NVIDIA driver (nvidia-smi above)."))
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
report.update(gpu=props.name, capability=f"{cap[0]}.{cap[1]}", memory_gb=round(props.total_memory / 2 ** 30, 2), arch_list=arch)
print(f"GPU {props.name}, capability {cap[0]}.{cap[1]}, {report['memory_gb']} GB; build architectures {arch}")
if f"sm_{cap[0]}{cap[1]}" not in arch and not any(x.startswith("compute_") and int(x[8:]) <= cap[0] * 10 + cap[1] for x in arch):
    report["arch_supported"] = False
    done(f"This PyTorch build has no kernels for sm_{cap[0]}{cap[1]}. Install a build for CUDA 12.8 or newer (Blackwell).")
try:
    x = torch.randn(1024, 1024, device=dev); y = (x @ x).sum().item(); torch.cuda.synchronize()
except RuntimeError as e:
    report["kernel_error"] = str(e)
    done("A CUDA kernel failed to run: " + str(e))
report["arch_supported"] = True

torch.set_num_threads(1)
reps = 5 if a.quick else 50


def mlp(n_in, h1, h2, n_out):
    return torch.nn.Sequential(torch.nn.Linear(n_in, h1), torch.nn.ReLU(), torch.nn.Linear(h1, h2), torch.nn.ReLU(),
                               torch.nn.Linear(h2, n_out))


def timed(fn, device):
    for _ in range(3): fn()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def sparse_batch(n, n_in, active, device):  # one-hot-like inputs, as the engine's tensors
    x = torch.zeros(n, n_in, device=device)
    x.scatter_(1, torch.randint(0, n_in, (n, active), device=device), 1.0)
    return x


bench = []
cases = [("policy 1832->128x64->8", (1832, 128, 64, 8), 512, 4096, 40),
         ("policy 1832->512x512->8", (1832, 512, 512, 8), 512, 4096, 40),
         ("ReBeL value 2762->128x128->2652", (2762, 128, 128, 2652), 32, 124, 2000),
         ("ReBeL value 2762->512x512->2652", (2762, 512, 512, 2652), 32, 124, 2000)]
for name, shape, train_batch, infer_batch, active in cases:
    row = {"network": name}
    for device in (torch.device("cpu"), dev):
        net = mlp(*shape).to(device); opt = torch.optim.Adam(net.parameters(), 1e-3)
        xt = sparse_batch(train_batch, shape[0], active, device); yt = torch.randn(train_batch, shape[3], device=device)
        xi = sparse_batch(infer_batch, shape[0], active, device)

        def train_step():
            opt.zero_grad(); loss = ((net(xt) - yt) ** 2).mean(); loss.backward(); opt.step()

        def forward():
            with torch.no_grad(): net(xi)
        row[f"{device.type}_train_ms(batch {train_batch})"] = round(timed(train_step, device), 3)
        row[f"{device.type}_forward_ms(batch {infer_batch})"] = round(timed(forward, device), 3)
    bench.append(row); print(row)
row = {"network": "ReBeL CFR step 373 rows x 1035 pairs"}
for device in (torch.device("cpu"), dev):
    regret = torch.randn(373, 1035, device=device); reach = torch.rand(373, 1035, device=device)

    def cfr_step():
        pos = regret.clamp(min=0); z = pos.sum(0, keepdim=True)
        sigma = torch.where(z > 0, pos / z.clamp(min=1e-30), torch.full_like(pos, 1.0 / 373))
        regret.add_(sigma * reach - 0.5 * regret * 0.01)
    row[f"{device.type}_ms"] = round(timed(cfr_step, device), 3)
bench.append(row); print(row)
report["benchmarks"] = bench
report["ok"] = True
done("GPU PyTorch works. Send this JSON (or commit it) so the GPU baselines can be sized from these timings.")
