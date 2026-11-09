"""Network-training process for ESCHER with --remote-train 1 (sota/remote_train.hpp).

Usage:
  python gpu_trainer.py [--backend torch|numpy] [--device cuda|cpu] [--log FILE] -- TRAINER [ARGS...]
  python gpu_trainer.py --self-test [--device cuda|cpu]
The first form starts TRAINER (aswiss_stud(.exe) train --partition net --algo escher ..., or compare/nn_tree --algo
escher ...) with its stdin and stdout as pipes (--remote-train 1 is appended when missing), keeps mirrors of its
reservoirs and trains its networks: the history value network, the two regret networks, the average-policy network of a
snapshot and the online-LBR monitor network. The trainer's stderr passes through, and its exit code is returned.

The training is that of sota/escher.hpp, step for step: the same initial weights (sent by the trainer), minibatches
drawn uniformly with replacement, the same losses (value: squared error; regret: sum over legal actions of
t'/t * squared error; average policy: t'/t * cross-entropy against sigma), the mean over the batch, and Adam with
beta = (0.9, 0.999), eps = 1e-8 whose first-layer rows are updated only when a batch touches them (the C++ "lazy Adam";
torch.optim.SparseAdam). Only the random batch indices and the floating-point order differ from the C++ version.
Backends: numpy (float32 reference, CPU) and torch (any device; the first layer is an embedding bag over the sparse
inputs). --self-test runs both backends on the same random data, initial weights and batch indices and compares the
trained weights.
"""
import argparse, io, os, struct, subprocess, sys, time, zlib

import numpy as np

HELLO, VALUE_TRAIN, RES_UPDATE, REGRET_TRAIN, AVG_TRAIN = 1, 2, 3, 4, 5
VERSION = 1
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


# ---- protocol --------------------------------------------------------------------------------------------------------
class Pipe:
    def __init__(self, reader, writer):
        self.r, self.w = reader, writer

    def read(self, n):
        data = self.r.read(n)
        if len(data) != n:
            raise EOFError("trainer closed its stdout")
        return data

    def u8(self): return self.read(1)[0]
    def u16(self): return struct.unpack("<H", self.read(2))[0]
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def f32(self): return struct.unpack("<f", self.read(4))[0]
    def floats(self, n): return np.frombuffer(self.read(4 * n), dtype="<f4").astype(np.float32)

    def sparse(self):
        nnz = self.u16()
        idx = np.frombuffer(self.read(4 * nnz), dtype="<i4")
        val = np.frombuffer(self.read(4 * nnz), dtype="<f4")
        return idx, val

    def net(self):
        i, h1, h2, o = struct.unpack("<4i", self.read(16))
        W1 = self.floats(i * h1).reshape(i, h1); b1 = self.floats(h1)
        W2 = self.floats(h1 * h2).reshape(h1, h2); b2 = self.floats(h2)
        W3 = self.floats(h2 * o).reshape(h2, o); b3 = self.floats(o)
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def answer(self, kind, loss, net):
        W1 = net["W1"]
        out = [struct.pack("<If", kind, float(loss)), struct.pack("<4i", W1.shape[0], W1.shape[1], net["W2"].shape[1], net["W3"].shape[1])]
        out += [np.ascontiguousarray(net[k], dtype="<f4").tobytes() for k in NAMES]
        self.w.write(b"".join(out))
        self.w.flush()


# ---- sample stores ---------------------------------------------------------------------------------------------------
class Store:
    """Samples with padded sparse inputs: idx (n, K) int32 (pad = input dim), val (n, K) float32, and targets."""

    def __init__(self, capacity, in_dim, A, value=False, K=64):
        self.cap, self.pad, self.A, self.value, self.K = capacity, in_dim, A, value, K
        self.idx = np.full((capacity, K), in_dim, np.int32); self.val = np.zeros((capacity, K), np.float32)
        if value:
            self.ret = np.zeros(capacity, np.float32)
        else:
            self.mask = np.zeros((capacity, A), bool); self.v = np.zeros((capacity, A), np.float32); self.w = np.zeros(capacity, np.float32)
        self.size = 0
        self.version = 0  # bumped on every change (device copies refresh on change)
        self.changed = set()

    def grow(self, K):
        idx = np.full((self.cap, K), self.pad, np.int32); idx[:, :self.K] = self.idx
        val = np.zeros((self.cap, K), np.float32); val[:, :self.K] = self.val
        self.idx, self.val, self.K = idx, val, K
        self.changed = set(range(self.size)) | self.changed

    def put_input(self, slot, idx, val):
        if len(idx) > self.K: self.grow(max(len(idx), 2 * self.K))
        self.idx[slot] = self.pad; self.val[slot] = 0
        self.idx[slot, :len(idx)] = idx; self.val[slot, :len(val)] = val
        self.size = max(self.size, slot + 1); self.changed.add(slot); self.version += 1


def read_res_update(pipe, stores, A):
    r, n = pipe.u8(), pipe.u32()
    st = stores[r]
    for _ in range(n):
        slot = pipe.u32(); idx, val = pipe.sparse(); mask = pipe.u8(); v = pipe.floats(A); w = pipe.f32()
        st.put_input(slot, idx, val)
        st.mask[slot] = [(mask >> a) & 1 for a in range(A)]; st.v[slot] = v; st.w[slot] = w


# ---- numpy backend ---------------------------------------------------------------------------------------------------
class NumpyBackend:
    name = "numpy"

    def __init__(self, seed):
        self.opt = {}  # key -> (adam state, step)
        self.seed = seed

    def rows(self, key, t, size, batch, steps):
        rng = np.random.default_rng([self.seed & 0xffffffff, zlib.crc32(key.encode()), t])
        return [rng.integers(0, size, batch) for _ in range(steps)]

    def train(self, kind, net, store, steps, batch, lr, t, key, reset, rows=None):
        P = {k: v.copy() for k, v in net.items()}
        pad = P["W1"].shape[0]
        W1p = np.vstack([P["W1"], np.zeros((1, P["W1"].shape[1]), np.float32)])
        if reset or key not in self.opt or self.opt[key][0]["W1"].shape != W1p.shape:
            self.opt[key] = ({k: np.zeros_like(W1p if k == "W1" else P[k]) for k in NAMES},
                             {k: np.zeros_like(W1p if k == "W1" else P[k]) for k in NAMES}, [0])
        M, V, step = self.opt[key]
        rows = rows if rows is not None else self.rows(key, t, store.size, batch, steps)
        loss = 0.0
        for s, r in enumerate(rows):
            idx, val = store.idx[r], store.val[r]
            idx = np.where(idx == store.pad, pad, idx)
            a1 = np.maximum(np.einsum("bk,bkh->bh", val, W1p[idx]) + P["b1"], 0)
            a2 = np.maximum(a1 @ P["W2"] + P["b2"], 0)
            y = a2 @ P["W3"] + P["b3"]
            B = len(r)
            if kind == "value":
                d = y[:, 0] - store.ret[r]
                dy = np.zeros_like(y); dy[:, 0] = 2 * d / B
                loss = float(np.mean(d * d))
            elif kind == "regret":
                m = store.mask[r]; wn = store.w[r] / np.float32(t)
                d = np.where(m, y - store.v[r], 0).astype(np.float32)
                dy = (2 * wn[:, None] * d / B).astype(np.float32)
                loss = float(np.sum(wn[:, None] * d * d) / max(1e-12, np.sum(wn)))
            else:  # average policy: weighted cross-entropy against sigma over the legal actions
                m = store.mask[r]; wn = store.w[r] / np.float32(t)
                z = np.where(m, y, -np.inf); z = z - z.max(1, keepdims=True)
                e = np.where(m, np.exp(z), 0); p = e / e.sum(1, keepdims=True)
                dy = np.where(m, wn[:, None] * (p - store.v[r]) / B, 0).astype(np.float32)
                logp = np.where(m, np.log(np.maximum(p, 1e-30)), 0)
                loss = float(np.mean(wn * -(store.v[r] * logp).sum(1)))
            g = {"W3": a2.T @ dy, "b3": dy.sum(0)}
            d2 = (dy @ P["W3"].T) * (a2 > 0)
            g["W2"] = a1.T @ d2; g["b2"] = d2.sum(0)
            d1 = (d2 @ P["W2"].T) * (a1 > 0)
            g["b1"] = d1.sum(0)
            G = np.zeros_like(W1p)
            np.add.at(G, idx.reshape(-1), (val[..., None] * d1[:, None, :]).reshape(-1, d1.shape[1]))
            touched = np.unique(idx); touched = touched[touched != pad]
            step[0] += 1
            c1, c2 = 1 - BETA1 ** step[0], 1 - BETA2 ** step[0]
            for k in ("b1", "W2", "b2", "W3", "b3"):
                M[k] = BETA1 * M[k] + (1 - BETA1) * g[k]; V[k] = BETA2 * V[k] + (1 - BETA2) * g[k] * g[k]
                P[k] = (P[k] - lr * (M[k] / c1) / (np.sqrt(V[k] / c2) + EPS)).astype(np.float32)
            gr = G[touched]
            M["W1"][touched] = BETA1 * M["W1"][touched] + (1 - BETA1) * gr
            V["W1"][touched] = BETA2 * V["W1"][touched] + (1 - BETA2) * gr * gr
            W1p[touched] -= (lr * (M["W1"][touched] / c1) / (np.sqrt(V["W1"][touched] / c2) + EPS)).astype(np.float32)
        P["W1"] = W1p[:-1].copy()
        return P, loss


# ---- torch backend ---------------------------------------------------------------------------------------------------
class TorchBackend:
    name = "torch"

    def __init__(self, seed, device):
        import torch
        self.torch, self.device, self.seed = torch, torch.device(device), seed
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        self.opt = {}
        self.dev = {}  # id(store) -> (version, tensors)

    def tensors(self, store):
        torch = self.torch
        cached = self.dev.get(id(store))
        if cached and cached[0] == store.version and cached[1]["idx"].shape[1] == store.K:
            return cached[1]
        if cached and cached[1]["idx"].shape[1] == store.K and store.changed:
            T = cached[1]
            rows = np.fromiter(store.changed, dtype=np.int64)
            ri = torch.from_numpy(rows).to(self.device)
            T["idx"].index_copy_(0, ri, torch.from_numpy(store.idx[rows].astype(np.int64)).to(self.device))
            T["val"].index_copy_(0, ri, torch.from_numpy(store.val[rows]).to(self.device))
            if store.value: T["ret"].index_copy_(0, ri, torch.from_numpy(store.ret[rows]).to(self.device))
            else:
                T["mask"].index_copy_(0, ri, torch.from_numpy(store.mask[rows]).to(self.device))
                T["v"].index_copy_(0, ri, torch.from_numpy(store.v[rows]).to(self.device))
                T["w"].index_copy_(0, ri, torch.from_numpy(store.w[rows]).to(self.device))
        else:
            T = {"idx": torch.from_numpy(store.idx.astype(np.int64)).to(self.device), "val": torch.from_numpy(store.val).to(self.device)}
            if store.value: T["ret"] = torch.from_numpy(store.ret).to(self.device)
            else:
                T["mask"] = torch.from_numpy(store.mask).to(self.device)
                T["v"] = torch.from_numpy(store.v).to(self.device); T["w"] = torch.from_numpy(store.w).to(self.device)
        store.changed = set()
        self.dev[id(store)] = (store.version, T)
        return T

    def train(self, kind, net, store, steps, batch, lr, t, key, reset, rows=None):
        torch = self.torch
        F = torch.nn.functional
        dev = self.device
        pad = net["W1"].shape[0]
        W1p = np.vstack([net["W1"], np.zeros((1, net["W1"].shape[1]), np.float32)])
        P = {"W1": torch.nn.Parameter(torch.from_numpy(W1p).to(dev))}
        for k in NAMES[1:]: P[k] = torch.nn.Parameter(torch.from_numpy(net[k].copy()).to(dev))
        dense = [P[k] for k in NAMES[1:]]
        if reset or key not in self.opt or self.opt[key][0]["W1"].shape != P["W1"].shape:
            state = None
        else:
            state = self.opt[key]
        opt_s = torch.optim.SparseAdam([P["W1"]], lr=lr, betas=(BETA1, BETA2), eps=EPS)
        opt_d = torch.optim.Adam(dense, lr=lr, betas=(BETA1, BETA2), eps=EPS)
        if state is not None:  # carry the optimizer state of this network over (warm start)
            opt_s.load_state_dict(state[1]); opt_d.load_state_dict(state[2])
        T = self.tensors(store)
        if rows is None:
            gen = torch.Generator(device=dev); gen.manual_seed((self.seed * 1000003 + zlib.crc32(key.encode()) * 1009 + t) & 0x7fffffffffffffff)
        loss_v = 0.0
        for s in range(steps):
            r = torch.from_numpy(np.asarray(rows[s], np.int64)).to(dev) if rows is not None else \
                torch.randint(0, store.size, (batch,), device=dev, generator=gen)
            idx = T["idx"][r]; idx = torch.where(idx == store.pad, torch.full_like(idx, pad), idx)
            val = T["val"][r]
            a1 = torch.relu(F.embedding_bag(idx, P["W1"], per_sample_weights=val, mode="sum", sparse=True, padding_idx=pad) + P["b1"])
            a2 = torch.relu(a1 @ P["W2"] + P["b2"])
            y = a2 @ P["W3"] + P["b3"]
            if kind == "value":
                d = y[:, 0] - T["ret"][r]
                loss = (d * d).mean()
                report = loss
            elif kind == "regret":
                m = T["mask"][r]; wn = T["w"][r] / float(t)
                d = torch.where(m, y - T["v"][r], torch.zeros_like(y))
                per = (wn[:, None] * d * d).sum(1)
                loss = per.mean()
                report = per.sum() / torch.clamp(wn.sum(), min=1e-12)
            else:
                m = T["mask"][r]; wn = T["w"][r] / float(t)
                logp = torch.log_softmax(y.masked_fill(~m, float("-inf")), dim=1).masked_fill(~m, 0.0)
                per = wn * -(T["v"][r] * logp).sum(1)
                loss = per.mean()
                report = loss
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_s.step(); opt_d.step()
            if s == steps - 1: loss_v = float(report)
        self.opt[key] = ({"W1": P["W1"]}, opt_s.state_dict(), opt_d.state_dict())
        out = {k: P[k].detach().cpu().numpy().astype(np.float32) for k in NAMES}
        out["W1"] = out["W1"][:-1].copy()
        return out, loss_v


# ---- driver ----------------------------------------------------------------------------------------------------------
def serve(backend, cmd, log):
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    pipe = Pipe(io.BufferedReader(proc.stdout, 1 << 20), proc.stdin)
    stores, A, in_dim, hist_dim = None, None, None, None
    if log and os.path.dirname(log): os.makedirs(os.path.dirname(log), exist_ok=True)
    logf = open(log, "a") if log else None
    if logf and logf.tell() == 0: logf.write("request,t,steps,samples,wall_seconds,process_cpu_seconds,loss\n")
    try:
        while True:
            head = pipe.r.read(4)
            if len(head) < 4: break
            kind = struct.unpack("<I", head)[0]
            w0, c0 = time.perf_counter(), time.process_time()
            if kind == HELLO:
                in_dim, hist_dim, h1, h2, A, batch, rcap, scap = (pipe.u32() for _ in range(8))
                seed = pipe.u64()
                backend.seed = seed
                stores = [Store(rcap, in_dim, A), Store(rcap, in_dim, A), Store(scap, in_dim, A)]
                pipe.w.write(struct.pack("<II", HELLO, VERSION)); pipe.w.flush()
                print(f"gpu_trainer: {backend.name} backend, input {in_dim}, history {hist_dim}, hidden {h1}x{h2}, {A} actions",
                      file=sys.stderr, flush=True)
                continue
            if kind == RES_UPDATE:
                read_res_update(pipe, stores, A); continue
            if kind == VALUE_TRAIN:
                t, steps, batch, lr, reset = pipe.u32(), pipe.u32(), pipe.u32(), pipe.f32(), pipe.u8()
                net = pipe.net(); n = pipe.u32()
                st = Store(n, hist_dim, 1, value=True)
                for k in range(n):
                    idx, val = pipe.sparse(); st.put_input(k, idx, val); st.ret[k] = pipe.f32()
                out, loss = backend.train("value", net, st, steps, batch, lr, t, "value", bool(reset))
                samples, req = n, "value"
            elif kind == REGRET_TRAIN:
                seat = pipe.u8(); t, steps, batch, lr, reset = pipe.u32(), pipe.u32(), pipe.u32(), pipe.f32(), pipe.u8()
                net = pipe.net()
                out, loss = backend.train("regret", net, stores[seat], steps, batch, lr, t, f"regret{seat}", bool(reset))
                samples, req = stores[seat].size, f"regret{seat}"
            elif kind == AVG_TRAIN:
                avg_kind = pipe.u8(); t, steps, batch, lr = pipe.u32(), pipe.u32(), pipe.u32(), pipe.f32(); pipe.u64()
                net = pipe.net()
                key = "monitor" if avg_kind == 1 else "average"
                out, loss = backend.train("average", net, stores[2], steps, batch, lr, t, key, avg_kind == 0)
                samples, req = stores[2].size, key
            else:
                raise RuntimeError(f"unknown request {kind}")
            pipe.answer(kind, loss, out)
            if logf:
                logf.write(f"{req},{t},{steps},{samples},{time.perf_counter() - w0:.4f},{time.process_time() - c0:.4f},{loss:.6g}\n"); logf.flush()
    finally:
        if logf: logf.close()
        try: proc.stdin.close()
        except OSError: pass
    return proc.wait()


def self_test(device):
    rng = np.random.default_rng(7)
    in_dim, h1, h2, A, n, B, steps = 300, 64, 32, 8, 4000, 256, 30
    def init():
        return {"W1": rng.normal(0, 0.18, (in_dim, h1)).astype(np.float32), "b1": np.zeros(h1, np.float32),
                "W2": rng.normal(0, 0.18, (h1, h2)).astype(np.float32), "b2": np.zeros(h2, np.float32),
                "W3": (0.1 * rng.normal(0, 0.25, (h2, A))).astype(np.float32), "b3": np.zeros(A, np.float32)}
    st = Store(n, in_dim, A)
    for k in range(n):
        nnz = rng.integers(1, 20)
        st.put_input(k, rng.choice(in_dim, nnz, replace=False).astype(np.int32), rng.normal(0, 1, nnz).astype(np.float32))
        m = rng.random(A) < 0.6; m[rng.integers(A)] = True
        st.mask[k] = m; v = np.where(m, rng.random(A), 0).astype(np.float32)
        st.v[k] = v / v.sum(); st.w[k] = rng.integers(1, 11)
    vs = Store(n, in_dim, 1, value=True)
    for k in range(n):
        nnz = rng.integers(1, 20)
        vs.put_input(k, rng.choice(in_dim, nnz, replace=False).astype(np.int32), rng.normal(0, 1, nnz).astype(np.float32)); vs.ret[k] = rng.normal()
    rows = [rng.integers(0, n, B) for _ in range(steps)]
    nb, tb = NumpyBackend(1), TorchBackend(1, device)
    worst = 0.0
    for kind, store in (("regret", st), ("average", st), ("value", vs)):
        net = init()
        if kind == "value": net["W3"] = net["W3"][:, :1].copy(); net["b3"] = net["b3"][:1].copy()
        a, la = nb.train(kind, net, store, steps, B, 1e-3, 10, kind, True, rows)
        b, lb = tb.train(kind, net, store, steps, B, 1e-3, 10, kind, True, rows)
        diff = max(float(np.max(np.abs(a[k] - b[k])) / (np.max(np.abs(a[k])) + 1e-12)) for k in NAMES)
        moved = max(float(np.max(np.abs(a[k] - net[k]))) for k in NAMES)
        worst = max(worst, diff)
        print(f"{kind:8s} max relative weight difference numpy vs torch({device}) after {steps} steps: {diff:.2e} "
              f"(weights moved by up to {moved:.3g}); last loss {la:.6g} vs {lb:.6g}")
    print("self-test", "PASSED" if worst < 1e-3 else "FAILED", f"(worst {worst:.2e}, threshold 1e-3)")
    return 0 if worst < 1e-3 else 1


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        k = argv.index("--"); cmd = argv[k + 1:]; argv = argv[:k]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("torch", "numpy"), default="torch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test: sys.exit(self_test(a.device))
    if not cmd: ap.error("give the trainer command after --")
    if "--remote-train" not in cmd: cmd += ["--remote-train", "1"]
    backend = TorchBackend(1, a.device) if a.backend == "torch" else NumpyBackend(1)
    sys.exit(serve(backend, cmd, a.log))


if __name__ == "__main__":
    main()
