"""Train a 7th-street Deep CFR prototype with C++ traversals and PyTorch."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOY_ROOT = ROOT.parent / "Toy-Card-Game-Agent"
for dependency in (TOY_ROOT / ".deep_cfr_deps", TOY_ROOT / ".open_spiel"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np
import torch
from torch import nn


DIMENSIONS = 1832
ACTIONS = 8
HEADER = struct.Struct("<8sIIIQ")
RECORD_DTYPE = np.dtype([
    ("iteration", "<f4"),
    ("player", "u1"),
    ("kind", "u1"),
    ("legal_mask", "u1"),
    ("reserved", "u1"),
    ("state", "<f4", (DIMENSIONS,)),
    ("target", "<f4", (ACTIONS,)),
])


def make_model(hidden: int, layers: int) -> nn.Module:
    modules: list[nn.Module] = []
    width = DIMENSIONS
    for _ in range(layers):
        modules.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    modules.append(nn.Linear(width, ACTIONS))
    return nn.Sequential(*modules)


def zero_model(hidden: int, layers: int) -> nn.Module:
    model = make_model(hidden, layers)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    return model


class Reservoir:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = np.empty((capacity, DIMENSIONS), dtype=np.float32)
        self.targets = np.empty((capacity, ACTIONS), dtype=np.float32)
        self.iterations = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0

    def add(self, records: np.ndarray, rng: np.random.Generator) -> None:
        for record in records:
            self.seen += 1
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                index = int(rng.integers(0, self.seen))
                if index >= self.capacity:
                    continue
            self.states[index] = record["state"]
            self.targets[index] = record["target"]
            self.iterations[index] = record["iteration"]

    def payload(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_states": self.states[:self.size],
            f"{prefix}_targets": self.targets[:self.size],
            f"{prefix}_iterations": self.iterations[:self.size],
            f"{prefix}_meta": np.array([self.size, self.seen], dtype=np.int64),
        }

    def restore(self, archive: np.lib.npyio.NpzFile, prefix: str) -> None:
        states = archive[f"{prefix}_states"]
        if len(states) > self.capacity:
            raise RuntimeError("saved reservoir exceeds configured capacity")
        self.size, self.seen = map(int, archive[f"{prefix}_meta"])
        self.states[:self.size] = states
        self.targets[:self.size] = archive[f"{prefix}_targets"]
        self.iterations[:self.size] = archive[f"{prefix}_iterations"]


def read_records(path: Path) -> np.memmap:
    with path.open("rb") as source:
        magic, version, dimensions, actions, count = HEADER.unpack(source.read(HEADER.size))
    if not magic.startswith(b"DCFRS1") or version != 1:
        raise RuntimeError(f"invalid sample file: {path}")
    if dimensions != DIMENSIONS or actions != ACTIONS:
        raise RuntimeError("sample tensor schema mismatch")
    expected = HEADER.size + count * RECORD_DTYPE.itemsize
    if path.stat().st_size != expected:
        raise RuntimeError("truncated Deep CFR sample file")
    return np.memmap(
        path, mode="r", offset=HEADER.size, dtype=RECORD_DTYPE, shape=(count,)
    )


def train_model(
    model: nn.Module,
    reservoir: Reservoir,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    if not reservoir.size:
        return float("nan")
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_value = float("nan")
    for _ in range(steps):
        indices = rng.integers(0, reservoir.size, size=min(batch_size, reservoir.size))
        states = torch.from_numpy(reservoir.states[indices]).to(device)
        targets = torch.from_numpy(reservoir.targets[indices]).to(device)
        weights = torch.from_numpy(
            np.sqrt(reservoir.iterations[indices])[:, None]
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(weights * (model(states) - targets)))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    return loss_value


def export_model(model: nn.Module, path: Path, device: torch.device) -> None:
    model.to("cpu").eval()
    torch.jit.trace(model, torch.zeros(1, DIMENSIONS)).save(str(path))
    model.to(device)


def build_generator(executable: Path) -> None:
    subprocess.run([
        "g++", "-O3", "-std=c++17",
        str(ROOT / "cpp_mccfr" / "deep_cfr_traverse.cpp"),
        "-o", str(executable), "-lws2_32",
    ], check=True)


def generate_samples(
    args: argparse.Namespace,
    player: int,
    iteration: int,
    models: list[Path],
    output: Path,
) -> dict[str, object]:
    server_command = [
        sys.executable, "-B", str(ROOT / "deep_cfr_ipc_server.py"),
        "--port", str(args.port),
        "--model-p0", str(models[0]),
        "--model-p1", str(models[1]),
        "--device", args.device,
        "--threads", str(args.threads),
        "--seed", str(args.seed + iteration * 17 + player),
    ]
    server = subprocess.Popen(
        server_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8"
    )
    ready = server.stdout.readline() if server.stdout else ""
    if '"server": "ready"' not in ready:
        server.kill()
        raise RuntimeError(f"inference server failed: {ready.strip()}")
    command = [
        str(args.generator),
        "--port", str(args.port),
        "--traverser", str(player),
        "--traversals", str(args.traversals),
        "--iteration", str(iteration),
        "--ante", str(args.ante),
        "--stack-ante", str(args.stack_ante),
        "--seed", str(args.seed + iteration * 1009 + player),
        "--output", str(output),
    ]
    generated = subprocess.run(command, check=True, capture_output=True, text=True)
    server_tail, _ = server.communicate(timeout=30)
    if server.returncode:
        raise RuntimeError(f"inference server exited {server.returncode}: {server_tail}")
    return json.loads(generated.stdout)


def save_checkpoint(
    run_dir: Path,
    completed_iteration: int,
    advantages: list[nn.Module],
    policy: nn.Module,
    buffers: list[Reservoir],
    strategy: Reservoir,
    rng: np.random.Generator,
) -> None:
    checkpoint_tmp = run_dir / "checkpoint.tmp.pt"
    torch.save({
        "completed_iteration": completed_iteration,
        "advantages": [model.state_dict() for model in advantages],
        "policy": policy.state_dict(),
        "rng_state": rng.bit_generator.state,
    }, checkpoint_tmp)
    os.replace(checkpoint_tmp, run_dir / "checkpoint.pt")

    payload = {}
    payload.update(buffers[0].payload("adv0"))
    payload.update(buffers[1].payload("adv1"))
    payload.update(strategy.payload("strategy"))
    replay_tmp = run_dir / "reservoirs.tmp.npz"
    with replay_tmp.open("wb") as output:
        np.savez(output, **payload)
    os.replace(replay_tmp, run_dir / "reservoirs.npz")


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch build is CPU-only")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.build or not args.generator.exists():
        build_generator(args.generator)

    rng = np.random.default_rng(args.seed)
    advantages = [zero_model(args.hidden, args.layers) for _ in range(2)]
    policy = zero_model(args.hidden, args.layers)
    advantage_buffers = [Reservoir(args.memory_capacity) for _ in range(2)]
    strategy_buffer = Reservoir(args.memory_capacity)
    start_iteration = 1
    checkpoint = args.run_dir / "checkpoint.pt"
    replay = args.run_dir / "reservoirs.npz"
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for model, state in zip(advantages, saved["advantages"]):
            model.load_state_dict(state)
        policy.load_state_dict(saved["policy"])
        rng.bit_generator.state = saved["rng_state"]
        with np.load(replay) as archive:
            advantage_buffers[0].restore(archive, "adv0")
            advantage_buffers[1].restore(archive, "adv1")
            strategy_buffer.restore(archive, "strategy")
        start_iteration = int(saved["completed_iteration"]) + 1

    model_paths = [args.run_dir / f"advantage_p{player}.pt" for player in range(2)]
    for model, path in zip(advantages, model_paths):
        export_model(model, path, device)

    started = time.perf_counter()
    for iteration in range(start_iteration, args.iterations + 1):
        iteration_result: dict[str, object] = {"iteration": iteration, "players": []}
        for player in range(2):
            sample_path = args.run_dir / f"samples_i{iteration}_p{player}.bin"
            generation = generate_samples(
                args, player, iteration, model_paths, sample_path
            )
            records = read_records(sample_path)
            advantage_buffers[player].add(
                records[(records["kind"] == 0) & (records["player"] == player)], rng
            )
            strategy_buffer.add(records[records["kind"] == 1], rng)
            del records
            if not args.keep_samples:
                sample_path.unlink()

            advantages[player] = make_model(args.hidden, args.layers)
            loss = train_model(
                advantages[player], advantage_buffers[player],
                steps=args.advantage_steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, device=device, rng=rng,
            )
            export_model(advantages[player], model_paths[player], device)
            iteration_result["players"].append({
                **generation,
                "advantage_buffer": advantage_buffers[player].size,
                "advantage_seen": advantage_buffers[player].seen,
                "advantage_loss": loss,
            })

        policy = make_model(args.hidden, args.layers)
        policy_loss = train_model(
            policy, strategy_buffer, steps=args.policy_steps,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            device=device, rng=rng,
        )
        export_model(policy, args.run_dir / "policy.pt", device)
        save_checkpoint(
            args.run_dir, iteration, advantages, policy,
            advantage_buffers, strategy_buffer, rng,
        )
        iteration_result.update({
            "strategy_buffer": strategy_buffer.size,
            "strategy_seen": strategy_buffer.seen,
            "policy_loss": policy_loss,
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(json.dumps(iteration_result), flush=True)

    summary = {
        "trainer": "deep-cfr-7th-ipc",
        "completed_iterations": args.iterations,
        "traversals_per_player_iteration": args.traversals,
        "run_dir": str(args.run_dir.resolve()),
        "policy_model": str((args.run_dir / "policy.pt").resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--traversals", type=int, default=1000)
    parser.add_argument("--memory-capacity", type=int, default=20_000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--advantage-steps", type=int, default=500)
    parser.add_argument("--policy-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--port", type=int, default=28731)
    parser.add_argument("--ante", type=int, default=1000)
    parser.add_argument("--stack-ante", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--generator", type=Path,
        default=ROOT / "cpp_mccfr" / "deep_cfr_traverse.exe",
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-samples", action="store_true")
    args = parser.parse_args()
    numeric = (
        args.iterations, args.traversals, args.memory_capacity, args.hidden,
        args.layers, args.batch_size, args.advantage_steps, args.policy_steps,
        args.threads, args.ante, args.stack_ante,
    )
    if min(numeric) <= 0 or args.learning_rate <= 0 or not 0 < args.port < 65536:
        parser.error("counts, sizes, learning rate, and port must be positive")
    run(args)


if __name__ == "__main__":
    main()
