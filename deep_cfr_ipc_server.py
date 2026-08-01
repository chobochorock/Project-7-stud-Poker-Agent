"""Batched PyTorch inference server for the C++ 7-Stud traverser."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path


TOY_ROOT = Path(__file__).resolve().parents[1] / "Toy-Card-Game-Agent"
for dependency in (TOY_ROOT / ".deep_cfr_deps", TOY_ROOT / ".open_spiel"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np
import torch
from torch import nn


MAGIC = 0x52464344  # "DCFR" in little-endian bytes.
VERSION = 1
ACTIONS = 8
HEADER = struct.Struct("<4I")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray(size)
    view = memoryview(chunks)
    while view:
        received = connection.recv_into(view)
        if not received:
            raise EOFError
        view = view[received:]
    return bytes(chunks)


def make_model(input_size: int, hidden: int, layers: int) -> nn.Module:
    modules: list[nn.Module] = []
    width = input_size
    for _ in range(layers):
        modules.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    modules.append(nn.Linear(width, ACTIONS))
    return nn.Sequential(*modules)


def serve(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but the installed PyTorch is CPU-only")

    model: nn.Module | None = None
    seat_models: list[nn.Module | None] | None = None
    input_size = 0
    if args.model:
        model = torch.jit.load(str(args.model), map_location=device).eval()
    elif args.model_p0 or args.model_p1:
        seat_models = [
            torch.jit.load(str(path), map_location=device).eval() if path else None
            for path in (args.model_p0, args.model_p1)
        ]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(json.dumps({"server": "ready", "host": args.host, "port": args.port}), flush=True)
        connection, address = server.accept()
        batches = rows = payload_bytes = 0
        started = time.perf_counter()
        with connection:
            while True:
                try:
                    magic, version, batch_size, dimensions = HEADER.unpack(
                        receive_exact(connection, HEADER.size)
                    )
                except EOFError:
                    break
                if magic != MAGIC or version != VERSION or not batch_size or not dimensions:
                    raise RuntimeError("invalid Deep CFR IPC request")
                if model is None and seat_models is None:
                    input_size = dimensions
                    model = make_model(dimensions, args.hidden, args.layers).to(device).eval()
                elif input_size and dimensions != input_size:
                    raise RuntimeError("input dimension changed during connection")

                size = batch_size * dimensions * 4
                payload = receive_exact(connection, size)
                array = np.frombuffer(payload, dtype="<f4").reshape(batch_size, dimensions)
                inputs = torch.from_numpy(array.copy()).to(device)
                with torch.inference_mode():
                    if seat_models is None:
                        output = model(inputs)
                    else:
                        output = torch.zeros(
                            (batch_size, ACTIONS), device=device, dtype=torch.float32
                        )
                        for seat, seat_model in enumerate(seat_models):
                            selected = torch.nonzero(inputs[:, seat] > 0.5).flatten()
                            if seat_model is not None and selected.numel():
                                output[selected] = seat_model(inputs[selected])
                if output.shape != (batch_size, ACTIONS):
                    raise RuntimeError(f"model returned {tuple(output.shape)}, expected {(batch_size, ACTIONS)}")
                response = output.detach().to("cpu", torch.float32).contiguous().numpy()
                connection.sendall(HEADER.pack(MAGIC, VERSION, batch_size, ACTIONS))
                connection.sendall(response.astype("<f4", copy=False).tobytes())
                batches += 1
                rows += batch_size
                payload_bytes += size + response.nbytes

    elapsed = time.perf_counter() - started
    print(json.dumps({
        "server": "stopped",
        "client": address[0],
        "batches": batches,
        "rows": rows,
        "rows_per_second": rows / max(elapsed, 1e-9),
        "payload_mib_per_second": payload_bytes / max(elapsed, 1e-9) / 2**20,
        "elapsed_seconds": elapsed,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28731)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-p0", type=Path)
    parser.add_argument("--model-p1", type=Path)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or min(args.hidden, args.layers, args.threads) <= 0:
        parser.error("port, hidden size, layers, and threads must be positive")
    serve(args)


if __name__ == "__main__":
    main()
