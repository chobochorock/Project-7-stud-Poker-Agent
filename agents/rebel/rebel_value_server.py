"""TorchScript scalar-value inference server for recursive ReBeL search."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "PROJECT_LAYOUT.json").is_file()
)
TOY_ROOT = ROOT.parent / "Toy-Card-Game-Agent"
for dependency in (TOY_ROOT / ".deep_cfr_deps", TOY_ROOT / ".open_spiel"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np
import torch


MAGIC = 0x4C564252  # "RBVL" in little-endian bytes.
VERSION = 1
HEADER = struct.Struct("<4I")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    while view:
        received = connection.recv_into(view)
        if not received:
            raise EOFError
        view = view[received:]
    return bytes(data)


def serve(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch has no CUDA support")
    model = torch.jit.load(str(args.model), map_location=device).eval()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            json.dumps({"server": "ready", "kind": "rebel-value", "port": args.port}),
            flush=True,
        )
        connection, address = server.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        batches = rows = 0
        started = time.perf_counter()
        with connection:
            while True:
                try:
                    magic, version, batch, dimensions = HEADER.unpack(
                        receive_exact(connection, HEADER.size)
                    )
                except EOFError:
                    break
                if magic != MAGIC or version != VERSION or not batch or not dimensions:
                    raise RuntimeError("invalid ReBeL value request")
                payload = receive_exact(connection, batch * dimensions * 4)
                array = np.frombuffer(payload, dtype="<f4").reshape(batch, dimensions)
                inputs = torch.from_numpy(array.copy()).to(device)
                with torch.inference_mode():
                    output = model(inputs)
                if output.shape != (batch, 1):
                    raise RuntimeError(
                        f"value model returned {tuple(output.shape)}, expected {(batch, 1)}"
                    )
                values = output.detach().to("cpu", torch.float32).contiguous().numpy()
                connection.sendall(HEADER.pack(MAGIC, VERSION, batch, 1))
                connection.sendall(values.astype("<f4", copy=False).tobytes())
                batches += 1
                rows += batch
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "server": "stopped",
                "client": address[0],
                "batches": batches,
                "rows": rows,
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28741)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or args.threads <= 0:
        parser.error("port and threads must be positive")
    serve(args)


if __name__ == "__main__":
    main()
