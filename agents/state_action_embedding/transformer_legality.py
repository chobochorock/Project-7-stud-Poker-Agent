"""Matched GRU/causal-Transformer rules learning with frozen event encoders.

Usage (project root, PyTorch Python):
    python -m agents.state_action_embedding.transformer_legality --out-dir RUN
    python -m agents.state_action_embedding.transformer_legality --out-dir RUN --inspect
Input: unchanged low-fold corpus, frozen S1 encoders and prior GRU reference.
Output: family data/RUN/{gru,transformer}, checkpoints, latents and comparison.
Metric: legal-only KL + illegal-mass penalty; not poker strength or abstraction.
See TRANSFORMER_LEGALITY.md for initialization, matched budgets and limitations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from . import gru_legality as g
from .experiment import MemoryMeter


MAX_EVENTS = 128
REFERENCE = g.FAMILY / "data/gru_legality_20260927"


class HistoryTransformer(g.nn.Module):
    def __init__(self, input_width: int, width: int = 32):
        super().__init__()
        if width <= 0 or width % 4:
            raise ValueError("Transformer width must be a positive multiple of 4")
        self.project = g.nn.Linear(input_width, width)
        self.layers = g.nn.ModuleList(
            [
                g.nn.TransformerEncoderLayer(
                    width,
                    4,
                    dim_feedforward=2 * width,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )
        self.norm = g.nn.LayerNorm(width)
        self.to_hidden = g.nn.Linear(width, 2 * width)
        self.head = g.nn.Linear(2 * width, len(g.SYMBOLS))
        position = g.torch.arange(MAX_EVENTS).float()[:, None]
        frequency = g.torch.exp(
            g.torch.arange(0, width, 2).float() * (-math.log(10000) / width)
        )
        pe = g.torch.zeros(MAX_EVENTS, width)
        pe[:, 0::2], pe[:, 1::2] = g.torch.sin(position * frequency), g.torch.cos(
            position * frequency
        )
        self.register_buffer("position", pe)

    def forward(self, x):
        length = x.shape[1]
        if not 0 < length <= MAX_EVENTS:
            raise ValueError(
                f"Expected 1..{MAX_EVENTS} events; no truncation permitted"
            )
        hidden = g.torch.tanh(self.project(x)) + self.position[:length]
        future = g.torch.ones(length, length, dtype=g.torch.bool, device=x.device).triu(
            1
        )
        # Right padding is in the masked future of every scored position.
        for layer in self.layers:
            hidden = layer(hidden, src_mask=future)
        hidden = g.torch.tanh(self.to_hidden(self.norm(hidden)))
        return self.head(hidden), hidden


def initialize_transformer(input_width: int, seed: int) -> HistoryTransformer:
    """Match the GRU input projection/head exactly; seed attention separately."""
    g.torch.manual_seed(seed + 600000)
    reference = g.HistoryGRU(input_width)
    g.torch.manual_seed(seed + 700000)
    reference.gru.reset_parameters()
    reference.head.reset_parameters()
    g.torch.manual_seed(seed + 900000)
    model = HistoryTransformer(input_width)
    model.project.load_state_dict(reference.project.state_dict())
    model.head.load_state_dict(reference.head.state_dict())
    return model


def run(args) -> None:
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to((g.FAMILY / "data").resolve()):
        raise ValueError("Output must remain in family data/")
    if args.inspect:
        from .gru_legality_report import inspect

        inspect(args.out_dir / "gru")
        inspect(args.out_dir / "transformer", HistoryTransformer)
        return
    if min(args.steps, args.batch, args.log_every, args.threads) < 1:
        raise ValueError("Budgets must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be distinct")
    reference_config = json.loads(
        (REFERENCE / "config.json").read_text(encoding="utf-8")
    )
    corpus_hash = g.sha(args.corpus / "separate_raw.npz")
    if corpus_hash != reference_config["data_sha256"]:
        raise ValueError("Comparison corpus differs from the GRU reference")
    if (
        g.sha(g.FAMILY / "action_window.py")
        != reference_config["sources"]["action_window.py"]
    ):
        raise ValueError("Pretraining helper source changed")
    pretrained = {
        str(args.pretrained / "S1" / f"seed{s}" / "model.pt"): g.sha(
            args.pretrained / "S1" / f"seed{s}" / "model.pt"
        )
        for s in args.seeds
    }
    for path, value in pretrained.items():
        if value != reference_config["pretrained_sha256"][path]:
            raise ValueError("Frozen checkpoint differs from reference")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    g.torch.set_num_threads(args.threads)
    started = time.perf_counter()
    base = g.prepare(args.corpus)
    data = g.sequence_data(base, args.corpus)
    if data["manifest"]["max_events"] > MAX_EVENTS:
        raise ValueError("Dataset exceeds fixed context bound")
    reference_paths = [
        REFERENCE / c / f"seed{s}" / "model.pt"
        for c in ("raw", "random", "skipgram")
        for s in args.seeds
    ]
    reference_hashes = {str(p): g.sha(p) for p in reference_paths}
    source_paths = [
        Path(__file__),
        g.FAMILY / "gru_legality.py",
        g.FAMILY / "gru_legality_report.py",
        g.FAMILY / "TRANSFORMER_LEGALITY.md",
    ]
    config = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        torch_version=str(g.torch.__version__),
        data_sha256=corpus_hash,
        source_sha256={p.name: g.sha(p) for p in source_paths},
        reference_config=reference_config,
        pretrained_sha256=pretrained,
        reference_sha256=reference_hashes,
        sequence=data["manifest"],
        architecture=dict(
            layers=2,
            d_model=32,
            heads=4,
            feedforward=64,
            norm_first=True,
            dropout=0.0,
            positional="fixed sinusoidal",
            exported_hidden=64,
            max_events=MAX_EVENTS,
        ),
    )
    g.write_json(args.out_dir / "config.json", config)
    for architecture in ("gru", "transformer"):
        folder = args.out_dir / architecture
        folder.mkdir()
        g.write_json(folder / "config.json", config)
    results = []
    for seed in args.seeds:
        for condition in ("raw", "random", "skipgram"):
            pair = []
            # Alternate order across seeds to reduce one-sided warmup effects.
            order = (
                ("gru", "transformer")
                if args.seeds.index(seed) % 2 == 0
                else ("transformer", "gru")
            )
            for architecture in order:
                folder = args.out_dir / architecture / condition / f"seed{seed}"
                with MemoryMeter() as meter:
                    result = g.train_one(
                        args,
                        base,
                        data,
                        condition,
                        seed,
                        folder,
                        initialize_transformer
                        if architecture == "transformer"
                        else None,
                    )
                result.update(architecture=architecture, resources=meter.result())
                if (
                    architecture == "gru"
                    and args.steps == reference_config["arguments"]["steps"]
                    and args.batch == reference_config["arguments"]["batch"]
                ):
                    old = g.torch.load(
                        REFERENCE / condition / f"seed{seed}" / "model.pt",
                        weights_only=True,
                    )
                    new = g.torch.load(folder / "model.pt", weights_only=True)
                    for key, value in old["model"].items():
                        g.torch.testing.assert_close(
                            value, new["model"][key], rtol=0, atol=0
                        )
                    result["prior_gru_weights_exactly_reproduced"] = True
                g.write_json(folder / "metrics.json", result)
                results.append(result)
                pair.append(result)
                g.write_json(args.out_dir / "progress.json", results)
            if pair[0]["event_exposures"] != pair[1]["event_exposures"]:
                raise AssertionError("Training event budgets do not match")
    assert corpus_hash == g.sha(args.corpus / "separate_raw.npz")
    assert pretrained == {p: g.sha(Path(p)) for p in pretrained}
    assert reference_hashes == {p: g.sha(Path(p)) for p in reference_hashes}
    for architecture in ("gru", "transformer"):
        g.write_json(
            args.out_dir / architecture / "summary.json",
            dict(results=[r for r in results if r["architecture"] == architecture]),
        )
    g.write_json(
        args.out_dir / "summary.json",
        dict(
            results=results,
            seconds=time.perf_counter() - started,
            source_data_and_reference_checkpoints_unchanged=True,
        ),
    )
    print(f"Saved comparison: {args.out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--corpus", type=Path, default=g.DEFAULT_CORPUS)
    parser.add_argument("--pretrained", type=Path, default=g.DEFAULT_PRETRAINED)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
