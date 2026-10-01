"""Width scaling and small-data overfitting, with unchanged legality supervision.

Usage (project root, PyTorch Python):
    python -m agents.state_action_embedding.capacity_legality --out-dir RUN
    python -m agents.state_action_embedding.capacity_legality --out-dir RUN --conditions L1 L2 R1 R2 S2
    python -m unittest agents.state_action_embedding.test_capacity_legality -v
Input: existing low-fold 10k corpus and frozen window encoders; default raw/S1.
Output: family data/RUN checkpoints, train/validation curves and manifests.
Metric: conditional-card illegal mass and joint BET metrics, not poker strength.
See CAPACITY_LEGALITY.md for budgets and the one-seed overfit limitation.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import time

from . import transformer_legality as t
from .experiment import MemoryMeter

g = t.g
REFERENCE = g.FAMILY / "data/transformer_legality_20260928"


class WideGRU(g.HistoryGRU):
    def __init__(self, input_width: int):
        g.nn.Module.__init__(self)
        self.project = g.nn.Linear(input_width, 128)
        self.gru = g.nn.GRU(128, 256, batch_first=True)
        self.head = g.nn.Linear(256, len(g.SYMBOLS))


class WideTransformer(t.HistoryTransformer):
    def __init__(self, input_width: int):
        super().__init__(input_width, width=128)


def initialize(input_width: int, seed: int, architecture: str, size: str):
    gru_class = g.HistoryGRU if size == "small" else WideGRU
    transformer_class = t.HistoryTransformer if size == "small" else WideTransformer
    g.torch.manual_seed(seed + 600000)
    reference = gru_class(input_width)
    g.torch.manual_seed(seed + 700000)
    reference.gru.reset_parameters()
    reference.head.reset_parameters()
    if architecture == "gru":
        return reference
    g.torch.manual_seed(seed + 900000)
    model = transformer_class(input_width)
    model.project.load_state_dict(reference.project.state_dict())
    model.head.load_state_dict(reference.head.state_dict())
    return model


def select_hands(data, count: int, seed: int):
    training = g.np.flatnonzero(data["split"] == 0)
    hands = g.np.unique(data["hands"][training])
    if not 0 < count <= len(hands):
        raise ValueError("Hand count must fit the training partition")
    chosen = g.np.sort(
        g.np.random.default_rng(seed + 500000).choice(hands, count, replace=False)
    )
    selected = training[g.np.isin(data["hands"][training], chosen)]
    limited = dict(data)
    limited["split"] = data["split"].copy()
    limited["split"][training] = 2
    limited["split"][selected] = 0
    assert not set(chosen) & set(data["hands"][data["split"] == 1])
    return limited, selected, chosen


def load_reference(base, data, args, condition, architecture, seed, panel):
    folder = REFERENCE / architecture / condition / f"seed{seed}"
    saved = g.torch.load(folder / "model.pt", weights_only=True, map_location="cpu")
    table, encoding = g.input_table(base, args.pretrained, condition, seed)
    for key in ("mean", "std"):
        g.torch.testing.assert_close(
            encoding[key], saved["encoding"][key], rtol=0, atol=0
        )
    model = initialize(saved["input_width"], seed, architecture, "small").eval()
    model.load_state_dict(saved["model"])
    previous = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    valid = g.np.flatnonzero(data["split"] == 1)
    measured = g.evaluate(model, table, data, valid, conditional_cards=True)
    for scope, values in previous["metrics"].items():
        for metric, value in values["hand_macro"].items():
            g.np.testing.assert_allclose(
                measured[scope]["hand_macro"][metric], value, rtol=0, atol=1e-10
            )
    diagnostic = json.loads(
        (REFERENCE / architecture / "diagnostics.json").read_text(encoding="utf-8")
    )
    old_cards = next(
        r for r in diagnostic if r["condition"] == condition and r["seed"] == seed
    )
    expected = old_cards["conditional"]["visible_deal"]["model"]["hand_macro"]
    for metric, value in expected.items():
        g.np.testing.assert_allclose(
            measured["conditional_cards"]["hand_macro"][metric],
            value,
            rtol=0,
            atol=1e-10,
        )
    result = dict(previous)
    result.update(
        metrics=measured,
        train_metrics=g.evaluate(model, table, data, panel, conditional_cards=True),
        reference_checkpoint=str(folder / "model.pt"),
        reused=True,
        original_metrics_reproduced=True,
    )
    return result


def run(args) -> None:
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to((g.FAMILY / "data").resolve()):
        raise ValueError("Output must stay in family data/")
    if (
        min(
            args.full_steps,
            args.overfit_steps,
            args.hands,
            args.batch,
            args.log_every,
            args.threads,
        )
        < 1
    ):
        raise ValueError("Budgets must be positive")
    if args.overfit_seed not in args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Use distinct seeds including the overfit seed")
    canonical = ["S1" if c == "skipgram" else c for c in args.conditions]
    if len(canonical) != len(set(canonical)):
        raise ValueError("Input conditions must be distinct; skipgram aliases S1")
    original = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    if g.sha(args.corpus / "separate_raw.npz") != original["data_sha256"]:
        raise ValueError("Corpus differs from the previous comparison")
    hashes = {str(args.corpus / "separate_raw.npz"): original["data_sha256"]}
    if (
        g.sha(g.FAMILY / "action_window.py")
        != original["reference_config"]["sources"]["action_window.py"]
    ):
        raise ValueError("Pretraining/raw-feature helper differs from reference")
    for seed in args.seeds:
        for window in canonical:
            if window == "raw":
                continue
            checkpoint = args.pretrained / window / f"seed{seed}" / "model.pt"
            hashes[str(checkpoint)] = g.sha(checkpoint)
            if (
                window == "S1"
                and hashes[str(checkpoint)]
                != original["pretrained_sha256"][str(checkpoint)]
            ):
                raise ValueError("Frozen embedding changed")
        for architecture in ("gru", "transformer"):
            for condition in args.conditions:
                if condition not in ("raw", "skipgram"):
                    continue
                path = REFERENCE / architecture / condition / f"seed{seed}" / "model.pt"
                hashes[str(path)] = g.sha(path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    g.torch.set_num_threads(args.threads)
    started = time.perf_counter()
    base = g.prepare(args.corpus)
    data = g.sequence_data(base, args.corpus)
    subsets = {seed: select_hands(data, args.hands, seed) for seed in args.seeds}
    config = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        original_config=original,
        source_sha256={
            p.name: g.sha(p)
            for p in (
                Path(__file__),
                g.FAMILY / "gru_legality.py",
                g.FAMILY / "transformer_legality.py",
            )
        },
        input_sha256=hashes,
        torch_version=str(g.torch.__version__),
        training_hand_panels={
            str(s): values[2].tolist() for s, values in subsets.items()
        },
        widths={"small": 32, "wide": 128},
        preprocessing="Original 9000-hand normalization and frozen window encoders; subset changes sequence-training labels only",
    )
    g.write_json(args.out_dir / "config.json", config)
    results = []
    for regime in ("overfit", "full"):
        seeds = [args.overfit_seed] if regime == "overfit" else args.seeds
        for seed in seeds:
            limited, panel, _ = subsets[seed]
            training_data = limited if regime == "overfit" else data
            args.steps = args.overfit_steps if regime == "overfit" else args.full_steps
            for condition in args.conditions:
                for architecture in ("gru", "transformer"):
                    for size in ("small", "wide"):
                        folder = (
                            args.out_dir
                            / regime
                            / condition
                            / architecture
                            / size
                            / f"seed{seed}"
                        )
                        print(
                            f"START {regime} {condition} {architecture} {size} seed={seed}",
                            flush=True,
                        )
                        if (
                            regime == "full"
                            and condition in ("raw", "skipgram")
                            and size == "small"
                            and args.steps == original["arguments"]["steps"]
                            and args.batch == original["arguments"]["batch"]
                        ):
                            with MemoryMeter() as meter:
                                result = load_reference(
                                    base,
                                    data,
                                    args,
                                    condition,
                                    architecture,
                                    seed,
                                    panel,
                                )
                            folder.mkdir(parents=True)
                            result["reevaluation_resources"] = meter.result()
                        else:
                            with MemoryMeter() as meter:
                                result = g.train_one(
                                    args,
                                    base,
                                    training_data,
                                    condition,
                                    seed,
                                    folder,
                                    model_factory=partial(
                                        initialize, architecture=architecture, size=size
                                    ),
                                    train_panel=panel,
                                    conditional_cards=True,
                                    export_latents=False,
                                )
                            result.update(resources=meter.result(), reused=False)
                        result.update(
                            regime=regime,
                            architecture=architecture,
                            size=size,
                            trained_hands=args.hands if regime == "overfit" else 9000,
                            train_panel_hands=args.hands,
                            folder=str(folder),
                        )
                        g.write_json(folder / "metrics.json", result)
                        results.append(result)
                        g.write_json(args.out_dir / "progress.json", results)
    for regime in ("overfit", "full"):
        for seed in [args.overfit_seed] if regime == "overfit" else args.seeds:
            exposures = {
                r["event_exposures"]
                for r in results
                if r["regime"] == regime and r["seed"] == seed
            }
            assert len(exposures) == 1, "Within-regime event budgets differ"
    assert hashes == {path: g.sha(Path(path)) for path in hashes}
    g.write_json(
        args.out_dir / "summary.json",
        dict(
            results=results,
            seconds=time.perf_counter() - started,
            inputs_unchanged=True,
        ),
    )
    print(f"Saved {args.out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=g.DEFAULT_CORPUS)
    parser.add_argument("--pretrained", type=Path, default=g.DEFAULT_PRETRAINED)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("raw", "skipgram", *g.CONDITIONS),
        default=["raw", "skipgram"],
        help="Frozen window inputs; skipgram is the backward-compatible S1 alias",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--overfit-seed", type=int, default=11)
    parser.add_argument("--hands", type=int, default=64)
    parser.add_argument("--full-steps", type=int, default=1000)
    parser.add_argument("--overfit-steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
