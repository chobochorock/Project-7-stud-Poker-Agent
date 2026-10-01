"""Inspect saved strategy pilots; never overwrite or resume their training.

Usage: python -m agents.state_action_embedding.raw_cfr_ppo_diagnose --run RUN --out-dir NEW
Output: diagnostics.json under this model family's data/. Reconstructs PPO's last
rollout from its pre-update snapshot and fixed seeds; computes fresh gradients.
Limit: new gradient probes are not historical gradient logs or strength tests.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path

from . import raw_cfr_ppo as r

np, torch = r.np, r.torch


def group(name):
    if name.startswith("history.layers."):
        return ".".join(name.split(".")[:3])
    return name.rsplit(".", 1)[0]


def gradient_norms(model):
    sums = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            key = group(name)
            sums[key] = sums.get(key, 0.0) + float(param.grad.double().square().sum())
    return {key: value**0.5 for key, value in sums.items()}


def deltas(initial, final):
    result = {}
    for name, param in final.named_parameters():
        old = dict(initial.named_parameters())[name]
        key = group(name)
        item = result.setdefault(
            key, dict(initial_sq=0.0, delta_sq=0.0, changed=0, count=0)
        )
        item["initial_sq"] += float(old.detach().double().square().sum())
        item["delta_sq"] += float(
            (param.detach() - old.detach()).double().square().sum()
        )
        item["changed"] += int((param != old).sum())
        item["count"] += param.numel()
    for item in result.values():
        item["delta_l2"] = item.pop("delta_sq") ** 0.5
        item["initial_l2"] = item.pop("initial_sq") ** 0.5
    return result


def objective(model, data, rows, cfr_loss="mse"):
    tokens, lengths, legal = r.ppo.batch(data, rows, "cpu")
    scores, value = model.raw_outputs(tokens, lengths)
    if model.mode == "cfr":
        target = torch.from_numpy(data["target"][rows])
        loss = r.regret_loss(scores, target, legal, cfr_loss)
        info = dict(regression_loss=float(loss.detach()), loss_kind=cfr_loss)
    else:
        dist = torch.distributions.Categorical(logits=scores.masked_fill(~legal, -1e9))
        adv = data["advantage"]
        adv = (adv - adv.mean()) / max(float(adv.std()), 1e-8)
        pi_loss, kl, clipped = r.ppo.clipped_surrogate(
            dist.log_prob(torch.from_numpy(data["action"][rows])),
            torch.from_numpy(data["old_log_prob"][rows]),
            torch.from_numpy(adv[rows]),
        )
        target = torch.from_numpy(data["returns"][rows])
        value_loss = 0.5 * r.F.mse_loss(value, target)
        loss = pi_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
        info = dict(
            policy_loss=float(pi_loss.detach()),
            value_loss=float(value_loss.detach()),
            approx_kl=float(kl.detach()),
            clip_fraction=float(clipped.detach()),
        )
    score_gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    assert (score_gradient[~legal] == 0).all()
    return loss, {**info, "illegal_score_gradient_max": 0.0}


def gradient_probe(model, data, seed, batches=8, cfr_loss="mse"):
    rng = np.random.default_rng(seed)
    measurements = []
    for _ in range(batches):
        rows = rng.choice(
            len(data["legal"]), min(64, len(data["legal"])), replace=False
        )
        model.zero_grad(set_to_none=True)
        loss, info = objective(model, data, rows, cfr_loss)
        loss.backward()
        norms = gradient_norms(model)
        total = sum(v * v for v in norms.values()) ** 0.5
        threshold = 1.0 if model.mode == "cfr" else 0.5
        measurements.append(
            dict(
                **info,
                norms=norms,
                total_norm=total,
                clip_multiplier=min(1.0, threshold / (total + 1e-6)),
            )
        )
    model.zero_grad(set_to_none=True)
    return measurements


def panel_predictions(model, panel):
    tokens, lengths, legal = r.tensors(panel)
    with torch.no_grad():
        dist, _ = model.distribution_value(tokens, lengths, legal)
        uniform = legal / legal.sum(1, keepdim=True)
        encoded = model.encode(tokens)
        projected = model.history.project(encoded)[tokens[..., 0] != 0]
        result = dict(
            tv_from_legal_uniform=float((dist.probs - uniform).abs().sum(1).mean() / 2),
            normalized_entropy=float(
                (dist.entropy() / legal.sum(1).float().log()).mean()
            ),
            projection_tanh_saturation=float(
                (projected.tanh().abs() > 0.99).float().mean()
            ),
            max_abs_standardized_raw=float(encoded.abs().max()),
            examples=[],
        )
        for i in (0, len(panel) // 2):
            scores, _ = model.raw_outputs(tokens[i : i + 1], lengths[i : i + 1])
            result["examples"].append(
                dict(
                    panel_row=i,
                    tokens=panel[i]["tokens"].tolist(),
                    legal=panel[i]["legal"].tolist(),
                    scores=scores[0].tolist(),
                    probabilities=dist.probs[i].tolist(),
                )
            )
    return result


def cfr_fit(model, data):
    target = data["target"].astype(np.float64)
    legal = data["legal"]
    row_energy = np.square(target).sum(1) / legal.sum(1)
    top = max(1, int(np.ceil(len(row_energy) * 0.01)))
    result = dict(
        rows=len(target),
        abs_target_quantiles=dict(
            zip(
                ("p50", "p90", "p99", "p999", "max"),
                np.quantile(np.abs(target[legal]), [0.5, 0.9, 0.99, 0.999, 1]).tolist(),
            )
        ),
        zero_target_fraction=float((target[legal] == 0).mean()),
        top_one_percent_rows_share_squared_target=float(
            np.sort(row_energy)[-top:].sum() / row_energy.sum()
        ),
    )
    square_error, pred_sq, target_sq, pred_max = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(target), 128):
            rows = np.arange(start, min(start + 128, len(target)))
            tokens, lengths, mask = r.ppo.batch(data, rows, "cpu")
            scores = model.raw_outputs(tokens, lengths)[0].double().numpy()
            mask = mask.numpy()
            pred_max = max(pred_max, float(np.abs(scores[mask]).max()))
            square_error += float(
                (((scores - target[rows]) ** 2 * mask).sum(1) / mask.sum(1)).sum()
            )
            pred_sq += float(((scores**2 * mask).sum(1) / mask.sum(1)).sum())
            target_sq += float(row_energy[rows].sum())
    result.update(
        prediction_rms=(pred_sq / len(target)) ** 0.5,
        target_rms=(target_sq / len(target)) ** 0.5,
        prediction_abs_max=pred_max,
        mse_over_zero_baseline=square_error / target_sq,
    )
    return result


def ppo_values(model, data):
    values = []
    with torch.no_grad():
        for start in range(0, len(data["legal"]), 128):
            rows = np.arange(start, min(start + 128, len(data["legal"])))
            tokens, lengths, _ = r.ppo.batch(data, rows, "cpu")
            values.extend(model.raw_outputs(tokens, lengths)[1].tolist())
    target, values = data["returns"].astype(float), np.asarray(values)
    return dict(
        rows=len(target),
        target_std=float(target.std()),
        value_std=float(values.std()),
        explained_variance=float(1 - np.var(target - values) / np.var(target)),
        mse=float(np.mean((target - values) ** 2)),
        zero_baseline_mse=float(np.mean(target**2)),
    )


def run(args):
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for p, h in config["source_sha256"].items()
    )
    if not args.out_dir.resolve().is_relative_to((r.FAMILY / "data").resolve()):
        raise ValueError("Output must be under this family's data directory")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    panel = r.legal_panel(512)
    output = dict(
        source_run=str(args.run.resolve()),
        diagnostic_seed=928501,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods={},
        betting_actions=list(r.ppo.BETTING_ACTIONS),
        limits="Fresh gradients; original optimizer state/gradient logs were not saved. No strength evaluation or long training.",
    )
    for method in config.get("methods", ["cfr", "ppo"]):
        for seed in config["seeds"]:
            folder = args.run / f"{method}_seed{seed}"
            initial, final = r.load_model(folder / "initial.pt"), r.load_model(
                folder / "last.pt"
            )
            item = dict(
                parameter_count=sum(p.numel() for p in final.parameters()),
                deltas=deltas(initial, final),
                panel=panel_predictions(final, panel),
            )
            if method == "cfr":
                with np.load(
                    folder / "regret_replay.npz", allow_pickle=False
                ) as archive:
                    data = dict(archive)
                item["fit"] = cfr_fit(final, data)
            else:
                behavior = r.load_model(folder / f"behavior_{config['updates']:04d}.pt")
                hands_before = (config["updates"] - 1) * config["hands"]
                with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                    data = r.ppo_batch(
                        behavior,
                        config["hands"],
                        928000000000 + seed * 1000000 + hands_before,
                        929000000000 + seed * 1000000 + hands_before * 2,
                    )
                item["fit"] = ppo_values(final, data)
                item["behavior_fit"] = ppo_values(behavior, data)
            item["gradients"] = gradient_probe(
                final, data, 928501 + seed, cfr_loss=config.get("cfr_loss", "mse")
            )
            if seed == config["seeds"][0]:
                probe = copy.deepcopy(initial)
                rows = np.arange(min(64, len(data["legal"])))
                optimizer = torch.optim.Adam(
                    probe.parameters(), lr=config["lr"], eps=1e-5
                )
                steps = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = objective(
                        probe, data, rows, config.get("cfr_loss", "mse")
                    )
                    loss.backward()
                    steps.append(gradient_norms(probe))
                    r.nn.utils.clip_grad_norm_(
                        probe.parameters(), 1.0 if method == "cfr" else 0.5
                    )
                    optimizer.step()
                assert steps[0]["history.project"] == 0
                assert steps[1]["history.project"] > 0
                item["zero_head_two_step_probe"] = steps
                item[
                    "zero_head_probe_note"
                ] = "Initial weights on final saved data; isolated graph connectivity check, not historical training replay."
            output["methods"][f"{method}_{seed}"] = item
            print(
                method,
                seed,
                "fit",
                item["fit"],
                "gradient totals",
                [round(x["total_norm"], 4) for x in item["gradients"]],
                flush=True,
            )
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
