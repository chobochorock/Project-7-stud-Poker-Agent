"""Pilot: parent-local latent splits with signed outcome-sampling MCCFR.

Usage (project root, existing Torch Python):
    python -m agents.state_action_embedding.refinement_cfr train --out-dir RUN
    python -m agents.state_action_embedding.refinement_cfr evaluate --out-dir RUN
    python -m agents.state_action_embedding.refinement_cfr train --out-dir RUN --resume
Output: family data/RUN/{raw,fixed,split}, resumable checkpoints and gap JSONL.
Metric: fixed original-information-set Local BR-gap, NOT exploitability/LBR.
Limit: IRA-inspired sampled heuristic, NOT original FPIRA/CFR+IRA or a guarantee.
See REFINEMENT_EXPERIMENT.md for the split gate, costs and held-out limitations.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import time

from . import reconstruction_cfr as c
from . import reconstruction_br_gap as gap

np = c.np
WORKER = None
ARMS = ("raw", "fixed", "split")


def key_text(key):
    return ":".join(map(str, key))


class RoutedBuckets:
    """A per-agent view; the frozen encoder/centroids are shared, not copied."""

    def __init__(self, base, rules, seat):
        self.base, self.rules, self.seat = base, rules, seat
        self.depth = 0
        self.features = None

    def encode(self, tokens):
        return self.base.encode(tokens)

    def assign(self, method, features, truth):
        self.features = features
        node = self.base.assign(method, features, truth)
        while key_text((self.seat, self.depth, node)) in self.rules:
            rule = self.rules[key_text((self.seat, self.depth, node))]
            node = (
                rule["left"]
                if features[rule["feature"]] <= rule["threshold"]
                else rule["right"]
            )
        return node


class RefinementAgent(c.BucketAgent):
    def __init__(
        self, game, seat, buckets, method, table, seed, rules, exploration=0.0
    ):
        super().__init__(
            game,
            seat,
            RoutedBuckets(buckets, rules, seat),
            method,
            table,
            seed,
            exploration,
        )
        self.forced_action = None

    def choose_action(self, state, valid_actions):
        self.buckets.depth = len(self.rows)
        action = super().choose_action(state, valid_actions)
        self.rows[-1]["features"] = self.buckets.features
        if self.forced_action is not None:
            action, self.forced_action = self.forced_action, None
            if action not in valid_actions:
                raise ValueError("Illegal intervention")
            self.rows[-1]["action"] = c.r.BETTING_ACTIONS.index(action)
        return action


def init_worker(reference, shared_name=None, shape=None):
    global WORKER
    c.torch.set_num_threads(1)
    memory = shared_memory.SharedMemory(name=shared_name) if shared_name else None
    table = np.ndarray(shape, np.float64, buffer=memory.buf) if memory else None
    if table is not None:
        table.flags.writeable = False
    WORKER = (memory, table, c.FrozenBuckets(reference, 11, 512))


def sample_batch(task):
    indices, method, seed, exploration, rules, collect = task
    _, table, buckets = WORKER
    keys, targets, averages, features, tags = [], [], [], [], []
    decisions = missing = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for hand in indices:
            traverser = hand % 2
            game = c.r.events.EventGame()
            agents = [
                RefinementAgent(
                    game,
                    seat,
                    buckets,
                    method,
                    table,
                    seed + 2 * hand + seat,
                    rules,
                    exploration if seat == traverser else 0.0,
                )
                for seat in (0, 1)
            ]
            result = c.sampling.ppo.play(
                game, {a.name: a for a in agents}, 940_000_000 + seed + hand
            )
            utility = (result["final_chips"][f"p{traverser}"] - c.r.STACK) / 100.0
            rows = agents[traverser].rows
            increments = list(reversed(c.sampling.outcome_targets(rows, utility)))
            avg = c.average_increments(agents[0].rows + agents[1].rows, traverser)
            # A separate hash assigns whole hands, never individual decisions, to halves.
            tag = (
                hashlib.blake2b(f"{seed}:{hand}".encode(), digest_size=1).digest()[0]
                % 2
            )
            for row, increment, av in zip(rows, increments, avg):
                if row["key"][2] < 0:
                    continue
                keys.append(row["key"])
                targets.append(increment["target"])
                averages.append(av)
                if collect:
                    features.append(row["features"])
                    tags.append(tag)
            decisions += sum(len(a.rows) for a in agents)
            missing += sum(a.missing for a in agents)
    return (
        np.asarray(keys, np.int64).reshape(-1, 3),
        np.asarray(targets, np.float64).reshape(-1, 8),
        np.asarray(averages, np.float64).reshape(-1, 8),
        np.asarray(features, np.float32),
        np.asarray(tags, np.uint8),
        decisions,
        missing,
    )


class Monitor:
    """Bounded per-leaf reservoir since birth or its previous completed review."""

    def __init__(self, capacity, seed):
        self.capacity, self.rng = capacity, np.random.default_rng(seed)
        self.rows, self.counts = {}, {}

    def add(self, keys, features, targets, tags):
        for key, x, y, tag in zip(map(tuple, keys), features, targets, tags):
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
            rows = self.rows.setdefault(key, [])
            slot = (
                len(rows)
                if len(rows) < self.capacity
                else int(self.rng.integers(count))
            )
            if slot < self.capacity:
                value = (x.copy(), y.copy(), int(tag))
                if slot == len(rows):
                    rows.append(value)
                else:
                    rows[slot] = value

    def discard(self, key):
        self.rows.pop(key, None)
        self.counts.pop(key, None)

    def arrays(self):
        keys = sorted(self.rows)
        counts = [self.counts[k] for k in keys]
        sizes = [len(self.rows[k]) for k in keys]
        rows = [row for k in keys for row in self.rows[k]]
        return dict(
            monitor_keys=np.asarray(keys, np.int64).reshape(-1, 3),
            monitor_counts=np.asarray(counts, np.int64),
            monitor_offsets=np.r_[0, np.cumsum(sizes)].astype(np.int64),
            monitor_x=np.asarray([v[0] for v in rows], np.float32),
            monitor_y=np.asarray([v[1] for v in rows], np.float64).reshape(-1, 8),
            monitor_tags=np.asarray([v[2] for v in rows], np.uint8),
            monitor_rng=json.dumps(self.rng.bit_generator.state),
        )

    def restore(self, saved):
        self.rng.bit_generator.state = json.loads(saved["monitor_rng"].item())
        for i, key in enumerate(map(tuple, saved["monitor_keys"])):
            lo, hi = saved["monitor_offsets"][i : i + 2]
            self.rows[key] = [
                (x.copy(), y.copy(), int(tag))
                for x, y, tag in zip(
                    saved["monitor_x"][lo:hi],
                    saved["monitor_y"][lo:hi],
                    saved["monitor_tags"][lo:hi],
                )
            ]
            self.counts[key] = int(saved["monitor_counts"][i])


def propose_split(features, targets, tags, legal, min_child=24, z=2.58):
    """Fit one stump on half A; check its action-gain on untouched half B.

    The normal-SE gate is an exploratory noise filter, not a concentration bound
    for adaptive, heavy-tailed MCCFR data. No target clipping/normalization.
    """
    x, y, tags, legal = (
        np.asarray(features),
        np.asarray(targets),
        np.asarray(tags),
        np.asarray(legal, bool),
    )
    if (
        x.ndim != 2
        or y.shape != (len(x), 8)
        or len(tags) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not legal.any()
    ):
        raise ValueError("Invalid split samples")
    train, valid = tags == 0, tags == 1
    if min(train.sum(), valid.sum()) < 2 * min_child:
        return dict(accepted=False, reason="half_support")
    actions = np.flatnonzero(legal)
    a, b = x[train], y[train]
    parent_action = int(actions[b[:, actions].mean(0).argmax()])
    parent_value = b[:, parent_action].mean()
    best = None
    for feature in range(x.shape[1]):
        for threshold in np.unique(np.quantile(a[:, feature], (0.25, 0.5, 0.75))):
            left = a[:, feature] <= threshold
            if min(left.sum(), (~left).sum()) < min_child:
                continue
            left_action = int(actions[b[left][:, actions].mean(0).argmax()])
            right_action = int(actions[b[~left][:, actions].mean(0).argmax()])
            if left_action == right_action:
                continue
            gain = float(
                np.where(left, b[:, left_action], b[:, right_action]).mean()
                - parent_value
            )
            if best is None or gain > best["train_gain"]:
                best = dict(
                    feature=feature,
                    threshold=float(threshold),
                    actions=[left_action, right_action],
                    parent_action=parent_action,
                    train_gain=gain,
                )
    if best is None:
        return dict(accepted=False, reason="no_preference_conflict")
    left = x[valid, best["feature"]] <= best["threshold"]
    if min(left.sum(), (~left).sum()) < min_child:
        return dict(**best, accepted=False, reason="validation_support")
    values = y[valid]
    al, ar = best["actions"]
    delta = np.where(left, values[:, al], values[:, ar]) - values[:, parent_action]
    mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
    conflict = bool(
        (values[left, al] - values[left, ar]).mean() > 0
        and (values[~left, ar] - values[~left, al]).mean() > 0
    )
    accepted = conflict and mean - z * se > 0
    return dict(
        **best,
        accepted=accepted,
        reason="accepted" if accepted else "validation_rejected",
        validation_gain=mean,
        validation_se=se,
        lower_gate=mean - z * se,
        validation_children=[int(left.sum()), int((~left).sum())],
        conflict_reproduced=conflict,
    )


def refine(monitor, rules, table, average, visits, next_id, hand, args, groups):
    reviews, accepted = [], 0
    for key in sorted(list(monitor.rows)):
        rows = monitor.rows[key]
        if len(rows) < args.min_samples:
            continue
        base = key[2]
        # New leaves retain the exact legal partition of their ancestor.
        mask = int(groups[base] % 256)
        legal = (mask & (1 << np.arange(8))) != 0
        result = propose_split(
            [v[0] for v in rows],
            [v[1] for v in rows],
            [v[2] for v in rows],
            legal,
            args.min_child,
            args.z,
        )
        result.update(
            key=[int(v) for v in key],
            hands=hand,
            samples=len(rows),
            visits=monitor.counts[key],
        )
        if result["accepted"] and (
            accepted >= args.max_per_review or len(rules) >= args.max_splits
        ):
            result.update(accepted=False, reason="split_budget")
        if result["accepted"]:
            left, right = next_id, next_id + 1
            next_id += 2
            rules[key_text(key)] = dict(
                feature=result["feature"],
                threshold=result["threshold"],
                left=left,
                right=right,
            )
            groups[left] = groups[right] = groups[base]
            for node in (base, left, right):
                index = (key[0], key[1], node)
                table[index] = 0
                average[index] = 0
                visits[index] = 0
            result.update(children=[left, right])
            accepted += 1
        # Each completed review consumes its confirmation data, even if rejected.
        monitor.discard(key)
        reviews.append(result)
    return reviews, next_id


def save(
    path,
    table,
    average,
    visits,
    hand,
    decisions,
    missing,
    seconds,
    rules,
    groups,
    next_id,
    monitor,
    config_hash,
):
    tmp = path.with_suffix(".tmp.npz")
    replay = monitor.arrays() if path.name == "last.npz" else {}
    np.savez_compressed(
        tmp,
        regret=table,
        average=average,
        visits=visits,
        hands=hand,
        decisions=decisions,
        missing=missing,
        seconds=seconds,
        rules=json.dumps(rules, sort_keys=True),
        groups=groups,
        next_id=next_id,
        config_hash=config_hash,
        **replay,
    )
    os.replace(tmp, path)


def write_row(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, allow_nan=False) + "\n")


def configuration(args):
    old = json.loads(
        (args.baseline_run / "latent/config.json").read_text(encoding="utf-8")
    )
    hashes = dict(old["hashes"])
    hashes[str(Path(__file__).resolve())] = c.r.sha(Path(__file__))
    hashes[str(Path(gap.__file__).resolve())] = c.r.sha(Path(gap.__file__))
    raw_atlas = args.reference / "transformer/seed11/kmeans_raw_512.npz"
    hashes[str(raw_atlas.resolve())] = c.r.sha(raw_atlas)
    for source, digest in hashes.items():
        if c.r.sha(Path(source)) != digest:
            raise ValueError(f"Source/model mismatch: {source}")
    return dict(
        reference=str(args.reference.resolve()),
        baseline_run=str(args.baseline_run.resolve()),
        seed=args.seed,
        hands=args.hands,
        batch=128,
        exploration=0.6,
        interval=args.interval,
        min_samples=args.min_samples,
        reservoir=args.reservoir,
        min_child=args.min_child,
        z=args.z,
        max_splits=args.max_splits,
        max_per_review=args.max_per_review,
        milestones=args.milestones,
        hashes=hashes,
        protocol="signed batched outcome-sampling; uniform H4; conditional parent-local latent stumps; reset split leaves only; no bound guarantee",
    )


def train(args):
    config = configuration(args)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        if (
            not args.resume
            or json.loads(config_path.read_text(encoding="utf-8")) != config
        ):
            raise ValueError(
                "Use a fresh directory, or --resume with identical conditions"
            )
    else:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise FileExistsError("Nonempty output")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        c.r.write_json(config_path, config)
    shape = (2, c.MAX_DECISIONS, 512 + 2 * args.max_splits, 8)
    buckets = c.FrozenBuckets(args.reference, 11, 512)
    memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 8)
    try:
        table = np.ndarray(shape, np.float64, buffer=memory.buf)
        with mp.get_context("spawn").Pool(
            args.workers, init_worker, (str(args.reference), memory.name, shape)
        ) as pool:
            for arm in ARMS:
                directory = args.out_dir / arm
                directory.mkdir(exist_ok=True)
                method = "raw" if arm == "raw" else "latent"
                table.fill(0)
                average, visits = np.zeros(shape), np.zeros(shape[:-1], np.uint64)
                monitor = Monitor(args.reservoir, args.seed + 800_000_003)
                rules, next_id = {}, 512
                groups = np.full(shape[2], -1, np.int64)
                groups[:512] = buckets.atlases[method]["groups"]
                hand = decisions = missing = 0
                elapsed = 0.0
                last = directory / "last.npz"
                if last.exists():
                    saved = c.load_tables(last)
                    if saved["config_hash"].item() != config_hash:
                        raise ValueError("Checkpoint configuration mismatch")
                    table[:], average[:], visits[:] = (
                        saved["regret"],
                        saved["average"],
                        saved["visits"],
                    )
                    hand, decisions, missing = (
                        int(saved[k]) for k in ("hands", "decisions", "missing")
                    )
                    elapsed, next_id = float(saved["seconds"]), int(saved["next_id"])
                    rules, groups = json.loads(saved["rules"].item()), saved["groups"]
                    monitor.restore(saved)
                started, last_log = time.perf_counter(), 0.0
                while hand < args.hands:
                    boundary = min(
                        (hand // args.interval + 1) * args.interval, args.hands
                    )
                    milestones = [v for v in args.milestones if hand < v < boundary]
                    stop = min(hand + 128, min(milestones, default=boundary))
                    chunks = [
                        v.tolist()
                        for v in np.array_split(np.arange(hand, stop), args.workers)
                        if len(v)
                    ]
                    for keys, targets, av, features, tags, count, missed in pool.map(
                        sample_batch,
                        [
                            (chunk, method, args.seed, 0.6, rules, arm == "split")
                            for chunk in chunks
                        ],
                    ):
                        if not np.isfinite(targets).all() or not np.isfinite(av).all():
                            raise FloatingPointError("Nonfinite update")
                        index = tuple(keys.T)
                        np.add.at(table, index, targets)
                        np.add.at(average, index, av)
                        np.add.at(visits, index, 1)
                        if arm == "split":
                            monitor.add(keys, features, targets, tags)
                        decisions += count
                        missing += missed
                    hand = stop
                    seconds = elapsed + time.perf_counter() - started
                    snapshot_args = (
                        table,
                        average,
                        visits,
                        hand,
                        decisions,
                        missing,
                        seconds,
                        rules,
                        groups,
                        next_id,
                        monitor,
                        config_hash,
                    )
                    if hand in args.milestones or hand == args.hands:
                        save(directory / f"checkpoint_{hand:07d}.npz", *snapshot_args)
                    if (
                        arm == "split"
                        and hand % args.interval == 0
                        and hand < args.hands
                    ):
                        # Save the unmodified snapshot before any gate mutates its parent.
                        pre = directory / f"before_review_{hand:07d}.npz"
                        save(pre, *snapshot_args)
                        reviews, next_id = refine(
                            monitor,
                            rules,
                            table,
                            average,
                            visits,
                            next_id,
                            hand,
                            args,
                            groups,
                        )
                        for row in reviews:
                            write_row(directory / "reviews.jsonl", row)
                        n = sum(r["accepted"] for r in reviews)
                        if n:
                            save(
                                directory / f"after_review_{hand:07d}.npz",
                                table,
                                average,
                                visits,
                                hand,
                                decisions,
                                missing,
                                seconds,
                                rules,
                                groups,
                                next_id,
                                monitor,
                                config_hash,
                            )
                            print(
                                f"SPLIT hands={hand} accepted={n} total_splits={len(rules)} reviewed={len(reviews)}",
                                flush=True,
                            )
                        else:
                            pre.unlink()
                    now = time.perf_counter()
                    if (
                        now - last_log >= 30
                        or hand % args.interval == 0
                        or hand == args.hands
                    ):
                        save(
                            last,
                            table,
                            average,
                            visits,
                            hand,
                            decisions,
                            missing,
                            seconds,
                            rules,
                            groups,
                            next_id,
                            monitor,
                            config_hash,
                        )
                        monitor_arrays = monitor.arrays()
                        row = dict(
                            arm=arm,
                            hands=hand,
                            decisions=decisions,
                            missing=missing,
                            splits=len(rules),
                            logical_leaf_keys=2 * c.MAX_DECISIONS * 512 + len(rules),
                            occupied_keys=int(np.count_nonzero(visits)),
                            created_child_ids=next_id - 512,
                            allocated_table_bytes=table.nbytes
                            + average.nbytes
                            + visits.nbytes,
                            monitor_array_bytes=sum(
                                v.nbytes
                                for v in monitor_arrays.values()
                                if isinstance(v, np.ndarray)
                            ),
                            checkpoint_bytes=last.stat().st_size,
                            seconds=seconds,
                        )
                        write_row(directory / "progress.jsonl", row)
                        print(
                            "PROGRESS " + " ".join(f"{k}={v}" for k, v in row.items()),
                            flush=True,
                        )
                        last_log = now
                print(
                    f"TRAIN_COMPLETE arm={arm} hands={hand} splits={len(rules)}",
                    flush=True,
                )
    finally:
        memory.close()
        memory.unlink()


@lru_cache(maxsize=4)
def snapshot(path):
    with np.load(path, allow_pickle=False) as file:
        saved = {key: file[key] for key in ("regret", "average", "rules")}
    return saved, json.loads(saved["rules"].item())


def evaluate_root(task):
    path, arm, hand, phase, policy, index, particles, seed = task
    _, _, buckets = WORKER
    saved, rules = snapshot(path)
    table = saved["regret" if policy == "current" else "average"]
    method = "raw" if arm == "raw" else "latent"
    root_seed = seed + index * 104729
    root, viewer = gap.make_root(root_seed)
    agent = RefinementAgent(root, viewer, buckets, method, table, 0, rules)
    actions = root.get_valid_actions(root.players[viewer])
    agent.choose_action(root.get_ai_state(root.players[viewer], actions), actions)
    record = agent.rows[-1]
    legal, strategy = record["legal"], record["probabilities"]
    values = np.zeros((particles, 8))
    missing = queries = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for particle in range(particles):
            particle_seed = root_seed + 1_000_000_007 + particle * 65537
            hidden = gap.particle_root(root, viewer, particle_seed)
            for action in np.flatnonzero(legal):
                game = copy.deepcopy(hidden)
                agents = {
                    f"p{seat}": RefinementAgent(
                        game,
                        seat,
                        buckets,
                        method,
                        table,
                        particle_seed + 2_000_000_011 + seat,
                        rules,
                    )
                    for seat in (0, 1)
                }
                agents[f"p{viewer}"].forced_action = c.r.BETTING_ACTIONS[action]
                result = gap.finish_from_fifth(game, agents)
                values[particle, action] = (
                    result["final_chips"][f"p{viewer}"] - c.r.STACK
                ) / game.ante
                missing += sum(a.missing for a in agents.values())
                queries += sum(len(a.rows) for a in agents.values())
    return dict(
        arm=arm,
        hands=hand,
        phase=phase,
        policy=policy,
        root=index,
        root_seed=root_seed,
        observation_sha256=hashlib.sha256(
            c.sampling.prefix(root, viewer).tobytes()
        ).hexdigest(),
        root_bucket=int(record["key"][2]),
        particles=particles,
        rollouts=particles * int(legal.sum()),
        policy_queries=queries,
        missing=missing + agent.missing,
        splits=len(rules),
        **gap.gap_statistics(values, strategy, legal),
    )


def evaluate(args):
    config = json.loads((args.out_dir / "config.json").read_text(encoding="utf-8"))
    for source, digest in config["hashes"].items():
        if c.r.sha(Path(source)) != digest:
            raise ValueError(f"Changed source/model: {source}")
    paths, tasks = {}, []
    for arm in ARMS:
        for path in sorted((args.out_dir / arm).glob("*.npz")):
            name = path.stem
            if name.startswith("checkpoint_"):
                phase = "checkpoint"
            elif name.startswith(("before_review_", "after_review_")):
                phase = "before" if name.startswith("before") else "after"
            else:
                continue
            hand = int(name.split("_")[-1])
            saved = c.load_tables(path)
            expected_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            if (
                int(saved["hands"]) != hand
                or saved["config_hash"].item() != expected_hash
            ):
                raise ValueError("Invalid checkpoint metadata")
            paths[str(path.resolve())] = c.r.sha(path)
            roots = args.roots if phase == "checkpoint" else args.split_roots
            for policy in ("current", "average"):
                tasks.extend(
                    (
                        str(path.resolve()),
                        arm,
                        hand,
                        phase,
                        policy,
                        root,
                        args.particles,
                        args.eval_seed,
                    )
                    for root in range(roots)
                )
    meta = dict(
        roots=args.roots,
        split_roots=args.split_roots,
        particles=args.particles,
        seed=args.eval_seed,
        snapshots=paths,
        training_config_hash=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    )
    metadata = args.out_dir / "gap_config.json"
    output = args.out_dir / "gap.jsonl"
    if metadata.exists():
        if not args.resume or json.loads(metadata.read_text(encoding="utf-8")) != meta:
            raise ValueError("Evaluation exists or configuration differs; use --resume")
    else:
        if output.exists():
            raise FileExistsError(output)
        c.r.write_json(metadata, meta)
    finished = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = tuple(row[k] for k in ("arm", "hands", "phase", "policy", "root"))
            if key in finished:
                raise ValueError("Duplicate gap row")
            finished.add(key)
    pending = [task for task in tasks if tuple(task[1:6]) not in finished]
    started, last_log = time.perf_counter(), 0.0
    with mp.get_context("spawn").Pool(
        args.workers, init_worker, (config["reference"],)
    ) as pool:
        for count, row in enumerate(pool.imap(evaluate_root, pending), 1):
            write_row(output, row)
            now = time.perf_counter()
            if now - last_log >= 30 or count == len(pending):
                print(
                    f"GAP completed={len(finished) + count} total={len(tasks)} arm={row['arm']} hands={row['hands']} phase={row['phase']} elapsed_seconds={now-started:.1f}",
                    flush=True,
                )
                last_log = now
    print(f"EVAL_COMPLETE rows={len(tasks)} output={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=c.REFERENCE)
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=c.FAMILY / "data/reconstruction_cfr_20261002",
    )
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=[10000, 50000, 100000]
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interval", type=int, default=10000)
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--reservoir", type=int, default=512)
    parser.add_argument("--min-child", type=int, default=24)
    parser.add_argument("--z", type=float, default=2.58)
    parser.add_argument("--max-splits", type=int, default=32)
    parser.add_argument("--max-per-review", type=int, default=4)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--split-roots", type=int, default=8)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=861003)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if not args.out_dir.is_relative_to((c.FAMILY / "data").resolve()):
        parser.error("Output must stay inside the model family's data/")
    if (
        min(
            args.hands,
            args.seed,
            args.workers,
            args.interval,
            args.min_samples,
            args.min_child,
            args.max_splits,
            args.max_per_review,
            args.roots,
            args.split_roots,
            args.eval_seed,
            *args.milestones,
        )
        < 1
        or args.reservoir < args.min_samples
        or args.min_samples < 4 * args.min_child
        or not np.isfinite(args.z)
        or args.z < 0
        or args.particles < 4
        or args.particles % 2
    ):
        parser.error("Invalid positive budget, split support or even particle count")
    c.torch.set_num_threads(1)
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
