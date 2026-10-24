"""Read-only state-growth/revisit audit of a hand-recall SQLite checkpoint.

Usage: python -m agents.cpp_mccfr.hand_recall_growth --run agents/cpp_mccfr/data/RUN
Output: RUN/revisit_audit_HANDS.json; never trains or modifies model.sqlite.
Counts nonterminal training touches, not terminal nodes or evaluation queries.
One consistent read transaction can briefly delay the writer's WAL checkpoint.
Repeated touches may be within one hand; distinct-hand revisit counts are absent.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct

from .hand_recall_disk import connect, read_manifest


def read_number(handle, format_: str):
    size = struct.calcsize("<" + format_)
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("Truncated checkpoint")
    return struct.unpack("<" + format_, value)[0]


def read_text(handle) -> bytes:
    length = read_number(handle, "I")
    if length > 100000:
        raise ValueError("Oversized checkpoint string")
    value = handle.read(length)
    if len(value) != length:
        raise ValueError("Truncated checkpoint string")
    return value


def add(groups: dict, arm: int, street: int, touches: int) -> None:
    if arm not in (0, 1) or street not in (5, 6, 7) or touches < 1:
        raise ValueError("Invalid arm/street/touches")
    values = groups[(arm, street)]
    values[0] += 1
    values[1] += touches
    values[2] += touches == 1


def summarize(groups: dict) -> list:
    result = []
    for arm in (1, 0):
        for street in (0, 5, 6, 7):
            values = (
                groups[(arm, street)]
                if street
                else [sum(groups[(arm, s)][i] for s in (5, 6, 7)) for i in range(3)]
            )
            states, touches, singletons = values
            result.append(
                dict(
                    arm="recall" if arm else "baseline",
                    street=street or "all",
                    states=states,
                    touches=touches,
                    singletons=singletons,
                    revisited_state_fraction=(states - singletons) / states
                    if states
                    else 0,
                    revisit_event_fraction=1 - states / touches if touches else 0,
                    mean_touches=touches / states if states else 0,
                )
            )
    return result


def source_summary(run: Path, expected_identity: str) -> dict:
    groups = defaultdict(lambda: [0, 0, 0])
    with (run / "checkpoint.bin").open("rb") as handle:
        if (
            read_text(handle) != b"HANDRECALL1"
            or read_text(handle).decode() != expected_identity
        ):
            raise ValueError("Source identity mismatch")
        hands = read_number(handle, "Q")
        handle.read(32)  # Two timers and two committed CSV offsets.
        nodes = {}
        for expected_arm in (1, 0):
            arm = read_number(handle, "B")
            if arm != expected_arm:
                raise ValueError("Source arm mismatch")
            nodes[arm] = read_number(handle, "Q")
            read_text(handle)  # RNG, not a visit statistic.
            count = read_number(handle, "Q")
            for _ in range(count):
                key = read_text(handle)
                payload = handle.read(136)
                if len(key) < 3 or len(payload) != 136:
                    raise ValueError("Truncated source node")
                add(groups, arm, key[1], struct.unpack_from("<Q", payload, 128)[0])
        if handle.read(1):
            raise ValueError("Trailing source bytes")
    return dict(hands=hands, groups=summarize(groups), node_visits=nodes)


def audit(run: Path) -> dict:
    manifest = read_manifest(run)
    groups = defaultdict(lambda: [0, 0, 0])
    points = [[0, 0, 0]]
    with connect(run) as db:
        db.execute("PRAGMA cache_size=-16384")
        db.execute("BEGIN")
        state = io.BytesIO(
            db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0]
        )
        if read_text(state) != b"RECALLDISK1":
            raise ValueError("Unknown disk checkpoint")
        hands = read_number(state, "Q")
        state.read(16)
        expected = {}
        for _ in range(2):
            arm = read_number(state, "B")
            nodes = read_number(state, "Q")
            expected[arm] = dict(nodes=nodes, states=read_number(state, "Q"))
            read_number(state, "Q")
            read_text(state)
        for prefix, value in db.execute(
            "SELECT substr(k,1,3),substr(v,3,8) FROM nodes"
        ):
            if len(prefix) != 3 or len(value) != 8:
                raise ValueError("Invalid disk node")
            add(groups, prefix[0], prefix[2], struct.unpack("<Q", value)[0])
        rows = summarize(groups)
        for arm, name in ((1, "recall"), (0, "baseline")):
            total = next(
                row for row in rows if row["arm"] == name and row["street"] == "all"
            )
            if (
                total["states"] != expected[arm]["states"]
                or total["touches"] > expected[arm]["nodes"]
            ):
                raise ValueError("Counters disagree with snapshot")
        previous = 0
        for (line,) in db.execute(
            "SELECT line FROM metrics WHERE kind=0 ORDER BY hand,root"
        ):
            row = next(csv.reader([line]))
            hand = int(row[0])
            if hand != previous + 1:
                raise ValueError("Non-contiguous hand trace")
            previous = hand
            if hand % 1000 == 0 or hand == hands:
                points.append([hand, int(row[14]), int(row[15])])
        if previous != hands:
            raise ValueError("Trace/snapshot mismatch")
    # Release the WAL reader before scanning the immutable legacy checkpoint.
    old = source_summary(
        Path(manifest["config"]["source_run"]), manifest["config"]["source_identity"]
    )
    intervals = []
    for before, after in zip(points, points[1:]):
        intervals.append(
            dict(
                start=before[0] + 1,
                end=after[0],
                new_states_per_hand={
                    arm: (after[i] - before[i]) / (after[0] - before[0])
                    for i, arm in ((1, "recall"), (2, "baseline"))
                },
            )
        )
    period = []
    for arm in ("recall", "baseline"):
        before = next(
            row for row in old["groups"] if row["arm"] == arm and row["street"] == "all"
        )
        after = next(
            row for row in rows if row["arm"] == arm and row["street"] == "all"
        )
        new_states = after["states"] - before["states"]
        visits = after["touches"] - before["touches"]
        period.append(
            dict(
                arm=arm,
                start=old["hands"] + 1,
                end=hands,
                new_states=new_states,
                training_touches=visits,
                revisit_event_fraction=1 - new_states / visits if visits else None,
            )
        )
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        identity=manifest["identity"],
        hands=hands,
        current=rows,
        source=old,
        cumulative_state_points=points,
        growth_intervals=intervals,
        since_source=period,
        limitation="Touches include same-hand repeats; distinct-hand recurrence is not logged.",
    )


def check() -> None:
    groups = defaultdict(lambda: [0, 0, 0])
    add(groups, 1, 5, 1)
    add(groups, 1, 5, 3)
    add(groups, 1, 7, 2)
    row = summarize(groups)[0]
    assert row["states"] == 3 and row["touches"] == 6
    assert row["revisited_state_fraction"] == 2 / 3
    assert row["revisit_event_fraction"] == 0.5
    assert read_text(io.BytesIO(struct.pack("<I", 3) + b"abc")) == b"abc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    check()
    report = audit(args.run.resolve())
    path = args.run / f"revisit_audit_{report['hands']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
