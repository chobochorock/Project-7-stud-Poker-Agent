"""Export the observation streams of every information set (prior-posterior ASWISS, PRIOR_FEATURES_KO.md).

Usage (any directory): python export_obs.py GAME TREE_DIR
  GAME = small | tiny | small-quarter (mini-stud presets) | kuhn | leduc. TREE_DIR is the directory export.py or
  small/export_small.py wrote for the same game (the information set order must match; this script rebuilds the tree
  and checks the counts).
Writes into TREE_DIR:
  obs_start.bin   int64[infosets + 1]  offsets into obs.bin
  obs.bin         int32 codes  channel << 24 | round << 16 | order << 8 | symbol
  seat.bin        int32[infosets]       the acting player's seat
  constraint.bin  int32[infosets]       id of (player, legal action set): the starting cell of posterior and prior
  obs_meta.json   channel and symbol names, which symbols are cards, counts
The game adapter is the only game-specific part: it turns an information set into (channel, round, order, symbol)
observations from the acting player's point of view. Channels (all relative to the acting player):
  0 PC  private chance (only I saw it)        1 UM  public chance dealt to me
  2 UO  public chance dealt to the opponent   3 UC  public chance shared by both (a board card)
  4 HO  opponent's hidden chance (I know a card arrived, not which)
  5 AM  my private action                     6 PM  my public action      7 PO  the opponent's public action
order = index among the observations of the same kind (chance: channels 0-4, action: 5-7) in the same round, so the
interleaving of my and the opponent's actions is kept.
"""
import json, os, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANNELS = ["PC", "UM", "UO", "UC", "HO", "AM", "PM", "PO"]
CHANCE_CH = {0, 1, 2, 3, 4}


def ministud_streams(preset):
    sys.path.insert(0, str(HERE.parents[2]))
    from environments.mini_stud.game import PRESETS, build_tree
    t = build_tree(PRESETS[preset])
    out = []
    for i, key in enumerate(t.keys):
        parts = key.split("|"); me = int(parts[0][1]); obs = []
        d = {p[0]: p[1:] for p in parts[1:] if p[0] in "hdro"}
        for r in d["h"]: obs.append(("PC", 0, r))
        if "d" in d: obs.append(("AM", 0, d["d"]))
        if "r" in d: obs.append(("PM", 0, d["r"]))
        if "o" in d: obs.append(("PO", 0, d["o"]))
        for p in parts[1:]:
            if p[0] == "s":
                k = int(p[1:p.index(":")]); mine, _, theirs = p.split(":", 1)[1].partition("/")
                rnd = k + 1
                if mine:
                    obs.append(("PC", rnd, mine[0]) if mine.endswith("*") else ("UM", rnd, mine[0]))
                if theirs == "?": obs.append(("HO", rnd, "?"))
                elif theirs: obs.append(("UO", rnd, theirs))
        b = parts[-1][1:]
        for k in range(0, len(b), 3):
            st, actor, code = int(b[k]), int(b[k + 1]), b[k + 2]
            obs.append(("PM" if actor == me else "PO", st + 1, code))
        out.append((me, obs, tuple(t.actions[i])))
    cards = sorted({s for _, o, _ in out for ch, _, s in o if ch in ("PC", "UM", "UO", "UC")})
    return t, out, cards


def small_streams(game):
    sys.path.insert(0, str(HERE / "small"))
    from games import build_tree, RANKS
    t = build_tree(game)
    out = []
    for i, d in enumerate(t.info):
        me = t.owners[i]; obs = [("PC", 0, RANKS[d["private"]])]
        for rnd, actions in enumerate(d["rounds"]):
            if rnd == 1 and d["public"] is not None: obs.append(("UC", 1, RANKS[d["public"]]))
            for actor, a in actions: obs.append(("PM" if actor == me else "PO", rnd, a))
        if d["public"] is not None and len(d["rounds"]) < 2: obs.append(("UC", 1, RANKS[d["public"]]))
        out.append((me, obs, tuple(t.actions[i])))
    cards = sorted({s for _, o, _ in out for ch, _, s in o if ch in ("PC", "UC")})
    return t, out, cards


def main():
    game, tree_dir = sys.argv[1], sys.argv[2]
    if game in ("kuhn", "leduc"): t, streams, cards = small_streams(game)
    else: t, streams, cards = ministud_streams(game)
    counts = np.fromfile(f"{tree_dir}/counts.bin", np.int32)
    if len(counts) != len(streams) or any(counts[i] != len(a) for i, (_, _, a) in enumerate(streams)):
        raise SystemExit(f"{tree_dir} is not the exported tree of {game}")
    symbols = sorted({s for _, o, _ in streams for _, _, s in o})
    sym_id = {s: k for k, s in enumerate(symbols)}
    codes, offsets, seats, cons, cmap = [], [0], [], [], {}
    for me, obs, acts in streams:
        cnt = {}
        for ch, rnd, s in obs:
            kind = 0 if CHANNELS.index(ch) in CHANCE_CH else 1
            o = cnt.get((rnd, kind), 0); cnt[(rnd, kind)] = o + 1
            if rnd > 255 or o > 255: raise SystemExit("round or order out of range")
            codes.append(CHANNELS.index(ch) << 24 | rnd << 16 | o << 8 | sym_id[s])
        offsets.append(len(codes)); seats.append(me)
        cons.append(cmap.setdefault((me, acts), len(cmap)))
    keyset = {(seats[i], tuple(sorted(codes[offsets[i]:offsets[i + 1]])), cons[i]) for i in range(len(seats))}
    np.array(offsets, np.int64).tofile(f"{tree_dir}/obs_start.bin")
    np.array(codes, np.int32).tofile(f"{tree_dir}/obs.bin")
    np.array(seats, np.int32).tofile(f"{tree_dir}/seat.bin")
    np.array(cons, np.int32).tofile(f"{tree_dir}/constraint.bin")
    meta = dict(game=game, channels=CHANNELS, chance_channels=sorted(CHANCE_CH), symbols=symbols,
                card_symbols=[sym_id[c] for c in cards], infosets=len(seats), observations=len(codes),
                constraints=len(cmap), constraint_list=[[p, list(a)] for (p, a) in cmap],
                injective=len(keyset) == len(seats))
    json.dump(meta, open(f"{tree_dir}/obs_meta.json", "w"), indent=1)
    print({k: v for k, v in meta.items() if k != "constraint_list"})
    if not meta["injective"]: raise SystemExit("two information sets share a stream: the adapter loses information")


if __name__ == "__main__":
    main()
