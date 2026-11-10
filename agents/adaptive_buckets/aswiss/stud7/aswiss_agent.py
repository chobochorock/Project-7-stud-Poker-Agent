"""Heads-up EV-mode agent backed by a frozen 7-stud ASWISS snapshot (aswiss_stud serve).

The C++ process speaks the line protocol of stud_mccfr --agent-stdio (agent/cpp_mccfr_agent.py) plus EXPLAIN, which
returns the policy behind each decision (probabilities over the legal actions, the bucket-history key, the bucket it
resolves to and the bucket's split genealogy). Every decision goes through EXPLAIN, so the web UI can show why the
agent acted; `analyze` asks the same model about any player's decision (strategy analysis for a human seat).

Files (environment variables override):
  ASWISS_EXE       the trainer built with the serve mode (default stud7/aswiss_web.exe, then aswiss_stud.exe)
  ASWISS_ATLAS     the atlas of the run (default stud7/atlas/atlas_power1024_heur_eps20_seed37301.bin)
  ASWISS_SNAPSHOT  the snapshot to play (default: the newest snapshot listed in ASWISS_RUN_DIR/snapshots.csv)
  ASWISS_RUN_DIR   default stud7/data/runs_fair/history_bbf_street_s1
  ASWISS_PARTITION default history (the partition the snapshot was trained with)
  ASWISS_BUCKET_REGRET  each bucket's regret per visit and confidence (aswiss_stud bucket-regret); default
                   SNAPSHOT.regret.csv when it exists, else none. EXPLAIN then carries "regret" (visits, regret per
                   visit, confidence = r0 / (r0 + regret)); "none" turns it off.
The C++ side refuses a request whose hand it cannot replay from the 5th-street root (the bucket-history key depends on
the whole hand); the agent then plays the heuristic agent's action and records the error, instead of stopping the
game.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Sequence

from agent.base import BasePokerAgent
from agent.cpp_mccfr_agent import ROOT, CppMCCFRAgent
from agent.heuristic_agent import HeuristicPokerAgent


STUD7 = ROOT / "agents/adaptive_buckets/aswiss/stud7"
EXE_NAMES = ("aswiss_web.exe", "aswiss_stud.exe") if os.name == "nt" else ("aswiss_web", "aswiss_stud")
DEFAULT_ATLAS = STUD7 / "atlas/atlas_power1024_heur_eps20_seed37301.bin"
DEFAULT_RUN_DIR = STUD7 / "data/runs_fair/history_bbf_street_s1"
STREETS = {"5th", "6th", "7th_hidden"}


def newest_snapshot(run_dir: Path) -> Path:
    """The snapshot with the most training roots in run_dir/snapshots.csv."""
    index = run_dir / "snapshots.csv"
    if not index.is_file():
        raise FileNotFoundError(f"{index} (set ASWISS_SNAPSHOT to a snapshot file)")
    rows = [row for row in csv.DictReader(open(index)) if (run_dir / row["file"]).is_file()]
    if not rows:
        raise FileNotFoundError(f"no snapshot file of {index} exists")
    return run_dir / max(rows, key=lambda row: int(row["roots"]))["file"]


class ASWISSAgent(BasePokerAgent):
    """Persistent bridge to `aswiss_stud serve` for heads-up EV games."""

    def __init__(
        self,
        name: str,
        *,
        exe: str | os.PathLike[str] | None = None,
        atlas: str | os.PathLike[str] | None = None,
        snapshot: str | os.PathLike[str] | None = None,
        partition: str | None = None,
        policy: str = "average",
        seed: int | None = None,
    ):
        super().__init__(name)
        if exe or os.environ.get("ASWISS_EXE"):
            exe_path = Path(exe or os.environ["ASWISS_EXE"])
        else:
            exe_path = next((STUD7 / n for n in EXE_NAMES if (STUD7 / n).is_file()), STUD7 / EXE_NAMES[0])
        atlas_path = Path(atlas or os.environ.get("ASWISS_ATLAS", DEFAULT_ATLAS))
        if snapshot or os.environ.get("ASWISS_SNAPSHOT"):
            snapshot_path = Path(snapshot or os.environ["ASWISS_SNAPSHOT"])
        else:
            snapshot_path = newest_snapshot(Path(os.environ.get("ASWISS_RUN_DIR", DEFAULT_RUN_DIR)))
        for path in (exe_path, atlas_path, snapshot_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if seed is None:
            try:
                seed = 7 + int(name.rsplit("_", 1)[-1])
            except ValueError:
                seed = 7
        self.snapshot_path = snapshot_path
        regret_env = os.environ.get("ASWISS_BUCKET_REGRET", "")
        if regret_env.lower() == "none":
            regret_path = None
        elif regret_env:
            regret_path = Path(regret_env)
            if not regret_path.is_file():
                raise FileNotFoundError(regret_path)
        else:
            default = Path(str(snapshot_path) + ".regret.csv")
            regret_path = default if default.is_file() else None
        self.bucket_regret_path = regret_path
        self.last_explanation: dict[str, Any] | None = None
        self.explanations: list[dict[str, Any]] = []
        self.fallbacks = 0
        self._fallback = HeuristicPokerAgent(name)
        self._lock = threading.Lock()
        command = [
            str(exe_path), "serve",
            "--partition", partition or os.environ.get("ASWISS_PARTITION", "history"),
            "--atlas", str(atlas_path),
            "--snapshot", str(snapshot_path),
            "--policy", policy,
            "--seed", str(seed),
        ]
        if regret_path is not None:
            command += ["--bucket-regret", str(regret_path)]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, creationflags=creation_flags,
        )
        ready = self._process.stdout.readline().strip()
        if ready != "READY":
            error = self._process.stderr.read().strip()
            self.close()
            raise RuntimeError(
                (error or f"ASWISS agent failed to start: {ready}")
                + " (the executable needs the serve mode: run_100m.ps1 -Action build -ExeName aswiss_web.exe)"
            )
        self.info = json.loads(self._ask("INFO").split("\t", 1)[1])

    # ---- game interface -------------------------------------------------------------------------------------
    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None
        explanation = self.analyze(state, valid_actions)
        action = explanation.get("action")
        if "error" in explanation or action not in valid_actions:
            self.fallbacks += 1
            action = self._fallback.choose_action(state, valid_actions)
            explanation = {**explanation, "action": action, "fallback": "heuristic"}
        self.last_explanation = explanation
        self.explanations.append(explanation)
        return action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        response = self._ask(f"DISCARD\t{CppMCCFRAgent._cards(hidden_cards)}").split("\t")
        if len(response) != 3 or response[0] != "DISCARD":
            raise RuntimeError(f"Invalid ASWISS discard response: {response}")
        return int(response[1]), int(response[2])

    # ---- strategy analysis ----------------------------------------------------------------------------------
    def analyze(self, state: dict[str, Any], valid_actions: Sequence[str]) -> dict[str, Any]:
        """The snapshot's policy at the decision of the player whose view `state` is (game.get_ai_state).

        Returns the EXPLAIN record (action sampled from the policy, probabilities, key, bucket, genealogy) or
        {"error": ...} when the decision is outside what the model covers."""
        try:
            fields = self._request_fields(state, valid_actions)
            response = self._ask("\t".join(["EXPLAIN", *fields]))
            kind, _, payload = response.partition("\t")
            if kind != "EXPLAIN":
                raise RuntimeError(f"Invalid ASWISS response: {response[:200]}")
            record = json.loads(payload)
        except (RuntimeError, ValueError) as error:
            record = {"error": str(error)}
        record["street"] = record.get("street", state.get("street"))
        record["legal"] = list(valid_actions)
        return record

    def _request_fields(self, state: dict[str, Any], valid_actions: Sequence[str]) -> list[str]:
        if state.get("game_mode") != "ev" or int(state.get("seat_count", 0)) != 2:
            raise ValueError("ASWISS plays heads-up EV mode only (the game it was trained on).")
        if state["street"] not in STREETS:
            raise ValueError(f"ASWISS has no betting decisions on {state['street']}.")
        opponent = state["opponents"][0]
        own_stack = int(state["effective_stack"])
        opponent_stack = int(opponent.get("effective_stack") or own_stack)
        stacks = str(own_stack) if own_stack == opponent_stack else f"{own_stack},{opponent_stack}"
        history = ";".join(
            f"{event['street']}:{0 if event['actor'] == 'self' else 1}:{event['action']}"
            for event in state["betting_history"]
        ) or "-"
        mask = sum(1 << self.ACTIONS.index(action) for action in valid_actions)
        # the viewer's position in table order: a public-priority tie goes to the lower seat, as in poker_env
        position = 0 if int(state["seat_index"]) < int(opponent["seat_index"]) else 1
        return [
            state["street"], str(state["ante"]), str(state["pot"]), str(state["current_highest_bet"]), stacks,
            str(state["my_invested"]), str(state["my_round_bet"]), str(opponent["invested"]), str(opponent["round_bet"]),
            CppMCCFRAgent._cards(state["my_hidden_cards"]), CppMCCFRAgent._cards(state["my_public_cards"]),
            state["my_discarded_card"] or "-", CppMCCFRAgent._cards(opponent["public_cards"]), history, str(mask),
            str(position),
        ]

    # ---- process plumbing -----------------------------------------------------------------------------------
    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"agent": type(self).__name__, "trained": False, "reason": "Frozen ASWISS snapshot."}

    def diagnostics(self) -> dict[str, Any]:
        return {"snapshot": str(self.snapshot_path), "info": self.info, "decisions": len(self.explanations),
                "fallbacks": self.fallbacks}

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                process.wait(timeout=1)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def _ask(self, command: str) -> str:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(self._process.stderr.read().strip() or "ASWISS agent stopped.")
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            response = self._process.stdout.readline().strip()
        if response.startswith("ERROR\t"):
            raise RuntimeError(response.removeprefix("ERROR\t"))
        return response

    def __del__(self) -> None:
        self.close()
