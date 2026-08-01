"""Exact, finite 7th-street poker oracle for CFR+ versus AsymP.

The benchmark fixes public cards and gives each player three possible private
hands.  A single check/bet/call/fold round is small enough to enumerate as a
64 x 64 normal-form zero-sum matrix.  This is an algorithm oracle, not yet the
full v3 betting tree or a safe trunk resolver.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from poker_env import Card, get_best_hand


P0_PUBLIC = (Card("s", "2"), Card("d", "7"), Card("c", "J"))
P1_PUBLIC = (Card("h", "4"), Card("s", "8"), Card("d", "Q"))
P0_PRIVATE = (
    (Card("h", "3"), Card("s", "5"), Card("d", "9")),
    (Card("h", "J"), Card("s", "4"), Card("d", "6")),
    (Card("h", "7"), Card("c", "7"), Card("d", "2")),
)
P1_PRIVATE = (
    (Card("c", "2"), Card("h", "6"), Card("s", "9")),
    (Card("c", "Q"), Card("c", "3"), Card("d", "5")),
    (Card("c", "8"), Card("h", "8"), Card("c", "4")),
)


@dataclass(frozen=True)
class PurePlan:
    first: tuple[int, ...]
    response: tuple[int, ...]


def pure_plans(types: int) -> list[PurePlan]:
    # Per type: first action (check/bet) and facing-bet response (fold/call).
    return [
        PurePlan(tuple(code & 1 for code in choices),
                 tuple((code >> 1) & 1 for code in choices))
        for choices in itertools.product(range(4), repeat=types)
    ]


def showdown(type0: int, type1: int) -> int:
    score0 = get_best_hand(P0_PUBLIC + P0_PRIVATE[type0])
    score1 = get_best_hand(P1_PUBLIC + P1_PRIVATE[type1])
    return (score0 > score1) - (score0 < score1)


def terminal_payoff(
    plan0: PurePlan,
    plan1: PurePlan,
    type0: int,
    type1: int,
    pot: float,
    bet: float,
) -> float:
    """Return P0 chip payoff relative to the start of the betting round."""
    if plan0.first[type0]:
        if not plan1.response[type1]:
            return pot / 2.0
        return showdown(type0, type1) * (pot / 2.0 + bet)
    if not plan1.first[type1]:
        return showdown(type0, type1) * pot / 2.0
    if not plan0.response[type0]:
        return -pot / 2.0
    return showdown(type0, type1) * (pot / 2.0 + bet)


def build_matrix(pot: float, bet: float) -> tuple[np.ndarray, list[PurePlan]]:
    plans = pure_plans(len(P0_PRIVATE))
    probabilities = np.full((len(P0_PRIVATE), len(P1_PRIVATE)), 1.0 / 9.0)
    payoff = np.empty((len(plans), len(plans)), dtype=np.float64)
    for row, plan0 in enumerate(plans):
        for column, plan1 in enumerate(plans):
            value = 0.0
            for type0, type1 in itertools.product(range(3), repeat=2):
                value += probabilities[type0, type1] * terminal_payoff(
                    plan0, plan1, type0, type1, pot, bet
                )
            payoff[row, column] = value
    return -payoff, plans  # Row minimizes cost; column maximizes it.


def gap(matrix: np.ndarray, row: np.ndarray, column: np.ndarray) -> float:
    return float(np.max(row @ matrix) - np.min(matrix @ column))


def solve_lp(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rows, columns = matrix.shape
    row_result = linprog(
        np.r_[np.zeros(rows), 1.0],
        A_ub=np.c_[matrix.T, -np.ones(columns)],
        b_ub=np.zeros(columns),
        A_eq=np.r_[np.ones(rows), 0.0][None, :],
        b_eq=[1.0],
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
    )
    column_result = linprog(
        np.r_[np.zeros(columns), -1.0],
        A_ub=np.c_[-matrix, np.ones(rows)],
        b_ub=np.zeros(rows),
        A_eq=np.r_[np.ones(columns), 0.0][None, :],
        b_eq=[1.0],
        bounds=[(0.0, None)] * columns + [(None, None)],
        method="highs",
    )
    if not row_result.success or not column_result.success:
        raise RuntimeError("equilibrium LP failed")
    row = row_result.x[:-1]
    column = column_result.x[:-1]
    return row, column, float(row @ matrix @ column)


def regret_strategy(regret: np.ndarray) -> np.ndarray:
    positive = np.maximum(regret, 0.0)
    total = positive.sum()
    return positive / total if total else np.full(regret.size, 1.0 / regret.size)


def run_cfr_plus(matrix: np.ndarray, updates: int) -> dict[str, object]:
    iterations = updates // 2
    row_regret = np.zeros(matrix.shape[0])
    column_regret = np.zeros(matrix.shape[1])
    row_sum = np.zeros_like(row_regret)
    column_sum = np.zeros_like(column_regret)
    curve = []
    started = time.perf_counter()
    checkpoints = {max(1, iterations // scale) for scale in (100, 30, 10, 3, 1)}
    for iteration in range(1, iterations + 1):
        row = regret_strategy(row_regret)
        column = regret_strategy(column_regret)
        row_values = matrix @ column
        column_values = row @ matrix
        value = float(row @ row_values)
        row_regret = np.maximum(row_regret + value - row_values, 0.0)
        column_regret = np.maximum(column_regret + column_values - value, 0.0)
        row_sum += iteration * row
        column_sum += iteration * column
        if iteration in checkpoints:
            curve.append({
                "updates": 2 * iteration,
                "nash_conv": gap(matrix, row_sum / row_sum.sum(),
                                 column_sum / column_sum.sum()),
            })
    row = row_sum / row_sum.sum()
    column = column_sum / column_sum.sum()
    return {
        "updates": 2 * iterations,
        "nash_conv": gap(matrix, row, column),
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
        "row": row,
        "column": column,
    }


def project_simplex(vector: np.ndarray) -> np.ndarray:
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, vector.size + 1)
    rho = np.flatnonzero(ordered * indices > cumulative - 1.0)[-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def run_asymp(
    matrix: np.ndarray,
    updates: int,
    mu: float,
    learning_rate: float | None,
    seed: int,
) -> dict[str, object]:
    iterations = updates // 4
    rng = np.random.default_rng(seed)
    row0 = rng.dirichlet(np.ones(matrix.shape[0]))
    column0 = rng.dirichlet(np.ones(matrix.shape[1]))
    row_for_row, column_for_row = row0.copy(), column0.copy()
    row_for_column, column_for_column = row0.copy(), column0.copy()
    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    bound = mu / (mu * mu + spectral_norm * spectral_norm)
    step = learning_rate if learning_rate is not None else 0.9 * bound
    if step > bound:
        raise ValueError(f"learning rate {step} exceeds paper bound {bound}")
    curve = []
    started = time.perf_counter()
    checkpoints = {max(1, iterations // scale) for scale in (100, 30, 10, 3, 1)}
    for iteration in range(1, iterations + 1):
        row_for_row = project_simplex(
            row_for_row - step * (matrix @ column_for_row + mu * row_for_row)
        )
        column_for_row = project_simplex(
            column_for_row + step * (matrix.T @ row_for_row)
        )
        column_for_column = project_simplex(
            column_for_column + step * (
                matrix.T @ row_for_column - mu * column_for_column
            )
        )
        row_for_column = project_simplex(
            row_for_column - step * (matrix @ column_for_column)
        )
        if iteration in checkpoints:
            curve.append({
                "updates": 4 * iteration,
                "nash_conv": gap(matrix, row_for_row, column_for_column),
            })
    return {
        "updates": 4 * iterations,
        "nash_conv": gap(matrix, row_for_row, column_for_column),
        "elapsed_seconds": time.perf_counter() - started,
        "spectral_norm": spectral_norm,
        "paper_step_bound": bound,
        "learning_rate": step,
        "mu": mu,
        "curve": curve,
        "row": row_for_row,
        "column": column_for_column,
    }


def behavior_summary(mixture: np.ndarray, plans: list[PurePlan]) -> list[dict[str, float]]:
    return [
        {
            "first_aggressive": float(sum(
                weight * plan.first[type_] for weight, plan in zip(mixture, plans)
            )),
            "call_facing_bet": float(sum(
                weight * plan.response[type_] for weight, plan in zip(mixture, plans)
            )),
        }
        for type_ in range(3)
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    matrix, plans = build_matrix(args.pot, args.bet)
    exact_row, exact_column, exact_value = solve_lp(matrix)
    cfr = run_cfr_plus(matrix, args.updates)
    asymp = run_asymp(matrix, args.updates, args.mu, args.learning_rate, args.seed)
    result = {
        "game": "fixed_7th_one_bet_oracle",
        "scope": "3 private types/player; check-bet-call-fold; no future chance",
        "matrix_shape": list(matrix.shape),
        "pot": args.pot,
        "bet": args.bet,
        "exact": {
            "value_for_p0": -exact_value,
            "nash_conv": gap(matrix, exact_row, exact_column),
            "p0_behavior": behavior_summary(exact_row, plans),
            "p1_behavior": behavior_summary(exact_column, plans),
        },
        "cfr_plus": {
            key: value for key, value in cfr.items() if key not in {"row", "column"}
        },
        "asymp_l2": {
            key: value for key, value in asymp.items() if key not in {"row", "column"}
        },
    }
    result["cfr_plus"]["p0_behavior"] = behavior_summary(cfr["row"], plans)
    result["cfr_plus"]["p1_behavior"] = behavior_summary(cfr["column"], plans)
    result["asymp_l2"]["p0_behavior"] = behavior_summary(asymp["row"], plans)
    result["asymp_l2"]["p1_behavior"] = behavior_summary(asymp["column"], plans)
    return result


def self_test() -> None:
    matrix, plans = build_matrix(4.0, 1.0)
    assert matrix.shape == (64, 64) and len(plans) == 64
    row, column, value = solve_lp(matrix)
    assert math.isfinite(value) and gap(matrix, row, column) < 1e-8
    cfr = run_cfr_plus(matrix, 20_000)
    asymp = run_asymp(matrix, 20_000, 1.0, None, 7)
    assert cfr["nash_conv"] < 0.02
    assert asymp["nash_conv"] < 0.2


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--pot", type=float, default=4.0)
    parser.add_argument("--bet", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        print('{"self_test":"ok"}')
        return
    if args.updates < 4 or args.updates % 4 or min(args.pot, args.bet, args.mu) <= 0:
        parser.error("updates must be a positive multiple of 4; pot/bet/mu must be positive")
    result = run(args)
    payload = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as output:
            output.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
