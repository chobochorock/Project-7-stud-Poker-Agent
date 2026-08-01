from __future__ import annotations

import contextlib
import io
import random
import unittest

from agent.claude_belief_br import ClaudeBeliefBRAgent
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import PokerGame


def make_state(**overrides):
    """A 6th-street heads-up state: I hold trip kings, opponent shows junk."""
    state = {
        "game_mode": "ev",
        "street": "6th",
        "seat_count": 2,
        "seat_index": 0,
        "ante": 1,
        "pot": 20,
        "current_highest_bet": 0,
        "my_chips": 980,
        "my_invested": 20,
        "my_is_all_in": False,
        "my_round_bet": 0,
        "my_hidden_cards": ["sK", "dK"],
        "my_public_cards": ["hK", "c4", "s7"],
        "my_discarded_card": "c2",
        "call_amount": 0,
        "raise_count": 0,
        "raise_cap": 2,
        "my_bet_count": 0,
        "effective_stack": 1000,
        "opponents": [
            {
                "seat": "opponent_1",
                "seat_index": 1,
                "chips": 980,
                "invested": 20,
                "round_bet": 0,
                "public_cards": ["h2", "d5", "c9"],
                "is_folded": False,
                "is_all_in": False,
                "is_eliminated": False,
            }
        ],
        "betting_history": [],
    }
    state.update(overrides)
    return state


class BeliefBRInterfaceTests(unittest.TestCase):
    def test_returns_valid_action(self):
        agent = ClaudeBeliefBRAgent("A", belief_particles=64, seed=1)
        with contextlib.redirect_stdout(io.StringIO()):
            action = agent.choose_action(make_state(), ["CHECK", "QUARTER", "HALF", "FOLD"])
        self.assertIn(action, {"CHECK", "QUARTER", "HALF", "FOLD"})

    def test_discard_reveal_valid_indices(self):
        agent = ClaudeBeliefBRAgent("A", belief_particles=32, seed=1)
        discard, reveal = agent.choose_discard_and_reveal(["sA", "dA", "c2", "h7"])
        self.assertNotEqual(discard, reveal)
        self.assertIn(discard, range(4))
        self.assertIn(reveal, range(4))

    def test_learn_reports_planning_agent(self):
        agent = ClaudeBeliefBRAgent("A")
        info = agent.learn_from_database()
        self.assertFalse(info["trained"])


class BeliefUpdateTests(unittest.TestCase):
    def test_aggression_shifts_belief_toward_strong_hands(self):
        """Action-conditioned belief on a marginal hand (a pair of eights).

        ``equity`` (likelihood-weighted) and ``uniform_equity`` (likelihood off)
        share the same particles inside one call, so comparing them isolates the
        belief update from Monte Carlo noise. A big-betting opponent must push my
        marginal-hand equity below uniform; a checking opponent must push it up.
        """
        marginal = dict(my_hidden_cards=["s8", "d8"], my_public_cards=["h3", "cJ", "s6"])
        agent = ClaudeBeliefBRAgent("A", belief_particles=500, seed=3)

        aggressive = agent.estimate_belief(
            make_state(
                betting_history=[
                    {"street": "5th", "actor": "opponent_1", "action": "HALF"},
                    {"street": "6th", "actor": "opponent_1", "action": "HALF"},
                ],
                **marginal,
            )
        )
        passive = agent.estimate_belief(
            make_state(
                betting_history=[{"street": "6th", "actor": "opponent_1", "action": "CHECK"}],
                **marginal,
            )
        )

        # Believed-strong opponent -> my marginal hand is worth less than uniform.
        self.assertLess(aggressive["equity"], aggressive["uniform_equity"] - 0.03)
        # Believed-weak opponent -> my marginal hand is worth more than uniform.
        self.assertGreater(passive["equity"], passive["uniform_equity"] + 0.02)

    def test_strong_hand_bets_for_value(self):
        # With the conservatism knob off (aggression_margin=0), the raw
        # best-response must value-bet a monster (trip kings). The tuned default
        # margin deliberately pot-controls instead; that policy choice is
        # validated by the EV sweep, not here.
        agent = ClaudeBeliefBRAgent("A", belief_particles=200, aggression_margin=0.0, seed=5)
        with contextlib.redirect_stdout(io.StringIO()):
            action = agent.choose_action(make_state(), ["CHECK", "QUARTER", "HALF", "FOLD"])
        self.assertIn(action, {"QUARTER", "HALF"})

    def test_aggression_margin_suppresses_thin_bets(self):
        # The same monster-free marginal spot: a high margin must not bet.
        marginal = make_state(my_hidden_cards=["s8", "d8"], my_public_cards=["h3", "cJ", "s6"])
        passive_agent = ClaudeBeliefBRAgent("A", belief_particles=200, aggression_margin=0.8, seed=5)
        with contextlib.redirect_stdout(io.StringIO()):
            action = passive_agent.choose_action(marginal, ["CHECK", "QUARTER", "HALF", "FOLD"])
        self.assertEqual(action, "CHECK")

    def test_risk_aversion_lowers_certainty_equivalent(self):
        # CARA scoring: a symmetric gamble (mean 0) is worth < 0 to a risk-averse
        # agent and never worse than the worst outcome; lambda = 0 returns the mean.
        outcomes = [(1.0, 100.0), (1.0, -100.0)]  # mean 0, high variance
        neutral = ClaudeBeliefBRAgent("A", belief_particles=50, risk_lambda=0.0, seed=1)
        averse = ClaudeBeliefBRAgent("A", belief_particles=50, risk_lambda=5.0, seed=1)
        self.assertAlmostEqual(neutral._ce(outcomes, 100.0), 0.0, places=6)
        certainty_equivalent = averse._ce(outcomes, 100.0)
        self.assertLess(certainty_equivalent, 0.0)
        self.assertGreaterEqual(certainty_equivalent, -100.0)

    def test_trash_folds_to_big_bet(self):
        # I hold disconnected low cards and face a pot-sized bet.
        trash = make_state(
            my_hidden_cards=["s2", "d7"],
            my_public_cards=["h9", "cJ", "s4"],
            call_amount=40,
            current_highest_bet=40,
            pot=60,
            my_invested=20,
            betting_history=[{"street": "6th", "actor": "opponent_1", "action": "HALF"}],
        )
        agent = ClaudeBeliefBRAgent("A", belief_particles=200, seed=7)
        with contextlib.redirect_stdout(io.StringIO()):
            action = agent.choose_action(trash, ["CALL", "FOLD"])
        self.assertEqual(action, "FOLD")


class BeliefBRIntegrationTests(unittest.TestCase):
    def test_plays_full_ev_hand_zero_sum(self):
        random.seed(11)
        agents = {
            "Player_1": ClaudeBeliefBRAgent("Player_1", belief_particles=48, seed=2),
            "Player_2": HeuristicPokerAgent("Player_2"),
        }
        game = PokerGame(["Player_1", "Player_2"], log_file=None, ante=1000, game_mode="ev")
        with contextlib.redirect_stdout(io.StringIO()):
            result = game.play_hand(agents)
        chips = result["final_chips"]
        self.assertEqual(chips["Player_1"] + chips["Player_2"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
