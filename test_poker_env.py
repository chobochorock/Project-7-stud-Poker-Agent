import unittest
import json

from poker_env import Card, PokerGame, evaluate_5_cards, get_public_betting_priority


def cards(labels):
    return [Card(label[0], label[1:]) for label in labels]


class PokerRuleTests(unittest.TestCase):
    def test_back_straight_beats_ordinary_straights_but_loses_to_mountain(self):
        king_high = evaluate_5_cards(cards(["sK", "hQ", "dJ", "c10", "s9"]))
        back = evaluate_5_cards(cards(["sA", "h2", "d3", "c4", "s5"]))
        mountain = evaluate_5_cards(cards(["sA", "hK", "dQ", "cJ", "s10"]))

        self.assertGreater(back, king_high)
        self.assertGreater(mountain, back)

    def test_side_pot_limits_all_in_player_and_awards_remaining_side_pot(self):
        game = PokerGame(["A", "B", "C"], log_file=None)
        player_a, player_b, player_c = game.players

        for player in game.players:
            player.chips = 0
            player.is_all_in = True

        player_a.invested = 100
        player_b.invested = 300
        player_c.invested = 300
        game.pot = 700

        player_a.hidden_cards = cards(["sA", "hA", "dA", "cA", "sK"])
        player_b.hidden_cards = cards(["sK", "hK", "dK", "c2", "d3"])
        player_c.hidden_cards = cards(["sQ", "hQ", "dQ", "c4", "d5"])

        game.resolve_showdown()

        self.assertEqual(player_a.chips, 300)
        self.assertEqual(player_b.chips, 400)
        self.assertEqual(player_c.chips, 0)

    def test_valid_actions_include_check_and_full(self):
        game = PokerGame(["A", "B"], log_file=None)
        player = game.players[0]
        game.pot = 5

        self.assertIn("CHECK", game.get_valid_actions(player))
        self.assertIn("FULL", game.get_valid_actions(player))
        self.assertNotIn("CALL", game.get_valid_actions(player))

    def test_public_trips_take_betting_priority_over_weaker_public_hands(self):
        trips = get_public_betting_priority(cards(["s7", "h7", "d7", "c2"]))
        two_pair = get_public_betting_priority(cards(["sA", "hA", "dK", "cK"]))
        pair = get_public_betting_priority(cards(["sA", "hA", "dK", "c2"]))

        self.assertGreater(trips, two_pair)
        self.assertGreater(two_pair, pair)

    def test_betting_round_starts_from_highest_public_hand(self):
        order = []

        class CheckAgent:
            def __init__(self, name):
                self.name = name

            def choose_action(self, state, valid_actions):
                order.append(self.name)
                return "CHECK" if "CHECK" in valid_actions else "CALL"

        game = PokerGame(["A", "B", "C"], log_file=None)
        game.street = "6th"
        game.pot = 10
        game.players[0].public_cards = cards(["sA", "hA", "dK", "c2"])
        game.players[1].public_cards = cards(["s7", "h7", "d7", "c3"])
        game.players[2].public_cards = cards(["sK", "hQ", "dJ", "c9"])

        agents = {player.name: CheckAgent(player.name) for player in game.players}
        game.play_betting_round(agents)

        self.assertEqual(order, ["B", "C", "A"])

    def test_agent_state_has_betting_history_without_player_names(self):
        game = PokerGame(["Alice", "Bob"], log_file=None)
        game.pot = 5
        game.street = "4th"
        game._record_bet(game.players[1], "CHECK", 0, 0, 0)

        state = game.get_ai_state(game.players[0])
        dumped = json.dumps(state)

        self.assertIn("betting_history", state)
        self.assertIn("opponent_1", dumped)
        self.assertNotIn("Alice", dumped)
        self.assertNotIn("Bob", dumped)


if __name__ == "__main__":
    unittest.main()
