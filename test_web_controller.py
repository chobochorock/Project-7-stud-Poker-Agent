import random
import unittest

from web_controller import WebPokerController


class WebControllerTests(unittest.TestCase):
    def test_five_random_players_complete_without_human_input(self):
        random.seed(7)
        controller = WebPokerController()

        state = controller.start(["random", "random", "random", "random", "random"], log_file=None)

        self.assertEqual(state["phase"], "complete")
        self.assertEqual(len(state["players"]), 5)
        self.assertIsNotNone(state["result"])

    def test_human_player_waits_for_discard_choice(self):
        random.seed(7)
        controller = WebPokerController()

        state = controller.start(["human", "random"], log_file=None)

        self.assertEqual(state["phase"], "discard_reveal")
        self.assertEqual(state["waiting"]["type"], "discard")
        self.assertEqual(state["waiting"]["player"], "Player_1")
        self.assertEqual(len(state["waiting"]["cards"]), 4)

    def test_human_discard_and_betting_action_can_be_submitted(self):
        random.seed(11)
        controller = WebPokerController()

        state = controller.start(["human", "human"], log_file=None)
        self.assertEqual(state["waiting"]["player"], "Player_1")

        state = controller.submit_discard("Player_1", 0, 1)
        self.assertEqual(state["waiting"]["player"], "Player_2")

        state = controller.submit_discard("Player_2", 0, 1)
        self.assertEqual(state["waiting"]["type"], "bet")

        acting_player = state["waiting"]["player"]
        valid_actions = state["waiting"]["valid_actions"]
        action = "CHECK" if "CHECK" in valid_actions else "CALL"
        state = controller.submit_action(acting_player, action)

        self.assertIn(state["phase"], {"betting", "street_start", "showdown", "complete"})


if __name__ == "__main__":
    unittest.main()
