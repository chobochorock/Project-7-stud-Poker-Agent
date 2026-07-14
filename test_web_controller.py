import json
import random
import tempfile
import unittest
from pathlib import Path

from web_controller import WebPokerController, describe_hand_score


class WebControllerTests(unittest.TestCase):
    def test_five_random_players_complete_without_human_input(self):
        random.seed(7)
        controller = WebPokerController()

        state = controller.start(
            ["random", "random", "random", "random", "random"],
            log_file=None,
            replay_dir=None,
            game_mode="tournament",
        )

        self.assertIn(state["phase"], {"complete", "game_over"})
        self.assertEqual(len(state["players"]), 5)
        self.assertIsNotNone(state["result"])
        self.assertIn("round_summaries", state["result"])
        self.assertIn("hand_summaries", state["result"])
        self.assertIn("episode", state)
        live_players = sum(1 for player in state["players"] if player["chips"] > 0)
        self.assertEqual(state["episode_over"], live_players <= 1)

    def test_heuristic_players_complete_without_human_input(self):
        random.seed(13)
        controller = WebPokerController()

        state = controller.start(
            ["heuristic", "heuristic", "random"],
            log_file=None,
            replay_dir=None,
            game_mode="tournament",
        )

        self.assertIn(state["phase"], {"complete", "game_over"})
        self.assertEqual(len(state["players"]), 3)
        self.assertIsNotNone(state["result"])

    def test_human_player_waits_for_discard_choice(self):
        random.seed(7)
        controller = WebPokerController()

        state = controller.start(["human", "random"], log_file=None, replay_dir=None)

        self.assertEqual(state["phase"], "discard_reveal")
        self.assertEqual(state["waiting"]["type"], "discard")
        self.assertEqual(state["waiting"]["player"], "Player_1")
        self.assertEqual(len(state["waiting"]["cards"]), 4)

    def test_human_discard_and_betting_action_can_be_submitted(self):
        random.seed(11)
        controller = WebPokerController()

        state = controller.start(["human", "human"], log_file=None, replay_dir=None)
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

    def test_betting_wait_includes_action_costs_and_priority(self):
        random.seed(11)
        controller = WebPokerController()

        state = controller.start(["human", "human"], log_file=None, replay_dir=None)
        state = controller.submit_discard("Player_1", 0, 1)
        state = controller.submit_discard("Player_2", 0, 1)

        self.assertEqual(state["waiting"]["type"], "bet")
        self.assertIn("action_costs", state["waiting"])
        self.assertIn("HALF", state["waiting"]["action_costs"])
        self.assertEqual(state["acting_player"], state["waiting"]["player"])
        self.assertIn(state["priority_player"], state["turn_order"])

    def test_next_round_reuses_episode_stacks(self):
        random.seed(13)
        controller = WebPokerController()

        state = controller.start(
            ["heuristic", "heuristic", "random"],
            log_file=None,
            replay_dir=None,
            game_mode="tournament",
        )
        if not state["next_round_available"]:
            self.skipTest("Seed ended the episode in one round.")

        next_state = controller.start_next_round()

        self.assertEqual(next_state["round_number"], 2)
        self.assertEqual(next_state["episode"]["total_rounds"], 2)

    def test_replay_file_is_saved_once_per_episode_and_accumulates_rounds(self):
        random.seed(13)
        controller = WebPokerController()

        with tempfile.TemporaryDirectory() as tmp_dir:
            state = controller.start(
                ["heuristic", "heuristic", "random"],
                log_file=None,
                replay_dir=tmp_dir,
                game_mode="tournament",
            )

            replay_file = Path(state["result"]["replay_file"])
            self.assertTrue(replay_file.exists())
            self.assertEqual(replay_file.parent, Path(tmp_dir))
            self.assertIn("heuristic2", replay_file.name)
            self.assertIn("random1", replay_file.name)

            payload = json.loads(replay_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["replay_scope"], "episode")
            self.assertEqual(len(payload["rounds"]), 1)

            if not state["next_round_available"]:
                self.skipTest("Seed ended the episode in one round.")

            next_state = controller.start_next_round()
            self.assertEqual(Path(next_state["result"]["replay_file"]), replay_file)
            self.assertEqual(len(list(Path(tmp_dir).glob("*.json"))), 1)

            payload = json.loads(replay_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_rounds"], 2)
            self.assertEqual(len(payload["rounds"]), 2)

    def test_cash_mode_resets_round_stacks_and_accumulates_profit(self):
        random.seed(21)
        controller = WebPokerController()

        first_state = controller.start(["random", "random"], log_file=None, replay_dir=None, game_mode="cash")
        first_profit = dict(first_state["session"]["cumulative_profit"])
        second_state = controller.start_next_round()

        self.assertEqual(second_state["game_mode"], "cash")
        self.assertEqual(controller.round_start_stacks, {"Player_1": 1000, "Player_2": 1000})
        self.assertEqual(sum(first_profit.values()), 0)
        self.assertEqual(sum(second_state["session"]["cumulative_profit"].values()), 0)
        self.assertEqual(second_state["round_number"], 2)

    def test_cash_round_preserves_total_chips_when_players_fold(self):
        for seed in range(10):
            random.seed(seed)
            controller = WebPokerController()
            state = controller.start(["random", "random"], log_file=None, replay_dir=None, game_mode="cash")

            self.assertEqual(sum(state["session"]["final_chips"].values()), 2000)
            self.assertEqual(sum(state["session"]["cumulative_profit"].values()), 0)

    def test_describe_hand_score_uses_readable_names(self):
        self.assertEqual(describe_hand_score((4, 14)), "백스트레이트")
        self.assertEqual(describe_hand_score((3, 14, 13, 12)), "트리플 A")


if __name__ == "__main__":
    unittest.main()
