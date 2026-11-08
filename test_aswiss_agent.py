"""ASWISS web agent (agent/aswiss_agent.py + aswiss_stud serve). Skipped unless the serve executable and a snapshot are
available (ASWISS_EXE / ASWISS_SNAPSHOT or the defaults of agent/aswiss_agent.py)."""
import random
import unittest

from web_controller import WebPokerController


def aswiss_available() -> bool:
    try:
        from agent.aswiss_agent import ASWISSAgent
        ASWISSAgent("Probe").close()
        return True
    except (OSError, RuntimeError):
        return False


@unittest.skipUnless(aswiss_available(), "no aswiss_stud serve executable or snapshot")
class ASWISSWebTests(unittest.TestCase):
    def start(self, types):
        controller = WebPokerController()
        controller.start(types, game_mode="ev", ante=1000, log_file=None, replay_dir=None)
        return controller

    def test_both_seat_orders_play_without_errors_or_fallbacks(self):
        for types in (["aswiss", "heuristic"], ["heuristic", "aswiss"]):
            controller = self.start(types)
            decisions = 0
            for hand in range(40):
                for record in controller.analysis:
                    decisions += 1
                    self.assertNotIn("error", record)
                    self.assertNotIn("fallback", record)
                    self.assertAlmostEqual(sum(record["probabilities"].values()), 1.0, places=4)
                    self.assertTrue(set(record["probabilities"]) <= set(record["legal"]))
                    self.assertTrue(record["stack_matches_training"])
                if hand < 39:
                    controller.start_next_round()
            self.assertGreater(decisions, 0)

    def test_agent_records_hidden_from_a_seated_human_until_the_round_ends(self):
        rng = random.Random(3)
        controller = self.start(["human", "aswiss"])
        for _ in range(20):
            while controller.waiting:
                waiting = controller.waiting
                if waiting["type"] == "bet":
                    analysis = controller.analyze_decision(waiting["player"])
                    self.assertNotIn("error", analysis)
                    shown = controller.public_state()["analysis"]
                    self.assertTrue(all(record["by"] == "human" for record in shown))
                    controller.submit_action(waiting["player"], rng.choice(waiting["valid_actions"]))
                else:
                    controller.submit_discard(waiting["player"], 0, 1)
            state = controller.public_state()
            self.assertEqual(len(state["analysis"]), len(controller.analysis))
            for record in state["analysis"]:
                if record["by"] == "human":
                    self.assertIn(record["action_taken"], record["probabilities"])
            controller.start_next_round()

    def test_analysis_needs_heads_up_ev(self):
        controller = WebPokerController()
        controller.start(["human", "heuristic", "heuristic"], game_mode="ev", ante=1000, log_file=None, replay_dir=None)
        self.assertFalse(controller.public_state()["analysis_available"])
        with self.assertRaises(ValueError):
            controller.start(["aswiss", "heuristic", "heuristic"], game_mode="ev", ante=1000, log_file=None, replay_dir=None)


if __name__ == "__main__":
    unittest.main()
