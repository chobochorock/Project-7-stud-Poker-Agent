import unittest

from kuhn_cfr import KuhnCFR


class KuhnCFRTests(unittest.TestCase):
    def test_average_strategy_approaches_the_kuhn_equilibrium(self):
        solver = KuhnCFR(seed=7)
        solver.train(20_000)
        result = solver.summary()

        self.assertEqual(len(result["strategy"]), 12)
        self.assertAlmostEqual(
            result["profile_value_player_0"],
            result["known_game_value_player_0"],
            delta=0.02,
        )
        self.assertLess(result["exploitability"], 0.02)


if __name__ == "__main__":
    unittest.main()
