"""Check evaluation timing and trailing smoothing, without a training run.

Usage: python -m unittest agents.state_action_embedding.test_refinement_per_hand
"""

import unittest
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from . import refinement_per_hand as p


class PerHandTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Use plotting Python")
    def test_initial_panel_requires_untrained_matching_roots(self):
        from . import refinement_report as report

        row = dict(
            arm="raw",
            policy="current",
            root=0,
            root_seed=7,
            observation_sha256="test",
            legal=[True, True],
            strategy=[0.5, 0.5],
            policy_hands=0,
            local_br_gap=2.0,
            crossfit_deviation=-1.0,
        )
        evaluation = dict(seed=7, particles=32, roots=1)
        identities = {0: (7, "test", (True, True))}
        with patch.object(
            Path, "read_text", return_value=json.dumps(evaluation)
        ), patch.object(report, "read_rows", return_value=[row]):
            result = report.initial_panel(Path("test"), evaluation, identities)
            self.assertEqual(result["mean"], 2.0)
            self.assertEqual(result["crossfit"]["mean"], -1.0)
            row["policy_hands"] = 128
            with self.assertRaises(ValueError):
                report.initial_panel(Path("test"), evaluation, identities)

    def test_timing_and_mean(self):
        expected = {
            1: 0,
            128: 0,
            129: 128,
            1000: 896,
            9999: 9984,
            10000: 9984,
            10001: 10000,
            10128: 10000,
            10129: 10128,
        }
        for hand, budget in expected.items():
            self.assertEqual(p.policy_budget(hand), budget)
        p.np.testing.assert_allclose(
            p.moving_mean(p.np.arange(1, 201)), p.np.arange(50.5, 151.5)
        )
        p.np.testing.assert_array_equal(p.moving_mean([0.0] * 100), [0.0])
        with self.assertRaises(ValueError):
            p.moving_mean([1.0] * 99)


if __name__ == "__main__":
    unittest.main()
