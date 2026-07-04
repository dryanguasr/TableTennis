import unittest

import numpy as np

from table_tennis.search.service import _fallback_random_search


class ServiceParameterSearchProgressTests(unittest.TestCase):
    def test_fallback_reports_monotonic_progress(self):
        updates = []
        _fallback_random_search(
            objective=lambda vector: float(np.sum(vector**2)),
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            seed=7,
            iterations=40,
            progress_callback=updates.append,
        )
        self.assertTrue(updates)
        self.assertTrue(all(update.phase == "global" for update in updates))
        self.assertEqual(updates[-1].current, updates[-1].total)
        self.assertEqual(
            [update.current for update in updates],
            sorted(update.current for update in updates),
        )
        self.assertIsNotNone(updates[-1].best_cost)


if __name__ == "__main__":
    unittest.main()
