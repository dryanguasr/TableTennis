import pickle
import unittest

import numpy as np

from table_tennis.benchmarks import direct, racket
from table_tennis.events import net_crossings, table_bounces
from table_tennis.physics import simulate_racket_impact
from table_tennis.search.retune_all import DirectRetuneJob, _direct_metrics


class AceRetuningAcceptanceTests(unittest.TestCase):
    def test_all_54_direct_services_are_valid(self):
        rows = []
        for case in direct.build_cases():
            rows.append(
                _direct_metrics(
                    DirectRetuneJob(case.service, case.depth, case.lane),
                    case.initial_conditions.vel,
                )
            )
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertLessEqual(
            max(float(row["target_error_mm"]) for row in rows),
            direct.TARGET_MARGIN_MM,
        )
        self.assertLessEqual(
            max(float(row["max_height_above_net_mm"]) for row in rows),
            direct.MAX_FLIGHT_HEIGHT_ABOVE_NET_MM,
        )
        self.assertLessEqual(
            max(float(row["rebound_height_above_net_mm"]) for row in rows),
            direct.MAX_REBOUND_HEIGHT_ABOVE_NET_MM,
        )

    def test_all_54_racket_services_match_the_direct_suite(self):
        cases = list(racket.build_cases())
        self.assertEqual(len(cases), 54)
        errors = []
        for case in cases:
            result = simulate_racket_impact(case.params, t_max=1.8)
            bounces = table_bounces(result)
            self.assertGreaterEqual(len(bounces), 2)
            self.assertEqual(
                (bounces[0].side, bounces[1].side),
                ("server", "receiver"),
            )
            self.assertFalse(
                any(event.kind == "net_contact" for event in result.events)
            )
            self.assertTrue(net_crossings(result))
            max_height, rebound_height = direct.low_arc_metrics(result)
            self.assertLessEqual(
                max_height,
                direct.MAX_FLIGHT_HEIGHT_ABOVE_NET_MM,
            )
            self.assertLessEqual(
                rebound_height,
                direct.MAX_REBOUND_HEIGHT_ABOVE_NET_MM,
            )
            target = np.array(
                (
                    direct.DEPTHS[case.depth]["target_x"],
                    direct.LANES[case.lane]["y"],
                )
            )
            point = np.array(bounces[1].point[:2])
            errors.append(float(np.linalg.norm(point - target)))
        self.assertLessEqual(max(errors), direct.TARGET_MARGIN_MM)

    def test_retune_jobs_are_pickleable_for_windows_workers(self):
        job = DirectRetuneJob("pendulum", "short", "elbow", maxiter=1)
        self.assertEqual(pickle.loads(pickle.dumps(job)), job)


if __name__ == "__main__":
    unittest.main()
