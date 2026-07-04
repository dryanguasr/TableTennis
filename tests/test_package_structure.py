import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import table_tennis


ROOT = Path(__file__).resolve().parents[1]


class PackageStructureTests(unittest.TestCase):
    def test_public_api_exposes_core_simulation(self):
        self.assertTrue(callable(table_tennis.simulate))
        self.assertTrue(callable(table_tennis.simulate_racket_impact))
        self.assertTrue(callable(table_tennis.identify_trajectory_moments))
        self.assertIsNotNone(table_tennis.InitialConditions)
        self.assertIsNotNone(table_tennis.RacketImpactParameters)

    def test_physics_core_has_no_interface_or_optimizer_imports(self):
        source = (ROOT / "src/table_tennis/physics.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("matplotlib", "scipy", "ipywidgets", "argparse"):
            self.assertNotIn(forbidden, source)

    def test_legacy_root_python_entrypoints_are_removed(self):
        legacy_files = (
            "table_tennis_simulation.py",
            "service_parameter_search.py",
            "return_parameter_search.py",
            "benchmark_direct_services.py",
            "benchmark_racket_services.py",
            "benchmark_returns.py",
            "generate_return_videos.py",
            "generate_racket_benchmark_web.py",
        )
        self.assertEqual(
            [name for name in legacy_files if (ROOT / name).exists()],
            [],
        )

    def test_cli_help_works_outside_repository(self):
        commands = (
            ("--help",),
            ("doctor",),
            ("simulate", "--help"),
            ("benchmark", "direct", "--help"),
            ("benchmark", "racket", "--help"),
            ("benchmark", "returns", "--help"),
            ("search", "service", "--help"),
            ("search", "exercise", "--help"),
            ("generate", "return-videos", "--help"),
            ("generate", "benchmark-videos", "--help"),
            ("generate", "exercise-videos", "--help"),
            ("generate", "racket-viewer", "--help"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                completed = subprocess.run(
                    [sys.executable, "-m", "table_tennis", *command],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"{command}: {completed.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
