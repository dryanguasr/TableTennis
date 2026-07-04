import json
import pickle
import unittest
from pathlib import Path

from table_tennis.benchmarks.videos import (
    build_video_jobs,
    probe_mp4_duration,
)
from table_tennis.doctor import collect_diagnostics
from table_tennis.visualization.notebook_video import notebook_video_path


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentAndVideoBatchTests(unittest.TestCase):
    def test_doctor_environment_is_healthy(self):
        failures = [
            f"{check.name}: {check.detail}"
            for check in collect_diagnostics()
            if not check.passed
        ]
        self.assertEqual(failures, [])

    def test_registered_kernel_uses_absolute_venv_python(self):
        kernel = (
            Path.home()
            / "AppData/Roaming/jupyter/kernels/table-tennis/kernel.json"
        )
        data = json.loads(kernel.read_text(encoding="utf-8"))
        configured = Path(data["argv"][0]).resolve()
        expected = (ROOT / ".venv/Scripts/python.exe").resolve()
        self.assertEqual(configured, expected)

    def test_all_suites_build_118_pickleable_jobs(self):
        jobs = build_video_jobs(
            ffmpeg_path="ffmpeg",
            fps=30,
            duration=5.0,
        )
        self.assertEqual(len(jobs), 118)
        pickle.dumps(jobs)

    def test_notebook_path_creates_expected_directory(self):
        path = notebook_video_path(
            "test-notebook",
            "service / elbow",
            output_root=ROOT / "outputs/notebooks-test",
            timestamp="20260630_120000",
        )
        self.assertEqual(path.name, "20260630_120000_service-elbow.mp4")
        self.assertTrue(path.parent.is_dir())

    def test_smoke_video_is_decodable_when_present(self):
        path = ROOT / "outputs/serve_return_search/videos/smoke_test.mp4"
        if path.exists():
            self.assertIsNotNone(probe_mp4_duration(path))


if __name__ == "__main__":
    unittest.main()
