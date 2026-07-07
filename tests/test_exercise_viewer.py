import tempfile
import unittest
from pathlib import Path

from table_tennis.visualization.exercise_viewer import (
    HTML_TEMPLATE,
    build_exercise_viewer_data,
    main,
)


class ExerciseViewerTests(unittest.TestCase):
    def test_catalog_data_has_ten_exercises_and_relative_video_paths(self):
        items = build_exercise_viewer_data(
            Path("outputs/exercises"),
            Path("outputs/viewers"),
        )
        self.assertEqual(len(items), 10)
        self.assertEqual(len({item["id"] for item in items}), 10)
        for item in items:
            self.assertFalse(Path(item["video"]).is_absolute())
            self.assertEqual(item["cycles"], 3)
            self.assertEqual(item["strokeCount"], len(item["sequence"]))
            self.assertTrue(item["generateCommand"].startswith("table-tennis"))
            for stroke in item["sequence"]:
                self.assertIn(stroke["moment"], (2, 3, 4))
                self.assertIn(stroke["depth"], ("short", "long"))

    def test_html_supports_filters_playback_and_missing_video_state(self):
        self.assertIn('id="search"', HTML_TEMPLATE)
        self.assertIn('id="family"', HTML_TEMPLATE)
        self.assertIn('id="wing"', HTML_TEMPLATE)
        self.assertIn("controls autoplay muted loop playsinline", HTML_TEMPLATE)
        self.assertIn("Falta MP4", HTML_TEMPLATE)
        self.assertIn("No se encontró", HTML_TEMPLATE)
        self.assertIn("Secuencia ordenada", HTML_TEMPLATE)

    def test_main_writes_utf8_viewer_from_explicit_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site" / "exercise_viewer.html"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "drive_to_drive.mp4").write_bytes(b"placeholder")
            main(
                [
                    "--output",
                    str(output),
                    "--video-dir",
                    str(video_dir),
                ]
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn("Ejercicios de tenis de mesa", html)
            self.assertIn("Revés", html)
            self.assertIn("drive_to_drive", html)
            self.assertIn("../videos/drive_to_drive.mp4", html)
            self.assertNotIn(str(root), html)


if __name__ == "__main__":
    unittest.main()
