"""Shared helpers for notebook MP4 output."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..search.service import ServiceSearchResult
from .animation import animate_racket_impact, animate_simulation


def notebook_video_path(
    notebook: str,
    stem: str,
    *,
    output_root: Path | str = Path("outputs/notebooks"),
    timestamp: str | None = None,
) -> Path:
    """Return a timestamped, filesystem-safe MP4 path for one notebook."""

    safe_notebook = re.sub(r"[^a-zA-Z0-9_-]+", "-", notebook).strip("-")
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-")
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_root) / safe_notebook / f"{timestamp}_{safe_stem}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_service_search_video(
    result: ServiceSearchResult,
    output_path: Path | str,
    *,
    fps: int = 30,
    ffmpeg_path: str | None = None,
) -> Path:
    """Render a direct or racket search result without reconstructing its input."""

    if not result.success:
        raise ValueError("Only successful service searches can be rendered.")
    if result.mode == "direct":
        return animate_simulation(
            result.trajectory,
            save=str(output_path),
            ffmpeg_path=ffmpeg_path,
            fps=fps,
        )
    if result.mode == "racket" and result.racket_parameters is not None:
        return animate_racket_impact(
            result.trajectory,
            result.racket_parameters,
            save=str(output_path),
            ffmpeg_path=ffmpeg_path,
            fps=fps,
        )
    raise ValueError(f"Search result cannot be rendered for mode {result.mode!r}.")
