"""Animation and MP4 export for simulations and racket impacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import DT, PITCH, PLOT_PERIOD, TABLE_LENGTH, TABLE_WIDTH, YAW
from ..models import RacketImpactParameters, SimulationResult
from .plotting import draw_ball, draw_racket, draw_table, set_scene_axes


def unit_vector(vector, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def _bezier(points: np.ndarray, value: float) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    return _bezier((1.0 - value) * points[:-1] + value * points[1:], value)


def _cubic_bezier_segment(
    p0,
    p3,
    v0,
    v3,
    duration: float,
    samples: int,
) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p1 = p0 + np.asarray(v0, dtype=float) * duration / 3.0
    p2 = p3 - np.asarray(v3, dtype=float) * duration / 3.0
    points = np.array([p0, p1, p2, p3])
    return np.array([_bezier(points, value) for value in np.linspace(0, 1, samples)])


def racket_gesture_path(
    contact_point,
    racket_velocity,
    samples: int = 40,
) -> np.ndarray:
    """Build a short racket path around impact."""

    direction = unit_vector(racket_velocity)
    contact_point = np.asarray(contact_point, dtype=float)
    start = contact_point - 300.0 * direction
    end = contact_point + 300.0 * direction
    contact_velocity = np.asarray(racket_velocity, dtype=float)
    duration = 0.08
    before = _cubic_bezier_segment(
        start,
        contact_point,
        (0, 0, 0),
        contact_velocity,
        duration,
        samples // 2,
    )
    after = _cubic_bezier_segment(
        contact_point,
        end,
        contact_velocity,
        (0, 0, 0),
        duration,
        samples - samples // 2,
    )
    return np.vstack([before[:-1], after])


def pre_impact_ball_path(
    contact_point,
    ball_velocity,
    samples: int = 24,
    duration: float = 0.12,
) -> np.ndarray:
    """Build an incoming path ending at the racket contact point."""

    contact_point = np.asarray(contact_point, dtype=float)
    ball_velocity = np.asarray(ball_velocity, dtype=float)
    start = (
        contact_point
        - unit_vector(ball_velocity, fallback=(-1.0, 0.0, 0.0))
        * np.linalg.norm(ball_velocity)
        * duration
    )
    return np.linspace(start, contact_point, samples)


def resolve_ffmpeg_path(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """Return an executable FFmpeg path when MP4 output is available."""

    if ffmpeg_path:
        path = Path(ffmpeg_path)
        if path.exists():
            return str(path)
        return shutil.which(ffmpeg_path)
    return shutil.which("ffmpeg")


def animate_simulation(
    result: SimulationResult,
    save: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    racket_path=None,
    racket_angle=(0.0, -10.0, 0.0),
    fps: int | None = None,
) -> Path | None:
    """Animate a simulated trajectory."""

    if save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ffmpeg = resolve_ffmpeg_path(ffmpeg_path) if save else None
    if save and ffmpeg is None:
        raise RuntimeError(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        draw_table(ax)
        draw_ball(ax, result.x[:, frame])
        position = result.x[:, frame]
        ax.quiver(*position, *result.v[:, frame], length=10 / 98, color="g")
        ax.quiver(*position, *result.a[:, frame], length=1 / 49, color="c")
        ax.quiver(*position, *result.omega[:, frame], length=1, color="r")
        racket_center = (120.0, TABLE_WIDTH * 5 / 8, 1060.0)
        if racket_path is not None:
            path = np.asarray(racket_path, dtype=float)
            path_index = int(
                round(frame / max(1, result.x.shape[1] - 1) * (len(path) - 1))
            )
            racket_center = path[min(path_index, len(path) - 1)]
            ax.plot(*path.T, color="black", linestyle=":", alpha=0.65)
        draw_racket(ax, racket_center, racket_angle)
        set_scene_axes(ax, YAW, PITCH)

    dt = (
        float(result.t[1] - result.t[0])
        if result.t is not None and len(result.t) > 1
        else DT
    )
    frame_step = (
        max(1, int(round(1.0 / (dt * fps))))
        if fps is not None
        else PLOT_PERIOD
    )
    writer_fps = fps or max(1, int(round(1.0 / (dt * frame_step))))
    frames = range(0, result.x.shape[1], frame_step)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / writer_fps,
    )
    if save:
        output_path = Path(save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            movie.save(
                output_path,
                writer=animation.FFMpegWriter(fps=writer_fps),
            )
        finally:
            plt.close(fig)
        return output_path
    plt.show()
    return None


def animate_racket_impact(
    result: SimulationResult,
    params: RacketImpactParameters,
    save: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    fps: int | None = None,
) -> Path | None:
    """Animate an impact including synthetic pre-contact frames."""

    if save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    ffmpeg = resolve_ffmpeg_path(ffmpeg_path) if save else None
    if save and ffmpeg is None:
        raise RuntimeError(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg

    racket_path = racket_gesture_path(params.ball_position, params.racket_velocity)
    incoming_ball = pre_impact_ball_path(params.ball_position, params.ball_velocity)
    dt = (
        float(result.t[1] - result.t[0])
        if result.t is not None and len(result.t) > 1
        else DT
    )
    frame_step = (
        max(1, int(round(1.0 / (dt * fps))))
        if fps is not None
        else PLOT_PERIOD
    )
    writer_fps = fps or max(1, int(round(1.0 / (dt * frame_step))))
    post_frames = list(range(0, result.x.shape[1], frame_step))
    if post_frames[-1] != result.x.shape[1] - 1:
        post_frames.append(result.x.shape[1] - 1)
    total_frames = len(incoming_ball) + len(post_frames)
    contact_path_index = len(racket_path) // 2 - 1

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        draw_table(ax)
        ax.plot(*racket_path.T, color="black", linestyle=":", alpha=0.65)
        ax.plot(*incoming_ball.T, color="orange", linestyle="--", alpha=0.7)
        if frame < len(incoming_ball):
            ball_center = incoming_ball[frame]
            racket_index = int(
                frame / max(1, len(incoming_ball) - 1) * contact_path_index
            )
        else:
            post_frame = post_frames[frame - len(incoming_ball)]
            ball_center = result.x[:, post_frame]
            progress = (frame - len(incoming_ball)) / max(1, len(post_frames) - 1)
            racket_index = min(
                contact_path_index
                + 1
                + int(
                    progress * (len(racket_path) - contact_path_index - 1)
                ),
                len(racket_path) - 1,
            )
            ax.plot(
                *result.x[:, : post_frame + 1],
                color="purple",
            )
        draw_racket(ax, racket_path[racket_index], params.racket_angle)
        draw_ball(ax, ball_center)
        set_scene_axes(ax, YAW, PITCH)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / writer_fps,
    )
    if save:
        output_path = Path(save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            movie.save(
                output_path,
                writer=animation.FFMpegWriter(fps=writer_fps),
            )
        finally:
            plt.close(fig)
        return output_path
    plt.show()
    return None
