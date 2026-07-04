"""Continuous MP4 rendering and resumable batch generation for exercises."""

from __future__ import annotations

import argparse
import bisect
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..constants import PITCH, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH, YAW
from ..physics import racket_normal, rotation_matrix_xyz
from ..presets.exercises import EXERCISE_NAMES, build_exercise
from ..rally import ExerciseResult, simulate_exercise, wing_y
from .animation import (
    pre_impact_ball_path,
    resolve_ffmpeg_path,
    unit_vector,
)
from .plotting import draw_ball, draw_racket, draw_table


DEFAULT_OUTPUT_DIR = Path("outputs/exercises")
PRE_ROLL_SECONDS = 0.55
MAX_APPROACH_SECONDS = 0.45
MAX_RECOVERY_SECONDS = 0.45
PREPARATION_LEAD_SECONDS = 0.18
FOLLOW_THROUGH_SECONDS = 0.18
PLAYBACK_SLOWDOWN = 2.0


@dataclass(frozen=True)
class ExerciseVideoJob:
    exercise: str
    cycles: int
    output_path: str
    fps: int
    ffmpeg_path: str


@dataclass(frozen=True)
class ExerciseVideoResult:
    exercise: str
    path: str
    status: str
    duration: float | None = None
    error: str = ""


@dataclass(frozen=True)
class RacketMotionTrack:
    """One smooth stand-by-to-stand-by stroke motion."""

    player: str
    stroke_index: int
    times: tuple[float, ...]
    positions: np.ndarray
    angles: np.ndarray


def _contact_times(result: ExerciseResult) -> list[float]:
    times = [0.0]
    elapsed = 0.0
    for segment in result.segments[:-1]:
        elapsed += float(segment.result.t[segment.stop_index])
        times.append(elapsed)
    return times


def standby_pose(
    player: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Neutral ready pose halfway across the backhand quadrant."""

    if player == "near":
        center = (-330.0, TABLE_WIDTH * 0.75, TABLE_HEIGHT + 270.0)
        # Local handle points along -Y. A -90-degree Z rotation points it
        # toward negative X, away from the table.
        angle = (0.0, -8.0, -90.0)
    else:
        center = (
            TABLE_LENGTH + 330.0,
            TABLE_WIDTH * 0.25,
            TABLE_HEIGHT + 270.0,
        )
        angle = (0.0, -8.0, 90.0)
    return center, angle


def _unwrap_angle(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Choose Euler-angle representatives nearest to the reference."""

    return reference + (target - reference + 180.0) % 360.0 - 180.0


def visual_racket_angle(
    player: str,
    wing: str,
    physical_angle,
) -> tuple[float, float, float]:
    """Rotate the handle around the face normal for a right-handed stroke."""

    _, angle_y, angle_z = (
        float(value) for value in physical_angle
    )
    # Looking toward the table, a right-handed player's handle points left on
    # the forehand and right on the backhand. Those directions are mirrored
    # between the two ends of the table.
    left_y = 1.0 if player == "near" else -1.0
    desired_y = left_y if wing == "forehand" else -left_y
    desired = np.array([0.0, desired_y, 0.0])
    normal = racket_normal((0.0, angle_y, angle_z))
    target = desired - float(np.dot(desired, normal)) * normal
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-9:
        target = desired
    else:
        target /= target_norm

    best_angle = 0.0
    best_score = float("-inf")
    local_handle = np.array([0.0, -1.0, 0.0])
    for angle_x in np.linspace(-180.0, 180.0, 721):
        handle = (
            rotation_matrix_xyz((angle_x, angle_y, angle_z))
            @ local_handle
        )
        score = float(np.dot(handle, target))
        if score > best_score:
            best_score = score
            best_angle = float(angle_x)
    return best_angle, angle_y, angle_z


def _bezier_track_value(
    times: tuple[float, ...],
    values: np.ndarray,
    time_value: float,
) -> np.ndarray:
    """Evaluate a C1 piecewise cubic Bézier curve through every key node."""

    time_array = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_value <= time_array[0]:
        return values[0].copy()
    if time_value >= time_array[-1]:
        return values[-1].copy()

    tangents = np.zeros_like(values)
    for index in range(1, len(values) - 1):
        duration = time_array[index + 1] - time_array[index - 1]
        tangents[index] = (
            values[index + 1] - values[index - 1]
        ) / max(duration, 1e-9)
    segment = int(np.searchsorted(time_array, time_value) - 1)
    duration = time_array[segment + 1] - time_array[segment]
    progress = (time_value - time_array[segment]) / duration
    p0 = values[segment]
    p3 = values[segment + 1]
    p1 = p0 + tangents[segment] * duration / 3.0
    p2 = p3 - tangents[segment + 1] * duration / 3.0
    inverse = 1.0 - progress
    return (
        inverse**3 * p0
        + 3.0 * inverse**2 * progress * p1
        + 3.0 * inverse * progress**2 * p2
        + progress**3 * p3
    )


def _stroke_motion_track(
    result: ExerciseResult,
    stroke_index: int,
    contact_times: list[float],
    rally_duration: float,
) -> RacketMotionTrack:
    """Build stand by → preparation → impact → finish → stand by."""

    segment = result.segments[stroke_index]
    player = segment.stroke.hitter
    contact_time = contact_times[stroke_index]
    player_contacts = [
        index
        for index, candidate in enumerate(result.segments)
        if candidate.stroke.hitter == player
    ]
    player_position = player_contacts.index(stroke_index)
    previous_time = (
        contact_times[player_contacts[player_position - 1]]
        if player_position > 0
        else -PRE_ROLL_SECONDS
    )
    next_time = (
        contact_times[player_contacts[player_position + 1]]
        if player_position + 1 < len(player_contacts)
        else rally_duration
    )
    approach = min(
        MAX_APPROACH_SECONDS,
        (
            contact_time - previous_time
            if player_position == 0
            else 0.48 * (contact_time - previous_time)
        ),
    )
    recovery = min(
        MAX_RECOVERY_SECONDS,
        (
            next_time - contact_time
            if player_position + 1 == len(player_contacts)
            else 0.48 * (next_time - contact_time)
        ),
    )
    approach = max(0.20, approach)
    recovery = max(0.20, recovery)
    preparation_lead = min(
        PREPARATION_LEAD_SECONDS,
        approach * 0.48,
    )
    follow_time = min(
        FOLLOW_THROUGH_SECONDS,
        recovery * 0.48,
    )
    times = (
        contact_time - approach,
        contact_time - preparation_lead,
        contact_time,
        contact_time + follow_time,
        contact_time + recovery,
    )

    standby_center, standby_angle = standby_pose(player)
    impact = np.asarray(segment.params.ball_position, dtype=float)
    direction = unit_vector(segment.params.racket_velocity)
    preparation = impact - 300.0 * direction
    finish = impact + 300.0 * direction
    positions = np.asarray(
        [
            standby_center,
            preparation,
            impact,
            finish,
            standby_center,
        ],
        dtype=float,
    )

    ready_angle = np.asarray(standby_angle, dtype=float)
    impact_angle = _unwrap_angle(
        np.asarray(
            visual_racket_angle(
                player,
                segment.stroke.wing,
                segment.params.racket_angle,
            )
        ),
        ready_angle,
    )
    preparation_angle = ready_angle + 0.72 * (
        impact_angle - ready_angle
    )
    side_rotation = -12.0 if segment.stroke.wing == "forehand" else 12.0
    finish_angle = impact_angle + np.array(
        [
            side_rotation,
            -6.0 * np.sign(segment.params.racket_velocity[2]),
            8.0 * np.sign(segment.params.racket_velocity[1]),
        ]
    )
    final_ready_angle = _unwrap_angle(ready_angle, finish_angle)
    angles = np.asarray(
        [
            ready_angle,
            preparation_angle,
            impact_angle,
            finish_angle,
            final_ready_angle,
        ]
    )
    return RacketMotionTrack(
        player=player,
        stroke_index=stroke_index,
        times=times,
        positions=positions,
        angles=angles,
    )


def build_racket_motion_tracks(
    result: ExerciseResult,
    contact_times: list[float] | None = None,
) -> dict[str, tuple[RacketMotionTrack, ...]]:
    """Precompute non-overlapping Bézier motion tracks for both players."""

    contacts = contact_times or _contact_times(result)
    final_duration = float(
        result.segments[-1].result.t[result.segments[-1].stop_index]
    )
    rally_duration = contacts[-1] + final_duration
    tracks = {"near": [], "far": []}
    for index in range(len(result.segments)):
        track = _stroke_motion_track(
            result,
            index,
            contacts,
            rally_duration,
        )
        tracks[track.player].append(track)
    return {
        player: tuple(player_tracks)
        for player, player_tracks in tracks.items()
    }


def _racket_pose(
    player: str,
    time_value: float,
    tracks: dict[str, tuple[RacketMotionTrack, ...]],
):
    for track in tracks[player]:
        if track.times[0] <= time_value <= track.times[-1]:
            center = _bezier_track_value(
                track.times,
                track.positions,
                time_value,
            )
            angle = _bezier_track_value(
                track.times,
                track.angles,
                time_value,
            )
            return tuple(center), tuple(angle)
    return standby_pose(player)


def save_exercise_video(
    result: ExerciseResult,
    output_path: Path | str,
    *,
    ffmpeg_path: str | None = None,
    fps: int = 30,
) -> Path:
    """Render a validated exercise as one continuous, physically timed MP4."""

    if not result.passed:
        raise RuntimeError(
            "Refusing to render an invalid exercise: "
            + "; ".join(result.violations)
        )
    resolved_ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    if resolved_ffmpeg is None:
        raise RuntimeError(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path."
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams["animation.ffmpeg_path"] = resolved_ffmpeg
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    contacts = _contact_times(result)
    final_duration = float(
        result.segments[-1].result.t[result.segments[-1].stop_index]
    )
    rally_duration = contacts[-1] + final_duration
    total_duration = PLAYBACK_SLOWDOWN * (
        PRE_ROLL_SECONDS + rally_duration
    )
    total_frames = max(1, int(np.ceil(total_duration * fps)))
    racket_tracks = build_racket_motion_tracks(result, contacts)
    first = result.segments[0]
    incoming_ball = pre_impact_ball_path(
        first.params.ball_position,
        first.params.ball_velocity,
        samples=max(2, int(round(PRE_ROLL_SECONDS * fps))),
        duration=PRE_ROLL_SECONDS,
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        draw_table(ax)
        absolute_time = (
            frame / fps / PLAYBACK_SLOWDOWN - PRE_ROLL_SECONDS
        )
        if absolute_time < 0.0:
            progress = np.clip(
                (absolute_time + PRE_ROLL_SECONDS) / PRE_ROLL_SECONDS,
                0.0,
                1.0,
            )
            ball_index = min(
                len(incoming_ball) - 1,
                int(round(progress * (len(incoming_ball) - 1))),
            )
            ball_center = incoming_ball[ball_index]
            stroke_index = 0
            local_index = 0
        else:
            stroke_index = min(
                len(result.segments) - 1,
                bisect.bisect_right(contacts, absolute_time) - 1,
            )
            local_time = max(0.0, absolute_time - contacts[stroke_index])
            segment = result.segments[stroke_index]
            local_index = min(
                segment.stop_index,
                int(np.searchsorted(segment.result.t, local_time)),
            )
            ball_center = segment.result.x[:, local_index]
            trail_start = max(0, local_index - 90)
            ax.plot(
                segment.result.x[0, trail_start : local_index + 1],
                segment.result.x[1, trail_start : local_index + 1],
                segment.result.x[2, trail_start : local_index + 1],
                color="tab:purple",
                linewidth=2.0,
            )

        for player, color in (("near", "tab:red"), ("far", "tab:green")):
            center, angle = _racket_pose(
                player,
                absolute_time,
                racket_tracks,
            )
            draw_racket(ax, center, angle)
            ax.scatter(*center, color=color, s=12, alpha=0.8)
        draw_ball(ax, ball_center)
        stroke = result.segments[stroke_index].stroke
        cycle_text = f" · vuelta {stroke.cycle}" if stroke.cycle else ""
        ax.set_title(
            f"{result.definition.title}\n"
            f"{stroke.label or stroke.kind}{cycle_text}"
        )
        ax.set_xlim(-650, TABLE_LENGTH + 650)
        ax.set_ylim(-450, TABLE_WIDTH + 450)
        ax.set_zlim(0, 1900)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(PITCH, YAW)
        ax.grid(True)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
    )
    try:
        movie.save(output, writer=animation.FFMpegWriter(fps=fps))
    finally:
        plt.close(fig)
    return output


def build_exercise_video_jobs(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    cycles: int = 3,
    fps: int = 30,
    ffmpeg_path: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[ExerciseVideoJob]:
    selected = EXERCISE_NAMES if names is None else tuple(names)
    return [
        ExerciseVideoJob(
            exercise=name,
            cycles=cycles,
            output_path=str(output_dir / f"{name}.mp4"),
            fps=fps,
            ffmpeg_path=ffmpeg_path,
        )
        for name in selected
    ]


def render_exercise_video_job(
    job: ExerciseVideoJob,
) -> ExerciseVideoResult:
    """Calibrate and atomically render one pickleable exercise job."""

    from ..benchmarks.videos import probe_mp4_duration, resolve_ffprobe_path

    output = Path(job.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    try:
        result = simulate_exercise(
            build_exercise(job.exercise, cycles=job.cycles)
        )
        if not result.passed:
            raise RuntimeError("; ".join(result.violations))
        save_exercise_video(
            result,
            temporary,
            ffmpeg_path=job.ffmpeg_path,
            fps=job.fps,
        )
        duration = probe_mp4_duration(
            temporary,
            resolve_ffprobe_path(job.ffmpeg_path),
        )
        if duration is None:
            raise RuntimeError("FFprobe could not validate the rendered MP4.")
        temporary.replace(output)
        return ExerciseVideoResult(
            job.exercise,
            str(output),
            "rendered",
            duration,
        )
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        return ExerciseVideoResult(
            job.exercise,
            str(output),
            "failed",
            error=str(exc),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="table-tennis generate exercise-videos",
        description=__doc__,
    )
    parser.add_argument(
        "--exercise",
        action="append",
        choices=EXERCISE_NAMES,
        help="Exercise to render; repeat it or omit it for all.",
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "manifest.json",
    )
    args = parser.parse_args(argv)
    if args.cycles < 3:
        parser.error("--cycles must be at least 3")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be positive")
    ffmpeg = resolve_ffmpeg_path(args.ffmpeg)
    if ffmpeg is None and not args.dry_run:
        parser.error("MP4 output requires FFmpeg; install it or pass --ffmpeg.")
    ffmpeg = ffmpeg or args.ffmpeg or "ffmpeg"
    jobs = build_exercise_video_jobs(
        args.exercise,
        cycles=args.cycles,
        fps=args.fps,
        ffmpeg_path=ffmpeg,
        output_dir=args.output_dir,
    )
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    from ..benchmarks.videos import probe_mp4_duration, resolve_ffprobe_path

    ffprobe = resolve_ffprobe_path(ffmpeg)
    results: list[ExerciseVideoResult] = []
    pending: list[ExerciseVideoJob] = []
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        duration = probe_mp4_duration(job.output_path, ffprobe)
        if duration is not None and not args.overwrite:
            results.append(
                ExerciseVideoResult(
                    job.exercise,
                    job.output_path,
                    "skipped",
                    duration,
                )
            )
            print(f"[{index}/{total}] SKIP {job.output_path}")
        elif args.dry_run:
            status = "OVERWRITE" if duration is not None else "RENDER"
            results.append(
                ExerciseVideoResult(
                    job.exercise,
                    job.output_path,
                    "dry-run",
                    duration,
                )
            )
            print(f"[{index}/{total}] {status} {job.output_path}")
        else:
            pending.append(job)

    completed = total - len(pending)
    if args.workers == 1:
        for job in pending:
            result = render_exercise_video_job(job)
            completed += 1
            results.append(result)
            print(
                f"[{completed}/{total}] {result.status.upper()} {result.path}"
            )
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(render_exercise_video_job, job): job
                for job in pending
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                results.append(result)
                print(
                    f"[{completed}/{total}] "
                    f"{result.status.upper()} {result.path}"
                )
    results.sort(key=lambda item: item.exercise)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            [asdict(result) for result in results],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    failures = [result for result in results if result.status == "failed"]
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("rendered", "skipped", "dry-run", "failed")
    }
    print(
        "summary "
        + " ".join(f"{name}={value}" for name, value in counts.items())
        + f" manifest={args.manifest}"
    )
    if failures:
        for failure in failures:
            print(f"ERROR {failure.path}: {failure.error}")
        raise SystemExit(1)


__all__ = [
    "ExerciseVideoJob",
    "ExerciseVideoResult",
    "build_exercise_video_jobs",
    "render_exercise_video_job",
    "save_exercise_video",
]
