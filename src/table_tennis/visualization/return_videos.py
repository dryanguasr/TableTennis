"""Generate one MP4 for each validated pilot serve-return trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..constants import DT, PITCH, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH, YAW
from ..exchange import ServiceTargets, StrokeTargets
from ..physics import simulate_racket_impact
from ..presets.returns import (
    PILOT_SERVICE_PARAMS,
    PROFILE_TARGETS,
    PROFILE_TARGET_X,
)
from ..search.returns import (
    build_return_preset,
)
from ..validation import validate_return, validate_service
from .animation import (
    pre_impact_ball_path,
    racket_gesture_path,
    resolve_ffmpeg_path,
)
from .plotting import (
    draw_ball,
    draw_racket,
    draw_table,
)


DEFAULT_VIDEO_DIR = Path("outputs/benchmarks/returns")


def _timeline_indices(
    service_result,
    contact_index: int,
    return_result,
    fps: int,
    max_duration: float,
) -> tuple[int, list[int], list[int]]:
    """Build a real-time timeline capped by duration or the first Z < 0 sample."""

    pre_duration = 0.12
    contact_time = float(service_result.t[contact_index])
    pre_frames = max(1, round(pre_duration * fps))
    service_frames = max(1, round(contact_time * fps))
    total_frame_cap = max(1, round(max_duration * fps))
    remaining_frames = max(1, total_frame_cap - pre_frames - service_frames)
    available_return_time = remaining_frames / fps

    below_floor = np.where(return_result.x[2] < 0.0)[0]
    if below_floor.size and return_result.t[below_floor[0]] < available_return_time:
        return_duration = float(return_result.t[below_floor[0]])
        return_frames = max(1, round(return_duration * fps))
    else:
        return_duration = available_return_time
        return_frames = remaining_frames

    service_indices = np.rint(
        np.linspace(0, contact_index, service_frames)
    ).astype(int).tolist()
    return_stop = min(
        return_result.x.shape[1] - 1,
        round(return_duration / DT),
    )
    return_indices = np.rint(
        np.linspace(0, return_stop, return_frames)
    ).astype(int).tolist()
    return pre_frames, service_indices, return_indices


def _set_axes(ax, title: str) -> None:
    ax.set_xlim(-500, TABLE_LENGTH + 500)
    ax.set_ylim(-400, TABLE_WIDTH + 400)
    ax.set_zlim(0, 1900)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(PITCH, YAW)
    ax.set_title(title)
    ax.grid(True)


def save_exchange_video(
    profile: str | None,
    stroke_side: str,
    output_path: Path,
    ffmpeg_path: str | None = None,
    fps: int = 30,
    max_duration: float = 5.0,
    *,
    service_params=None,
    service_result=None,
    contact=None,
    contact_index: int | None = None,
    return_params=None,
    service_targets: ServiceTargets | None = None,
    return_targets: StrokeTargets | None = None,
    video_title: str | None = None,
) -> None:
    """Render a validated preset or custom serve-return exchange."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if profile is not None:
        depth, spin = PROFILE_TARGETS[profile]
        service_params = PILOT_SERVICE_PARAMS
        service_result = simulate_racket_impact(service_params, t_max=3.0)
        contact, contact_index, return_params = build_return_preset(
            profile,
            stroke_side,
            service_result,
        )
        service_targets = ServiceTargets()
        return_targets = StrokeTargets(
            depth=depth,
            direction="elbow",
            spin_rps=spin,
            stroke_side=stroke_side,
            target_x=PROFILE_TARGET_X.get(profile),
        )
    required = {
        "service_params": service_params,
        "service_result": service_result,
        "contact": contact,
        "contact_index": contact_index,
        "return_params": return_params,
        "service_targets": service_targets,
        "return_targets": return_targets,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Custom exchange video is missing: " + ", ".join(missing)
        )

    return_result = simulate_racket_impact(return_params, t_max=max_duration)
    service_report = validate_service(
        service_params,
        service_result,
        service_targets,
    )
    return_report = validate_return(return_params, return_result, return_targets)
    if not service_report.passed or not return_report.passed:
        raise RuntimeError(
            f"Refusing to render an invalid exchange: service={service_report.violations}, "
            f"return={return_report.violations}"
        )

    resolved_ffmpeg = resolve_ffmpeg_path(ffmpeg_path)
    if resolved_ffmpeg is None:
        raise RuntimeError(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path."
        )
    plt.rcParams["animation.ffmpeg_path"] = resolved_ffmpeg
    pre_frames, service_indices, return_indices = _timeline_indices(
        service_result,
        contact_index,
        return_result,
        fps,
        max_duration,
    )

    service_racket_path = racket_gesture_path(
        service_params.ball_position,
        service_params.racket_velocity,
    )
    return_racket_path = racket_gesture_path(
        return_params.ball_position,
        return_params.racket_velocity,
    )
    incoming_service_ball = pre_impact_ball_path(
        service_params.ball_position,
        service_params.ball_velocity,
        samples=pre_frames,
    )
    total_frames = pre_frames + len(service_indices) + len(return_indices)
    service_contact_racket = len(service_racket_path) // 2 - 1
    return_contact_racket = len(return_racket_path) // 2 - 1

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()
        draw_table(ax)
        ax.plot(
            service_racket_path[:, 0],
            service_racket_path[:, 1],
            service_racket_path[:, 2],
            color="tab:red",
            linestyle=":",
            alpha=0.55,
        )
        ax.plot(
            return_racket_path[:, 0],
            return_racket_path[:, 1],
            return_racket_path[:, 2],
            color="tab:green",
            linestyle=":",
            alpha=0.55,
        )

        if frame < pre_frames:
            ball_center = incoming_service_ball[frame]
            racket_index = int(frame / max(1, pre_frames - 1) * service_contact_racket)
            draw_racket(
                ax,
                service_racket_path[racket_index],
                service_params.racket_angle,
            )
            title = "Preparación del servicio pendular invertido"
        elif frame < pre_frames + len(service_indices):
            local = frame - pre_frames
            service_index = service_indices[local]
            ball_center = service_result.x[:, service_index]
            ax.plot(
                service_result.x[0, : service_index + 1],
                service_result.x[1, : service_index + 1],
                service_result.x[2, : service_index + 1],
                color="tab:orange",
                linewidth=2,
            )
            service_progress = local / max(1, len(service_indices) - 1)
            racket_index = min(
                service_contact_racket + 1 + round(service_progress * (len(service_racket_path) - service_contact_racket - 2)),
                len(service_racket_path) - 1,
            )
            draw_racket(
                ax,
                service_racket_path[racket_index],
                service_params.racket_angle,
            )
            if local >= max(0, len(service_indices) - return_contact_racket - 1):
                approach = local - max(0, len(service_indices) - return_contact_racket - 1)
                draw_racket(
                    ax,
                    return_racket_path[min(approach, return_contact_racket)],
                    return_params.racket_angle,
                )
            title = f"Servicio hasta el punto {contact.moment}"
        else:
            local = frame - pre_frames - len(service_indices)
            return_index = return_indices[local]
            ball_center = return_result.x[:, return_index]
            ax.plot(
                service_result.x[0, : contact_index + 1],
                service_result.x[1, : contact_index + 1],
                service_result.x[2, : contact_index + 1],
                color="tab:orange",
                linewidth=2,
            )
            ax.plot(
                return_result.x[0, : return_index + 1],
                return_result.x[1, : return_index + 1],
                return_result.x[2, : return_index + 1],
                color="tab:purple",
                linewidth=2,
            )
            return_progress = local / max(1, len(return_indices) - 1)
            racket_index = min(
                return_contact_racket + 1 + round(return_progress * (len(return_racket_path) - return_contact_racket - 2)),
                len(return_racket_path) - 1,
            )
            draw_racket(ax, return_racket_path[racket_index], return_params.racket_angle)
            title = video_title or (
                f"{profile} · {stroke_side}"
                if profile is not None
                else f"Servicio y recepción · {stroke_side}"
            )

        draw_ball(ax, ball_center)
        _set_axes(ax, title)

    movie = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        movie.save(output_path, writer=animation.FFMpegWriter(fps=fps))
    finally:
        plt.close(fig)


def iter_cases(selected_profiles: list[str] | None = None):
    profiles = selected_profiles or list(PROFILE_TARGETS)
    for profile in profiles:
        for stroke_side in ("forehand", "backhand"):
            yield profile, stroke_side


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--ffmpeg", help="Path or executable name for FFmpeg.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Maximum MP4 duration in seconds; stops earlier when ball Z falls below zero.",
    )
    parser.add_argument("--profile", action="append", choices=list(PROFILE_TARGETS))
    parser.add_argument("--limit", type=int, help="Render only the first N selected cases.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and print paths without rendering.")
    args = parser.parse_args(argv)

    ffmpeg = resolve_ffmpeg_path(args.ffmpeg)
    if not args.dry_run and ffmpeg is None:
        raise SystemExit(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )

    cases = list(iter_cases(args.profile))
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    service_result = simulate_racket_impact(PILOT_SERVICE_PARAMS, t_max=3.0)
    for profile, stroke_side in cases:
        output_path = args.video_dir / f"{profile}_{stroke_side}.mp4"
        depth, spin = PROFILE_TARGETS[profile]
        _, _, params = build_return_preset(profile, stroke_side, service_result)
        report = validate_return(
            params,
            simulate_racket_impact(params, t_max=3.0),
            StrokeTargets(
                depth=depth,
                spin_rps=spin,
                stroke_side=stroke_side,
                target_x=PROFILE_TARGET_X.get(profile),
            ),
        )
        if not report.passed:
            raise SystemExit(f"{profile}/{stroke_side} is invalid: {report.violations}")
        if args.dry_run:
            print(f"VALID {profile}/{stroke_side} -> {output_path}")
        elif output_path.exists() and not args.overwrite:
            print(f"SKIP {output_path}")
        else:
            print(f"RENDER {output_path}")
            save_exchange_video(
                profile,
                stroke_side,
                output_path,
                ffmpeg,
                args.fps,
                args.duration,
            )


if __name__ == "__main__":
    main()
