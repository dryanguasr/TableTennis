"""Benchmark common table-tennis serves from racket-impact parameters.

This script benchmarks the impact model that includes racket movement. It covers
typical serve families in 9 variants each: short, two-bounce ("enfermería") and
long, aimed to forehand, elbow and backhand lanes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import benchmark_direct_services as direct_services
from table_tennis_simulation import (
    BALL_RADIUS,
    G,
    RacketImpactParameters,
    TABLE_HEIGHT,
    TABLE_WIDTH,
    animate_simulation,
    racket_normal,
    racket_gesture_path,
    resolve_ffmpeg_path,
    simulate_racket_impact,
)

SERVICE_TYPES = {
    "pendulum": {"angle": (8.0, -34.0, -22.0)},
    "reverse_pendulum": {"angle": (-8.0, -34.0, 22.0)},
    "hook": {"angle": (4.0, -38.0, -35.0)},
    "tomahawk": {"angle": (35.0, -30.0, 15.0)},
    "reverse_tomahawk": {"angle": (-35.0, -30.0, -15.0)},
    "backhand_standard": {"angle": (-5.0, -36.0, 12.0)},
}
DEPTHS = {
    "short": {"speed_scale": 0.78, "restitution": 0.72, "friction": 0.82},
    "two_bounce": {"speed_scale": 0.90, "restitution": 0.80, "friction": 0.72},
    "long": {"speed_scale": 1.12, "restitution": 0.92, "friction": 0.58},
}
TOSS_SPEEDS = {
    "short": float(np.sqrt(2.0 * G * 0.16 * 1000.0)),
    "two_bounce": 3500.0,
    "long": 5200.0,
}
LANES = {
    "forehand": {"y": TABLE_WIDTH * 0.72, "vy_bias": 320.0},
    "elbow": {"y": TABLE_WIDTH * 0.50, "vy_bias": 0.0},
    "backhand": {"y": TABLE_WIDTH * 0.28, "vy_bias": -320.0},
}


@dataclass(frozen=True)
class BenchmarkCase:
    service: str
    depth: str
    lane: str
    params: RacketImpactParameters


def _racket_velocity_for_post_velocity(
    post_velocity: tuple[float, float, float],
    incoming_velocity: tuple[float, float, float],
    racket_angle: tuple[float, float, float],
    friction: float,
    restitution: float,
) -> tuple[float, float, float]:
    normal = racket_normal(racket_angle)
    normal_projection = np.outer(normal, normal)
    tangent_projection = np.eye(3) - normal_projection
    impact_matrix = (1.0 - friction) * tangent_projection - restitution * normal_projection
    racket_matrix = np.eye(3) - impact_matrix
    racket_velocity = np.linalg.solve(
        racket_matrix,
        np.array(post_velocity, dtype=float) - impact_matrix @ np.array(incoming_velocity, dtype=float),
    )
    return tuple(float(x) for x in racket_velocity)


def _preimpact_spin_for_post_spin(
    post_spin: tuple[float, float, float],
    incoming_velocity: tuple[float, float, float],
    racket_velocity: tuple[float, float, float],
    racket_angle: tuple[float, float, float],
    friction: float,
) -> tuple[float, float, float]:
    normal = racket_normal(racket_angle)
    contact_radius = -BALL_RADIUS * normal
    relative_velocity = np.array(incoming_velocity, dtype=float) - np.array(racket_velocity, dtype=float)
    relative_tangent = relative_velocity - np.dot(relative_velocity, normal) * normal

    def post_for(ball_spin: np.ndarray) -> np.ndarray:
        surface_slip = relative_tangent + np.cross(ball_spin, contact_radius)
        return ball_spin - friction * np.cross(contact_radius, surface_slip) / (BALL_RADIUS**2)

    zero = np.zeros(3)
    offset = post_for(zero)
    matrix = np.column_stack([post_for(np.eye(3)[i]) - offset for i in range(3)])
    pre_spin = np.linalg.solve(matrix, np.array(post_spin, dtype=float) - offset)
    return tuple(float(x) for x in pre_spin)


def _racket_params_from_direct_case(case: direct_services.BenchmarkCase) -> RacketImpactParameters:
    service = SERVICE_TYPES[case.service]
    depth = DEPTHS[case.depth]
    incoming_velocity = (0.0, 0.0, -TOSS_SPEEDS[case.depth])
    racket_angle = service["angle"]
    racket_velocity = _racket_velocity_for_post_velocity(
        case.initial_conditions.vel,
        incoming_velocity,
        racket_angle,
        depth["friction"],
        depth["restitution"],
    )
    ball_omega = _preimpact_spin_for_post_spin(
        case.initial_conditions.omega,
        incoming_velocity,
        racket_velocity,
        racket_angle,
        depth["friction"],
    )
    return RacketImpactParameters(
        ball_velocity=incoming_velocity,
        ball_omega=ball_omega,
        rubber_friction=depth["friction"],
        rubber_restitution=depth["restitution"],
        racket_angle=racket_angle,
        racket_velocity=racket_velocity,
        ball_position=case.initial_conditions.pos,
    )


def build_cases() -> Iterable[BenchmarkCase]:
    for direct_case in direct_services.build_cases():
        yield BenchmarkCase(
            direct_case.service,
            direct_case.depth,
            direct_case.lane,
            _racket_params_from_direct_case(direct_case),
        )


def count_table_bounces(result) -> int:
    table_level = TABLE_HEIGHT + BALL_RADIUS
    contacts = np.isclose(result.x[2], table_level, atol=1e-6) & (result.v[2] > 0)
    starts = contacts & np.concatenate(([True], ~contacts[:-1]))
    return int(np.count_nonzero(starts))


def case_filename(case: BenchmarkCase) -> str:
    return f"{case.service}_{case.depth}_{case.lane}.mp4"


def save_case_video(case: BenchmarkCase, result, video_dir: Path, ffmpeg_path: str | None) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / case_filename(case)
    racket_path = racket_gesture_path(case.params.ball_position, case.params.racket_velocity)
    animate_simulation(
        result,
        save=str(path),
        ffmpeg_path=ffmpeg_path,
        racket_path=racket_path,
        racket_angle=case.params.racket_angle,
    )
    return path


def run_case(
    case: BenchmarkCase,
    repeat: int,
    dt: float,
    t_max: float,
    video_dir: Path | None,
    ffmpeg_path: str | None,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = simulate_racket_impact(case.params, dt=dt, t_max=t_max)
    elapsed = time.perf_counter() - start
    assert result is not None
    video_path = save_case_video(case, result, video_dir, ffmpeg_path) if video_dir is not None else None
    return {
        "service": case.service,
        "depth": case.depth,
        "lane": case.lane,
        "repeat": repeat,
        "total_ms": elapsed * 1000.0,
        "avg_ms": elapsed * 1000.0 / repeat,
        "samples": result.x.shape[1],
        "bounces": count_table_bounces(result),
        "initial_x_mm": case.params.ball_position[0],
        "incoming_vx_mm_s": case.params.ball_velocity[0],
        "incoming_vy_mm_s": case.params.ball_velocity[1],
        "incoming_vz_mm_s": case.params.ball_velocity[2],
        "final_z_mm": float(result.x[2, -1]),
        "video": str(video_path) if video_path is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark racket-impact table-tennis serve simulations.")
    parser.add_argument("--repeat", type=int, default=20, help="Runs per serve variant.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation step in seconds.")
    parser.add_argument("--t-max", type=float, default=3.0, help="Simulation horizon in seconds.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("benchmark_racket_services"),
        help="Directory where one MP4 per racket-service benchmark variant is saved.",
    )
    parser.add_argument("--no-video", action="store_true", help="Run the benchmark without saving animations.")
    parser.add_argument("--ffmpeg", help="Path to the FFmpeg executable used for MP4 output.")
    args = parser.parse_args()

    video_dir = None if args.no_video else args.video_dir
    ffmpeg_path = resolve_ffmpeg_path(args.ffmpeg) if video_dir is not None else None
    if video_dir is not None and ffmpeg_path is None:
        parser.error(
            "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
            "or pass its executable path with --ffmpeg."
        )

    print(
        "service,depth,lane,repeat,total_ms,avg_ms,samples,bounces,"
        "initial_x_mm,incoming_vx_mm_s,incoming_vy_mm_s,incoming_vz_mm_s,final_z_mm,video"
    )
    for case in build_cases():
        row = run_case(
            case,
            repeat=args.repeat,
            dt=args.dt,
            t_max=args.t_max,
            video_dir=video_dir,
            ffmpeg_path=ffmpeg_path,
        )
        print(
            f"{row['service']},{row['depth']},{row['lane']},{row['repeat']},"
            f"{row['total_ms']:.3f},{row['avg_ms']:.3f},{row['samples']},{row['bounces']},"
            f"{row['initial_x_mm']:.1f},{row['incoming_vx_mm_s']:.1f},"
            f"{row['incoming_vy_mm_s']:.1f},{row['incoming_vz_mm_s']:.1f},"
            f"{row['final_z_mm']:.1f},{row['video']}"
        )


if __name__ == "__main__":
    main()
