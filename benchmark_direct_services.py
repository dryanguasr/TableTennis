"""Benchmark common table-tennis serves from direct ball initial conditions.

The script runs the existing physics engine by giving the ball an initial
position, linear velocity and angular velocity directly. It covers typical serve
families in 9 variants each: short, two-bounce ("enfermería") and long, aimed to
forehand, elbow and backhand lanes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from table_tennis_simulation import (
    BALL_RADIUS,
    InitialConditions,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
    animate_simulation,
    resolve_ffmpeg_path,
    simulate,
)

SERVICE_TYPES = {
    "pendulum": {
        "position": (120.0, TABLE_WIDTH * 0.28, TABLE_HEIGHT + 310.0),
        "omega": (1.6, -4.2, 10.4),
        "velocity_adjust": (80.0, 0.0, -20.0),
    },
    "reverse_pendulum": {
        "position": (120.0, TABLE_WIDTH * 0.50, TABLE_HEIGHT + 310.0),
        "omega": (-1.6, 4.2, -10.4),
        "velocity_adjust": (-40.0, 0.0, 20.0),
    },
    "hook": {
        "position": (110.0, TABLE_WIDTH * 0.24, TABLE_HEIGHT + 300.0),
        "omega": (0.8, -6.4, 13.6),
        "velocity_adjust": (200.0, 0.0, 0.0),
    },
    "tomahawk": {
        "position": (120.0, TABLE_WIDTH * 0.74, TABLE_HEIGHT + 315.0),
        "omega": (10.4, 2.4, -4.8),
        "velocity_adjust": (60.0, 0.0, -10.0),
    },
    "reverse_tomahawk": {
        "position": (120.0, TABLE_WIDTH * 0.55, TABLE_HEIGHT + 315.0),
        "omega": (-10.4, -2.4, 4.8),
        "velocity_adjust": (40.0, 0.0, 0.0),
    },
    "backhand_standard": {
        "position": (130.0, TABLE_WIDTH * 0.50, TABLE_HEIGHT + 305.0),
        "omega": (-3.6, -1.8, 8.8),
        "depth_velocity": {
            "short": (4400.0, 0.0, -2390.0),
            "two_bounce": (5800.0, 0.0, -1755.0),
            "long": (7255.0, 0.0, -2360.0),
        },
    },
}
DEPTHS = {
    "short": {"vel": (4450.0, 0.0, -2140.0), "target_x": TABLE_LENGTH / 2 + 250.0},
    "two_bounce": {"vel": (5600.0, 0.0, -2395.0), "target_x": TABLE_LENGTH / 2 + 650.0},
    "long": {"vel": (7200.0, 0.0, -2350.0), "target_x": TABLE_LENGTH - 250.0},
}
LANES = {
    "forehand": {"y": TABLE_WIDTH * 0.72},
    "elbow": {"y": TABLE_WIDTH * 0.50},
    "backhand": {"y": TABLE_WIDTH * 0.28},
}
LANE_VY = {
    "pendulum": {
        "short": {"forehand": 2015.0, "elbow": 965.0, "backhand": -30.0},
        "two_bounce": {"forehand": 1965.0, "elbow": 960.0, "backhand": -40.0},
        "long": {"forehand": 1990.0, "elbow": 970.0, "backhand": -50.0},
    },
    "reverse_pendulum": {
        "short": {"forehand": 1030.0, "elbow": 30.0, "backhand": -975.0},
        "two_bounce": {"forehand": 1050.0, "elbow": 40.0, "backhand": -970.0},
        "long": {"forehand": 1070.0, "elbow": 50.0, "backhand": -970.0},
    },
    "hook": {
        "short": {"forehand": 2180.0, "elbow": 1160.0, "backhand": 140.0},
        "two_bounce": {"forehand": 2145.0, "elbow": 1135.0, "backhand": 125.0},
        "long": {"forehand": 2205.0, "elbow": 1155.0, "backhand": 115.0},
    },
    "tomahawk": {
        "short": {"forehand": -30.0, "elbow": -1025.0, "backhand": -2035.0},
        "two_bounce": {"forehand": -20.0, "elbow": -1010.0, "backhand": -2005.0},
        "long": {"forehand": -20.0, "elbow": -1040.0, "backhand": -2060.0},
    },
    "reverse_tomahawk": {
        "short": {"forehand": 710.0, "elbow": -290.0, "backhand": -1290.0},
        "two_bounce": {"forehand": 695.0, "elbow": -290.0, "backhand": -1320.0},
        "long": {"forehand": 715.0, "elbow": -305.0, "backhand": -1325.0},
    },
    "backhand_standard": {
        "short": {"forehand": 940.0, "elbow": -45.0, "backhand": -1035.0},
        "two_bounce": {"forehand": 975.0, "elbow": -50.0, "backhand": -1080.0},
        "long": {"forehand": 970.0, "elbow": -65.0, "backhand": -1100.0},
    },
}
TARGET_MARGIN_MM = 50.0


@dataclass(frozen=True)
class BenchmarkCase:
    service: str
    depth: str
    lane: str
    initial_conditions: InitialConditions
    target: tuple[float, float]


def build_cases() -> Iterable[BenchmarkCase]:
    for service_name, service in SERVICE_TYPES.items():
        for depth_name, depth in DEPTHS.items():
            for lane_name, lane in LANES.items():
                velocity = np.array(service.get("depth_velocity", {}).get(depth_name, depth["vel"]), dtype=float)
                velocity += np.array(service.get("velocity_adjust", (0.0, 0.0, 0.0)), dtype=float)
                velocity[1] = LANE_VY[service_name][depth_name][lane_name]
                position = service["position"]
                yield BenchmarkCase(
                    service_name,
                    depth_name,
                    lane_name,
                    InitialConditions(pos=position, vel=tuple(velocity), omega=service["omega"]),
                    (depth["target_x"], lane["y"]),
                )


def count_table_bounces(result) -> int:
    table_level = TABLE_HEIGHT + BALL_RADIUS
    return int(np.count_nonzero(np.isclose(result.x[2], table_level, atol=1e-6) & (result.v[2] > 0)))


def first_opponent_bounce(result) -> tuple[float, float] | None:
    table_level = TABLE_HEIGHT + BALL_RADIUS
    hits = np.where(
        (result.x[0] >= TABLE_LENGTH / 2)
        & np.isclose(result.x[2], table_level, atol=1e-6)
        & (result.v[2] > 0)
    )[0]
    if hits.size == 0:
        return None

    point = result.x[:2, int(hits[0])]
    return float(point[0]), float(point[1])


def case_filename(case: BenchmarkCase) -> str:
    return f"{case.service}_{case.depth}_{case.lane}.mp4"


def save_case_video(case: BenchmarkCase, result, video_dir: Path, ffmpeg_path: str | None) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / case_filename(case)
    animate_simulation(result, save=str(path), ffmpeg_path=ffmpeg_path)
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
        result = simulate(case.initial_conditions, dt=dt, t_max=t_max)
    elapsed = time.perf_counter() - start
    assert result is not None
    bounce = first_opponent_bounce(result)
    if bounce is None:
        bounce_x = float("nan")
        bounce_y = float("nan")
        target_error = float("inf")
    else:
        bounce_x, bounce_y = bounce
        target_error = float(np.linalg.norm(np.array(bounce) - np.array(case.target)))

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
        "initial_x_mm": case.initial_conditions.pos[0],
        "initial_y_mm": case.initial_conditions.pos[1],
        "initial_z_mm": case.initial_conditions.pos[2],
        "initial_vx_mm_s": case.initial_conditions.vel[0],
        "initial_vy_mm_s": case.initial_conditions.vel[1],
        "initial_vz_mm_s": case.initial_conditions.vel[2],
        "target_x_mm": case.target[0],
        "target_y_mm": case.target[1],
        "bounce_x_mm": bounce_x,
        "bounce_y_mm": bounce_y,
        "target_error_mm": target_error,
        "inside_margin": int(target_error <= TARGET_MARGIN_MM),
        "final_z_mm": float(result.x[2, -1]),
        "video": str(video_path) if video_path is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct-condition table-tennis serve simulations.")
    parser.add_argument("--repeat", type=int, default=20, help="Runs per serve variant.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation step in seconds.")
    parser.add_argument("--t-max", type=float, default=3.0, help="Simulation horizon in seconds.")
    parser.add_argument("--video-dir", type=Path, help="Directory where one MP4 per serve variant is saved.")
    parser.add_argument("--ffmpeg", help="Path to the FFmpeg executable used for MP4 output.")
    args = parser.parse_args()

    ffmpeg_path = None
    if args.video_dir is not None:
        ffmpeg_path = resolve_ffmpeg_path(args.ffmpeg)
        if ffmpeg_path is None:
            parser.error(
                "MP4 output requires FFmpeg. Install FFmpeg and add it to PATH, "
                "or pass its executable path with --ffmpeg."
            )

    print(
        "service,depth,lane,repeat,total_ms,avg_ms,samples,bounces,"
        "initial_x_mm,initial_y_mm,initial_z_mm,initial_vx_mm_s,initial_vy_mm_s,initial_vz_mm_s,"
        "target_x_mm,target_y_mm,bounce_x_mm,bounce_y_mm,target_error_mm,inside_margin,final_z_mm,video"
    )
    for case in build_cases():
        row = run_case(
            case,
            repeat=args.repeat,
            dt=args.dt,
            t_max=args.t_max,
            video_dir=args.video_dir,
            ffmpeg_path=ffmpeg_path,
        )
        print(
            f"{row['service']},{row['depth']},{row['lane']},{row['repeat']},"
            f"{row['total_ms']:.3f},{row['avg_ms']:.3f},{row['samples']},{row['bounces']},"
            f"{row['initial_x_mm']:.1f},{row['initial_y_mm']:.1f},{row['initial_z_mm']:.1f},"
            f"{row['initial_vx_mm_s']:.1f},{row['initial_vy_mm_s']:.1f},{row['initial_vz_mm_s']:.1f},"
            f"{row['target_x_mm']:.1f},{row['target_y_mm']:.1f},{row['bounce_x_mm']:.1f},{row['bounce_y_mm']:.1f},"
            f"{row['target_error_mm']:.1f},{row['inside_margin']},{row['final_z_mm']:.1f},{row['video']}"
        )


if __name__ == "__main__":
    main()
