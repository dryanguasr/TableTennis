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

from ..constants import BALL_RADIUS, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH
from ..models import InitialConditions
from ..physics import simulate
from ..visualization import animate_simulation, resolve_ffmpeg_path

RPS = 2 * np.pi

# Direct-service angular velocities are expressed as revolutions per second
# times RPS.  The benchmark intentionally uses 50-100 rps of combined spin so
# Magnus lift/sideswerve are visible instead of near-dead-ball trajectories.
SERVICE_TYPES = {
    "pendulum": {
        "position": (-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -45.0 * RPS, -35.0 * RPS),
        "velocity_adjust": (80.0, 0.0, 0.0),
    },
    "reverse_pendulum": {
        "position": (-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -45.0 * RPS, 35.0 * RPS),
        "velocity_adjust": (-40.0, 0.0, 0.0),
    },
    "hook": {
        "position": (-300.0, TABLE_WIDTH * 0.14, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -50.0 * RPS, -45.0 * RPS),
        "velocity_adjust": (200.0, 0.0, 0.0),
    },
    "tomahawk": {
        "position": (-300.0, TABLE_WIDTH * 0.74, TABLE_HEIGHT + 260.0),
        "omega": (50.0 * RPS, -30.0 * RPS, 0.0),
        "velocity_adjust": (60.0, 0.0, 0.0),
    },
    "reverse_tomahawk": {
        "position": (-300.0, TABLE_WIDTH * 0.68, TABLE_HEIGHT + 260.0),
        "omega": (-50.0 * RPS, -30.0 * RPS, 0.0),
        "velocity_adjust": (40.0, 0.0, 0.0),
    },
    "backhand_standard": {
        "position": (-300.0, TABLE_WIDTH * 0.50, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -40.0 * RPS, 35.0 * RPS),
        "depth_velocity": {
            "short": (7200.0, 0.0, -2300.0),
            "two_bounce": (8000.0, 0.0, -2300.0),
            "long": (9000.0, 0.0, -2500.0),
        },
    },
}
VELOCITY_OVERRIDES = {
    ("pendulum", "short", "forehand"): (5910.0, 2776.4, -3069.6),
    ("pendulum", "short", "elbow"): (6070.6, 1938.1, -3064.0),
    ("pendulum", "short", "backhand"): (6156.4, 1110.3, -3200.3),
    ("pendulum", "two_bounce", "forehand"): (6824.4, 2961.5, -3284.0),
    ("pendulum", "two_bounce", "elbow"): (6794.0, 2093.7, -3763.4),
    ("pendulum", "two_bounce", "backhand"): (7620.6, 1230.0, -1968.2),
    ("pendulum", "long", "forehand"): (7718.4, 3177.3, -3669.6),
    ("pendulum", "long", "elbow"): (8196.7, 2391.6, -3343.2),
    ("pendulum", "long", "backhand"): (8131.4, 1585.4, -3759.8),
    ("reverse_pendulum", "short", "forehand"): (6658.4, -125.9, -2269.6),
    ("reverse_pendulum", "short", "elbow"): (6224.8, -1088.2, -3174.3),
    ("reverse_pendulum", "short", "backhand"): (6177.6, -1962.2, -2692.7),
    ("reverse_pendulum", "two_bounce", "forehand"): (7791.8, -317.9, -2368.1),
    ("reverse_pendulum", "two_bounce", "elbow"): (7241.8, -1335.0, -3297.4),
    ("reverse_pendulum", "two_bounce", "backhand"): (6791.5, -2106.0, -3820.9),
    ("reverse_pendulum", "long", "forehand"): (8832.0, -650.0, -2935.7),
    ("reverse_pendulum", "long", "elbow"): (8693.7, -1538.0, -2865.6),
    ("reverse_pendulum", "long", "backhand"): (8772.2, -2415.3, -2384.6),
    ("hook", "short", "forehand"): (5782.7, 3169.6, -3610.0),
    ("hook", "short", "elbow"): (6012.4, 2385.3, -3371.0),
    ("hook", "short", "backhand"): (6384.9, 1549.8, -3147.7),
    ("hook", "two_bounce", "forehand"): (6992.5, 3583.8, -2810.6),
    ("hook", "two_bounce", "elbow"): (7152.4, 2694.2, -2844.1),
    ("hook", "two_bounce", "backhand"): (7667.4, 1790.7, -2383.0),
    ("hook", "long", "forehand"): (8587.4, 3955.7, -2435.5),
    ("hook", "long", "elbow"): (8105.0, 2995.9, -3489.8),
    ("hook", "long", "backhand"): (8464.0, 2207.3, -3369.2),
    ("tomahawk", "short", "forehand"): (6178.0, 1360.0, -2632.6),
    ("tomahawk", "short", "elbow"): (5954.5, 672.2, -3287.2),
    ("tomahawk", "short", "backhand"): (6323.8, -563.1, -2512.5),
    ("tomahawk", "two_bounce", "forehand"): (6999.9, 1549.3, -3318.9),
    ("tomahawk", "two_bounce", "elbow"): (7519.8, 158.1, -2005.5),
    ("tomahawk", "two_bounce", "backhand"): (7178.8, -263.9, -3257.3),
    ("tomahawk", "long", "forehand"): (8477.9, 1392.2, -2957.2),
    ("tomahawk", "long", "elbow"): (8772.8, 406.8, -2782.8),
    ("tomahawk", "long", "backhand"): (8984.8, -639.5, -2597.3),
    ("reverse_tomahawk", "short", "forehand"): (6022.0, -888.4, -3105.9),
    ("reverse_tomahawk", "short", "elbow"): (5915.2, -1857.6, -3351.8),
    ("reverse_tomahawk", "short", "backhand"): (6111.1, -2588.9, -2697.0),
    ("reverse_tomahawk", "two_bounce", "forehand"): (7047.2, -939.1, -3288.8),
    ("reverse_tomahawk", "two_bounce", "elbow"): (7110.3, -1712.5, -2928.1),
    ("reverse_tomahawk", "two_bounce", "backhand"): (7390.3, -2461.8, -2354.5),
    ("reverse_tomahawk", "long", "forehand"): (8396.5, -872.5, -3206.8),
    ("reverse_tomahawk", "long", "elbow"): (8436.7, -1725.8, -3054.5),
    ("reverse_tomahawk", "long", "backhand"): (8807.2, -2482.6, -2483.2),
    ("backhand_standard", "short", "forehand"): (6452.6, -162.4, -2597.2),
    ("backhand_standard", "short", "elbow"): (6107.9, -1100.0, -3054.8),
    ("backhand_standard", "short", "backhand"): (6116.6, -1951.5, -2507.0),
    ("backhand_standard", "two_bounce", "forehand"): (7624.7, -383.8, -2601.4),
    ("backhand_standard", "two_bounce", "elbow"): (7520.3, -1238.7, -2233.8),
    ("backhand_standard", "two_bounce", "backhand"): (7206.8, -2158.6, -2497.1),
    ("backhand_standard", "long", "forehand"): (8902.7, -634.4, -2736.4),
    ("backhand_standard", "long", "elbow"): (8141.0, -1567.9, -3489.2),
    ("backhand_standard", "long", "backhand"): (8160.2, -2400.1, -3151.1),
}
DEPTHS = {
    "short": {"vel": (7200.0, 0.0, -2300.0), "target_x": TABLE_LENGTH / 2 + 220.0},
    "two_bounce": {"vel": (8000.0, 0.0, -2300.0), "target_x": TABLE_LENGTH / 2 + 520.0},
    "long": {"vel": (9000.0, 0.0, -2500.0), "target_x": TABLE_LENGTH - 360.0},
}
LANES = {
    "forehand": {"y": TABLE_WIDTH * 0.72},
    "elbow": {"y": TABLE_WIDTH * 0.50},
    "backhand": {"y": TABLE_WIDTH * 0.28},
}
LANE_VY = {
    "pendulum": {
        "short": {"forehand": 4000.0, "elbow": 2700.0, "backhand": 1700.0},
        "two_bounce": {"forehand": 4000.0, "elbow": 2800.0, "backhand": 1800.0},
        "long": {"forehand": 4000.0, "elbow": 3000.0, "backhand": 2000.0},
    },
    "reverse_pendulum": {
        "short": {"forehand": 1030.0, "elbow": 30.0, "backhand": -975.0},
        "two_bounce": {"forehand": 1050.0, "elbow": 40.0, "backhand": -970.0},
        "long": {"forehand": 1070.0, "elbow": 50.0, "backhand": -970.0},
    },
    "hook": {
        "short": {"forehand": 4100.0, "elbow": 3000.0, "backhand": 1900.0},
        "two_bounce": {"forehand": 4200.0, "elbow": 3100.0, "backhand": 2300.0},
        "long": {"forehand": 4200.0, "elbow": 3200.0, "backhand": 2500.0},
    },
    "tomahawk": {
        "short": {"forehand": 1500.0, "elbow": 400.0, "backhand": -700.0},
        "two_bounce": {"forehand": 1500.0, "elbow": 400.0, "backhand": -700.0},
        "long": {"forehand": 1500.0, "elbow": 400.0, "backhand": -700.0},
    },
    "reverse_tomahawk": {
        "short": {"forehand": -1250.0, "elbow": -2250.0, "backhand": -3200.0},
        "two_bounce": {"forehand": -1250.0, "elbow": -2250.0, "backhand": -3250.0},
        "long": {"forehand": -1250.0, "elbow": -2250.0, "backhand": -3250.0},
    },
    "backhand_standard": {
        "short": {"forehand": -450.0, "elbow": -1500.0, "backhand": -2500.0},
        "two_bounce": {"forehand": -450.0, "elbow": -1500.0, "backhand": -2500.0},
        "long": {"forehand": -500.0, "elbow": -1600.0, "backhand": -2500.0},
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
    contacts = np.isclose(result.x[2], table_level, atol=1e-6) & (result.v[2] > 0)
    starts = contacts & np.concatenate(([True], ~contacts[:-1]))
    return int(np.count_nonzero(starts))


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct-condition table-tennis serve simulations.")
    parser.add_argument("--repeat", type=int, default=20, help="Runs per serve variant.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation step in seconds.")
    parser.add_argument("--t-max", type=float, default=3.0, help="Simulation horizon in seconds.")
    parser.add_argument("--video-dir", type=Path, help="Directory where one MP4 per serve variant is saved.")
    parser.add_argument("--ffmpeg", help="Path to the FFmpeg executable used for MP4 output.")
    args = parser.parse_args(argv)

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
