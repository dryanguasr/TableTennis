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

from ..constants import BALL_RADIUS, NET_HEIGHT, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH
from ..events import table_bounces
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
    },
    "reverse_pendulum": {
        "position": (-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -45.0 * RPS, 35.0 * RPS),
    },
    "hook": {
        "position": (-300.0, TABLE_WIDTH * 0.14, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -50.0 * RPS, -45.0 * RPS),
    },
    "tomahawk": {
        "position": (-300.0, TABLE_WIDTH * 0.74, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -30.0 * RPS, -45.0 * RPS),
    },
    "reverse_tomahawk": {
        "position": (-300.0, TABLE_WIDTH * 0.68, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -30.0 * RPS, 45.0 * RPS),
    },
    "backhand_standard": {
        "position": (-300.0, TABLE_WIDTH * 0.50, TABLE_HEIGHT + 260.0),
        "omega": (0.0, -40.0 * RPS, 35.0 * RPS),
    },
}
VELOCITY_OVERRIDES = {
    ("backhand_standard", "long", "backhand"): (6610.7728, -1271.0604, -871.5637),
    ("backhand_standard", "long", "elbow"): (6672.5440, -472.4652, -902.4925),
    ("backhand_standard", "long", "forehand"): (6817.9137, 315.5044, -1190.9629),
    ("backhand_standard", "short", "backhand"): (4974.5924, -1043.6443, -1087.3914),
    ("backhand_standard", "short", "elbow"): (5010.0254, -270.6272, -1082.8217),
    ("backhand_standard", "short", "forehand"): (5063.2715, 495.3634, -1089.7890),
    ("backhand_standard", "two_bounce", "backhand"): (5424.1351, -1102.1550, -1045.4896),
    ("backhand_standard", "two_bounce", "elbow"): (5372.1247, -312.5430, -765.6440),
    ("backhand_standard", "two_bounce", "forehand"): (5457.9430, 453.1984, -840.3857),
    ("hook", "long", "backhand"): (6710.2535, 987.6781, -1288.1652),
    ("hook", "long", "elbow"): (6644.8391, 1791.6037, -1084.9203),
    ("hook", "long", "forehand"): (6616.2006, 2619.9992, -1063.8062),
    ("hook", "short", "backhand"): (4977.9115, 752.8503, -1063.0504),
    ("hook", "short", "elbow"): (4957.2153, 1528.7113, -1091.8937),
    ("hook", "short", "forehand"): (4962.8816, 2334.1175, -1081.1327),
    ("hook", "two_bounce", "backhand"): (4975.7870, 740.6161, -160.4612),
    ("hook", "two_bounce", "elbow"): (5211.1271, 1542.5715, -555.5472),
    ("hook", "two_bounce", "forehand"): (5196.8187, 2327.3549, -564.3089),
    ("pendulum", "long", "backhand"): (6697.2422, 883.5467, -1264.6489),
    ("pendulum", "long", "elbow"): (6653.6825, 1689.1624, -1190.5581),
    ("pendulum", "long", "forehand"): (6602.6047, 2508.9579, -1032.7097),
    ("pendulum", "short", "backhand"): (4975.2250, 667.4475, -1070.2852),
    ("pendulum", "short", "elbow"): (4956.6510, 1443.6388, -1096.9834),
    ("pendulum", "short", "forehand"): (4959.7400, 2244.3519, -1106.5025),
    ("pendulum", "two_bounce", "backhand"): (5376.2418, 712.0833, -870.7831),
    ("pendulum", "two_bounce", "elbow"): (5349.0757, 1490.8859, -873.0242),
    ("pendulum", "two_bounce", "forehand"): (5374.6603, 2304.3965, -1014.7343),
    ("reverse_pendulum", "long", "backhand"): (6745.2198, -13.4803, -1151.9396),
    ("reverse_pendulum", "long", "elbow"): (6775.7256, 781.1098, -961.2155),
    ("reverse_pendulum", "long", "forehand"): (6914.4493, 1579.3374, -1094.8977),
    ("reverse_pendulum", "short", "backhand"): (5021.9161, 164.0648, -1080.7397),
    ("reverse_pendulum", "short", "elbow"): (5075.0061, 926.1168, -1053.9065),
    ("reverse_pendulum", "short", "forehand"): (5153.3839, 1655.7024, -1043.5146),
    ("reverse_pendulum", "two_bounce", "backhand"): (5456.8361, 122.9254, -974.3821),
    ("reverse_pendulum", "two_bounce", "elbow"): (5334.2036, 871.5216, -544.0097),
    ("reverse_pendulum", "two_bounce", "forehand"): (5095.1666, 1529.4306, -101.1254),
    ("reverse_tomahawk", "long", "backhand"): (6681.6975, -2101.7983, -892.0778),
    ("reverse_tomahawk", "long", "elbow"): (6780.9489, -1276.1984, -1031.2518),
    ("reverse_tomahawk", "long", "forehand"): (6847.5746, -452.8341, -1026.1395),
    ("reverse_tomahawk", "short", "backhand"): (4996.0625, -1790.9089, -1067.0533),
    ("reverse_tomahawk", "short", "elbow"): (5016.4692, -981.8434, -1053.1970),
    ("reverse_tomahawk", "short", "forehand"): (5062.1179, -194.0303, -1036.3235),
    ("reverse_tomahawk", "two_bounce", "backhand"): (5429.5750, -1863.0979, -975.5983),
    ("reverse_tomahawk", "two_bounce", "elbow"): (5265.0504, -1010.6398, -513.6243),
    ("reverse_tomahawk", "two_bounce", "forehand"): (5561.3951, -259.3291, -1127.9040),
    ("tomahawk", "long", "backhand"): (7065.2763, -1103.7348, -1106.6202),
    ("tomahawk", "long", "elbow"): (6891.3996, -288.9333, -920.5987),
    ("tomahawk", "long", "forehand"): (6872.1339, 530.4556, -1158.3934),
    ("tomahawk", "short", "backhand"): (5207.3257, -1298.2142, -1032.6381),
    ("tomahawk", "short", "elbow"): (5114.8002, -515.3324, -1003.2918),
    ("tomahawk", "short", "forehand"): (5054.7804, 264.9278, -1027.5737),
    ("tomahawk", "two_bounce", "backhand"): (5695.8195, -1251.1710, -1019.6440),
    ("tomahawk", "two_bounce", "elbow"): (5423.9697, -453.5271, -590.1045),
    ("tomahawk", "two_bounce", "forehand"): (5489.8347, 324.7757, -886.9578),
}
DEPTHS = {
    # With the ACE flight/table model, 330 mm beyond the net remains a
    # genuinely short service while retaining the required 5 mm net margin.
    "short": {"target_x": TABLE_LENGTH / 2 + 330.0},
    "two_bounce": {"target_x": TABLE_LENGTH / 2 + 520.0},
    "long": {"target_x": TABLE_LENGTH - 360.0},
}
LANES = {
    "forehand": {"y": TABLE_WIDTH * 0.72},
    "elbow": {"y": TABLE_WIDTH * 0.50},
    "backhand": {"y": TABLE_WIDTH * 0.28},
}
TARGET_MARGIN_MM = 50.0
NET_CLEARANCE_LEVEL = TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
MAX_FLIGHT_HEIGHT_ABOVE_NET_MM = 50.0
MAX_REBOUND_HEIGHT_ABOVE_NET_MM = 25.0


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
                key = (service_name, depth_name, lane_name)
                if key not in VELOCITY_OVERRIDES:
                    raise KeyError(
                        f"Missing calibrated direct-service velocity for {key}."
                    )
                velocity = VELOCITY_OVERRIDES[key]
                position = service["position"]
                yield BenchmarkCase(
                    service_name,
                    depth_name,
                    lane_name,
                    InitialConditions(pos=position, vel=tuple(velocity), omega=service["omega"]),
                    (depth["target_x"], lane["y"]),
                )


def count_table_bounces(result) -> int:
    return len(table_bounces(result))


def first_opponent_bounce(result) -> tuple[float, float] | None:
    hits = table_bounces(result, "receiver")
    if not hits:
        return None
    return float(hits[0].point[0]), float(hits[0].point[1])


def low_arc_metrics(result) -> tuple[float, float]:
    """Return serve flight/rebound maxima relative to the legal net height."""

    bounces = table_bounces(result)
    server = [event for event in bounces if event.side == "server"]
    receiver = [event for event in bounces if event.side == "receiver"]
    if not server or not receiver:
        return float("inf"), float("inf")

    server_index = server[0].index
    receiver_index = receiver[0].index
    flight_height = float(
        np.max(result.x[2, server_index : receiver_index + 1])
        - NET_CLEARANCE_LEVEL
    )
    later_bounces = [
        event.index for event in bounces if event.index > receiver_index
    ]
    rebound_end = later_bounces[0] if later_bounces else result.x.shape[1] - 1
    rebound_height = float(
        np.max(result.x[2, receiver_index : rebound_end + 1])
        - NET_CLEARANCE_LEVEL
    )
    return flight_height, rebound_height


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
    max_height_above_net, rebound_height_above_net = low_arc_metrics(result)

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
        "max_height_above_net_mm": max_height_above_net,
        "rebound_height_above_net_mm": rebound_height_above_net,
        "inside_low_arc": int(
            max_height_above_net <= MAX_FLIGHT_HEIGHT_ABOVE_NET_MM
            and rebound_height_above_net <= MAX_REBOUND_HEIGHT_ABOVE_NET_MM
        ),
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
        "target_x_mm,target_y_mm,bounce_x_mm,bounce_y_mm,target_error_mm,inside_margin,"
        "max_height_above_net_mm,rebound_height_above_net_mm,inside_low_arc,final_z_mm,video"
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
            f"{row['target_error_mm']:.1f},{row['inside_margin']},"
            f"{row['max_height_above_net_mm']:.1f},{row['rebound_height_above_net_mm']:.1f},"
            f"{row['inside_low_arc']},{row['final_z_mm']:.1f},{row['video']}"
        )


if __name__ == "__main__":
    main()
