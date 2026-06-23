"""Benchmark common table-tennis serves from direct ball initial conditions.

The script runs the existing physics engine by giving the ball an initial
position, linear velocity and angular velocity directly. It covers typical serve
families in 9 variants each: short, two-bounce ("enfermería") and long, aimed to
forehand, elbow and backhand lanes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from table_tennis_simulation import BALL_RADIUS, InitialConditions, TABLE_HEIGHT, TABLE_WIDTH, simulate

SERVICE_TYPES = {
    "pendulum": {"vel": (4700.0, -650.0, -1350.0), "omega": (80.0, -210.0, 520.0)},
    "reverse_pendulum": {"vel": (4700.0, 650.0, -1350.0), "omega": (-80.0, 210.0, -520.0)},
    "hook": {"vel": (4300.0, -1200.0, -1200.0), "omega": (40.0, -320.0, 680.0)},
    "tomahawk": {"vel": (4550.0, 900.0, -1300.0), "omega": (520.0, 120.0, -240.0)},
    "reverse_tomahawk": {"vel": (4550.0, -900.0, -1300.0), "omega": (-520.0, -120.0, 240.0)},
}
DEPTHS = {
    "short": {"speed_scale": 0.78, "z_boost": 150.0},
    "two_bounce": {"speed_scale": 0.90, "z_boost": 60.0},
    "long": {"speed_scale": 1.12, "z_boost": -60.0},
}
LANES = {
    "forehand": {"y": TABLE_WIDTH * 0.72, "vy_bias": 350.0},
    "elbow": {"y": TABLE_WIDTH * 0.50, "vy_bias": 0.0},
    "backhand": {"y": TABLE_WIDTH * 0.28, "vy_bias": -350.0},
}


@dataclass(frozen=True)
class BenchmarkCase:
    service: str
    depth: str
    lane: str
    initial_conditions: InitialConditions


def build_cases() -> Iterable[BenchmarkCase]:
    for service_name, service in SERVICE_TYPES.items():
        for depth_name, depth in DEPTHS.items():
            for lane_name, lane in LANES.items():
                velocity = np.array(service["vel"], dtype=float)
                velocity[:2] *= depth["speed_scale"]
                velocity[1] += lane["vy_bias"]
                velocity[2] += depth["z_boost"]
                position = (120.0, lane["y"], TABLE_HEIGHT + 310.0)
                yield BenchmarkCase(
                    service_name,
                    depth_name,
                    lane_name,
                    InitialConditions(pos=position, vel=tuple(velocity), omega=service["omega"]),
                )


def count_table_bounces(result) -> int:
    table_level = TABLE_HEIGHT + BALL_RADIUS
    return int(np.count_nonzero(np.isclose(result.x[2], table_level, atol=1e-6) & (result.v[2] > 0)))


def run_case(case: BenchmarkCase, repeat: int, dt: float, t_max: float) -> dict[str, float | int | str]:
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = simulate(case.initial_conditions, dt=dt, t_max=t_max)
    elapsed = time.perf_counter() - start
    assert result is not None
    return {
        "service": case.service,
        "depth": case.depth,
        "lane": case.lane,
        "repeat": repeat,
        "total_ms": elapsed * 1000.0,
        "avg_ms": elapsed * 1000.0 / repeat,
        "samples": result.x.shape[1],
        "bounces": count_table_bounces(result),
        "final_z_mm": float(result.x[2, -1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct-condition table-tennis serve simulations.")
    parser.add_argument("--repeat", type=int, default=20, help="Runs per serve variant.")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation step in seconds.")
    parser.add_argument("--t-max", type=float, default=3.0, help="Simulation horizon in seconds.")
    args = parser.parse_args()

    print("service,depth,lane,repeat,total_ms,avg_ms,samples,bounces,final_z_mm")
    for case in build_cases():
        row = run_case(case, repeat=args.repeat, dt=args.dt, t_max=args.t_max)
        print(
            f"{row['service']},{row['depth']},{row['lane']},{row['repeat']},"
            f"{row['total_ms']:.3f},{row['avg_ms']:.3f},{row['samples']},{row['bounces']},{row['final_z_mm']:.1f}"
        )


if __name__ == "__main__":
    main()
