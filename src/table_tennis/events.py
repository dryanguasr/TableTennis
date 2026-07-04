"""Trajectory event queries and coaching-moment detection."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import BALL_RADIUS, DT, TABLE_HEIGHT, TABLE_LENGTH
from .models import SimulationResult, TrajectoryMoment


def table_bounces(
    result: SimulationResult,
    side: str | None = None,
):
    """Return table-bounce events, optionally filtered by table side."""

    events = [event for event in result.events if event.kind == "table_bounce"]
    return [event for event in events if side is None or event.side == side]


def net_crossings(result: SimulationResult):
    """Return all net-plane crossing events."""

    return [event for event in result.events if event.kind == "net_cross"]


def identify_trajectory_moments(
    result: SimulationResult,
    table_level: float = TABLE_HEIGHT + BALL_RADIUS,
) -> Dict[int, TrajectoryMoment]:
    """Identify coaching moments 1-6 on the first receiver-side arc."""

    t = result.t if result.t is not None else np.arange(result.x.shape[1]) * DT
    x, z, vz = result.x[0], result.x[2], result.v[2]
    moments: Dict[int, TrajectoryMoment] = {}

    bounce_candidates = np.asarray(
        [
            event.index
            for event in result.events
            if event.kind == "table_bounce" and event.side == "receiver"
        ],
        dtype=int,
    )
    if bounce_candidates.size == 0:
        bounce_candidates = np.where(
            (x >= TABLE_LENGTH / 2)
            & (np.isclose(z, table_level, atol=1e-6))
            & (vz > 0)
        )[0]
    if bounce_candidates.size == 0:
        bounce_candidates = np.where(
            (x >= TABLE_LENGTH / 2) & (z <= table_level) & (vz > 0)
        )[0]
    if bounce_candidates.size == 0:
        return moments

    i1 = int(bounce_candidates[0])
    moments[1] = TrajectoryMoment(
        "primer impacto en el lado contrario",
        i1,
        float(t[i1]),
        tuple(result.x[:, i1]),
    )

    later_bounces = bounce_candidates[bounce_candidates > i1]
    if later_bounces.size:
        i5 = int(later_bounces[0])
    else:
        later_indices = np.arange(i1 + 1, result.x.shape[1])
        arrivals = later_indices[z[later_indices] <= table_level]
        i5 = int(arrivals[0]) if arrivals.size else result.x.shape[1] - 1

    arc = np.arange(i1, i5 + 1)
    apex_i = int(arc[np.argmax(z[arc])])
    rising = np.arange(i1 + 1, apex_i)
    if rising.size:
        r0, r1 = int(rising[0]), int(rising[-1])
        mid_i = int(round((r0 + r1) / 2))
        moments[2] = TrajectoryMoment(
            "fase ascendente",
            mid_i,
            float(t[mid_i]),
            tuple(result.x[:, mid_i]),
            (float(t[r0]), float(t[r1])),
            tuple(result.x[:, mid_i]),
        )

    moments[3] = TrajectoryMoment(
        "punto más alto",
        apex_i,
        float(t[apex_i]),
        tuple(result.x[:, apex_i]),
    )

    descending = np.arange(apex_i + 1, i5)
    if descending.size:
        d0, d1 = int(descending[0]), int(descending[-1])
        mid_i = int(round((d0 + d1) / 2))
        moments[4] = TrajectoryMoment(
            "fase descendente sobre la mesa",
            mid_i,
            float(t[mid_i]),
            tuple(result.x[:, mid_i]),
            (float(t[d0]), float(t[d1])),
            tuple(result.x[:, mid_i]),
        )

    if i5 < result.x.shape[1] - 1:
        moments[5] = TrajectoryMoment(
            "segunda llegada al nivel de la mesa",
            i5,
            float(t[i5]),
            tuple(result.x[:, i5]),
        )
        post = np.arange(i5 + 1, result.x.shape[1])
        below = post[z[post] < table_level]
        if below.size:
            i6 = int(below[0])
            moments[6] = TrajectoryMoment(
                "bola por debajo del nivel de la mesa",
                i6,
                float(t[i6]),
                tuple(result.x[:, i6]),
            )

    return moments
