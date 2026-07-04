"""Reusable legality and target validators for services and returns."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .constants import (
    BALL_RADIUS,
    G,
    NET_HEIGHT,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from .events import net_crossings, table_bounces
from .exchange import ServiceTargets, StrokeTargets, ValidationReport
from .models import RacketImpactParameters, SimulationResult
from .physics import apply_racket_impact


RPS = 2 * np.pi
NET_CLEARANCE_LEVEL = TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS


def _first_bounce_xy(
    result: SimulationResult,
) -> tuple[float, float] | None:
    bounces = table_bounces(result)
    if not bounces:
        return None
    return float(bounces[0].point[0]), float(bounces[0].point[1])


def net_clearance(result: SimulationResult) -> float:
    """Return clearance between the first net crossing and the legal net level."""

    crossings = net_crossings(result)
    if not crossings:
        return float("-inf")
    return float(crossings[0].point[2] - NET_CLEARANCE_LEVEL)


def spin_error(
    params: RacketImpactParameters,
    target_spin_rps: Iterable[float],
) -> float:
    """Return Euclidean post-impact spin error in revolutions per second."""

    actual = np.asarray(apply_racket_impact(params).omega, dtype=float) / RPS
    target = np.asarray(tuple(target_spin_rps), dtype=float)
    return float(np.linalg.norm(actual - target))


def _depth_bounce_violation(depth: str, count: int) -> str | None:
    if depth in {"short", "two_bounce"} and count < 2:
        return "the shot does not bounce twice on the target half"
    if depth == "long" and count != 1:
        return "the long shot does not leave the table after its first bounce"
    return None


def validate_service(
    params: RacketImpactParameters,
    result: SimulationResult,
    targets: ServiceTargets = ServiceTargets(),
) -> ValidationReport:
    """Validate serve legality, target depth, direction, clearance, and spin."""

    violations = []
    bounces = table_bounces(result)
    receiver_bounces = [
        event
        for event in table_bounces(result, "receiver")
        if event.point[0] < TABLE_LENGTH - BALL_RADIUS
    ]
    first_bounce = _first_bounce_xy(result)
    target = targets.target_point
    bounce_error = (
        float(np.linalg.norm(np.asarray(first_bounce) - np.asarray(target)))
        if first_bounce is not None
        and bounces
        and bounces[0].side == "receiver"
        else float("inf")
    )
    if (
        len(bounces) < 2
        or bounces[0].side != "server"
        or bounces[1].side != "receiver"
    ):
        violations.append("service bounces are not server then receiver")
    else:
        receiver_point = np.asarray(bounces[1].point[:2], dtype=float)
        bounce_error = float(
            np.linalg.norm(receiver_point - np.asarray(target))
        )
        first_bounce = tuple(float(value) for value in receiver_point)
    if bounce_error > targets.bounce_tolerance_mm:
        violations.append("service bounce is outside tolerance")
    if params.ball_position[0] >= 0 or params.ball_position[2] <= TABLE_HEIGHT:
        violations.append(
            "service contact is not behind the baseline and above the table"
        )
    if (
        abs(params.ball_velocity[0]) > 1e-6
        or abs(params.ball_velocity[1]) > 1e-6
    ):
        violations.append("service toss is not vertical")
    minimum_toss_speed = np.sqrt(2 * G * 160.0)
    maximum_toss_speed = np.sqrt(2 * G * 2000.0)
    if not minimum_toss_speed <= abs(params.ball_velocity[2]) <= maximum_toss_speed:
        violations.append("service toss is outside the 16 cm to 2 m range")
    if np.linalg.norm(params.ball_omega) > 1e-6:
        violations.append("service toss imparts spin before racket contact")
    service_cutoff = bounces[1].time if len(bounces) >= 2 else float("inf")
    if any(
        event.kind == "net_contact" and event.time <= service_cutoff
        for event in result.events
    ):
        violations.append("service touches the net")
    clearance = net_clearance(result)
    if clearance < targets.min_net_clearance_mm:
        violations.append("service net clearance is too small")
    measured_spin_error = spin_error(params, targets.spin_rps)
    if measured_spin_error > targets.spin_tolerance_rps:
        violations.append("service spin is outside tolerance")
    depth_violation = _depth_bounce_violation(
        targets.depth,
        len(receiver_bounces),
    )
    if depth_violation:
        violations.append(depth_violation)
    return ValidationReport(
        passed=not violations,
        violations=tuple(violations),
        target_point=target,
        first_bounce=first_bounce,
        bounce_error_mm=bounce_error,
        spin_error_rps=measured_spin_error,
        net_clearance_mm=clearance,
        bounces_on_target_side=len(receiver_bounces),
    )


def validate_return(
    params: RacketImpactParameters,
    result: SimulationResult,
    targets: StrokeTargets,
) -> ValidationReport:
    """Validate a return's legality, physical gesture, target, and spin."""

    violations = []
    bounces = table_bounces(result)
    server_bounces = [
        event
        for event in table_bounces(result, "server")
        if event.point[0] > BALL_RADIUS
    ]
    if targets.depth == "long" and server_bounces:
        first_index = server_bounces[0].index
        later = np.arange(first_index + 1, result.x.shape[1])
        exits = later[
            (result.x[0, later] <= BALL_RADIUS)
            | (result.x[1, later] <= 0.0)
            | (result.x[1, later] >= TABLE_WIDTH)
            | (result.x[2, later] < 0.0)
        ]
        if exits.size:
            server_bounces = [
                event
                for event in server_bounces
                if event.index < int(exits[0])
            ]

    first_bounce = _first_bounce_xy(result)
    target = targets.target_point
    bounce_error = (
        float(np.linalg.norm(np.asarray(first_bounce) - np.asarray(target)))
        if first_bounce is not None
        else float("inf")
    )
    if not bounces or bounces[0].side != "server":
        violations.append("return does not first bounce on the server half")
    if bounce_error > targets.bounce_tolerance_mm:
        violations.append("return bounce is outside tolerance")
    if apply_racket_impact(params).vel[0] >= 0:
        violations.append("return does not travel toward the server")
    return_cutoff = bounces[0].time if bounces else float("inf")
    if any(
        event.kind == "net_contact" and event.time <= return_cutoff
        for event in result.events
    ):
        violations.append("return touches the net")

    clearance = net_clearance(result)
    if clearance < targets.min_net_clearance_mm:
        violations.append("return net clearance is too small")
    cut_stroke = targets.spin_rps[1] > 0
    maximum_clearance = (
        targets.max_net_clearance_mm
        if targets.max_net_clearance_mm is not None
        else (180.0 if cut_stroke else 350.0)
    )
    if clearance > maximum_clearance:
        violations.append("return net clearance is unrealistically high")

    first_bounce_index = (
        bounces[0].index if bounces else result.x.shape[1] - 1
    )
    maximum_height = float(
        np.max(result.x[2, : first_bounce_index + 1])
        - (TABLE_HEIGHT + BALL_RADIUS)
    )
    allowed_height = (
        targets.max_height_above_table_mm
        if targets.max_height_above_table_mm is not None
        else (420.0 if cut_stroke else 650.0)
    )
    if maximum_height > allowed_height:
        violations.append("return arc is unrealistically high")

    post_impact = apply_racket_impact(params)
    if cut_stroke:
        if (
            params.racket_velocity[0] >= 0
            or params.racket_velocity[2] >= -100.0
        ):
            violations.append(
                "cut stroke racket must move forward and downward"
            )
        if (
            params.ball_position[2] < NET_CLEARANCE_LEVEL
            and post_impact.vel[2] <= 0
        ):
            violations.append(
                "cut return cannot rise from contact to net height"
            )
        if post_impact.vel[2] > 1500.0:
            violations.append(
                "cut return launches the ball too steeply upward"
            )
    elif params.racket_velocity[2] <= 100.0:
        violations.append("topspin stroke racket must move upward")

    if bounces and result.v[0, bounces[0].index] >= 0:
        violations.append("return reverses direction after its first bounce")
    if (
        len(server_bounces) >= 2
        and server_bounces[1].point[0] >= server_bounces[0].point[0]
    ):
        violations.append("return bounces back toward the net")

    measured_spin_error = spin_error(params, targets.spin_rps)
    if measured_spin_error > targets.spin_tolerance_rps:
        violations.append("return spin is outside tolerance")
    depth_violation = _depth_bounce_violation(
        targets.depth,
        len(server_bounces),
    )
    if depth_violation:
        violations.append(depth_violation)
    reported_bounces = (
        min(len(server_bounces), 2)
        if targets.depth in {"short", "two_bounce"}
        else len(server_bounces)
    )
    return ValidationReport(
        passed=not violations,
        violations=tuple(violations),
        target_point=target,
        first_bounce=first_bounce,
        bounce_error_mm=bounce_error,
        spin_error_rps=measured_spin_error,
        net_clearance_mm=clearance,
        bounces_on_target_side=reported_bounces,
    )
