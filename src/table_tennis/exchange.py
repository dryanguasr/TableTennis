"""Public models and orchestration for a complete serve-return exchange."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .constants import TABLE_LENGTH, TABLE_WIDTH
from .events import identify_trajectory_moments
from .models import InitialConditions, RacketImpactParameters, SimulationResult
from .physics import simulate_racket_impact


SERVICE_DEPTH_X = {
    "short": TABLE_LENGTH / 2 + 330.0,
    "two_bounce": TABLE_LENGTH / 2 + 520.0,
    "long": TABLE_LENGTH - 360.0,
}
RETURN_DEPTH_X = {
    "short": TABLE_LENGTH / 2 - 270.0,
    "two_bounce": TABLE_LENGTH / 2 - 320.0,
    "long": 360.0,
}
SERVICE_DIRECTION_Y = {
    "forehand": TABLE_WIDTH * 0.72,
    "elbow": TABLE_WIDTH * 0.50,
    "backhand": TABLE_WIDTH * 0.28,
}
RETURN_DIRECTION_Y = {
    "forehand": TABLE_WIDTH * 0.28,
    "elbow": TABLE_WIDTH * 0.50,
    "backhand": TABLE_WIDTH * 0.72,
}


@dataclass(frozen=True)
class ContactSelection:
    moment: Literal[2, 3, 4] = 4
    fraction: float = 0.5


@dataclass(frozen=True)
class RubberProperties:
    friction: float = 1.2
    restitution: float = 0.8


@dataclass(frozen=True)
class StrokeTargets:
    depth: Literal["short", "two_bounce", "long"] = "two_bounce"
    direction: Literal["forehand", "elbow", "backhand"] = "elbow"
    spin_rps: tuple[float, float, float] = (0.0, 45.0, 0.0)
    stroke_side: Literal["forehand", "backhand"] = "backhand"
    bounce_tolerance_mm: float = 75.0
    spin_tolerance_rps: float = 10.0
    min_net_clearance_mm: float = 5.0
    max_net_clearance_mm: float | None = None
    max_height_above_table_mm: float | None = None
    target_x: float | None = None
    target_y: float | None = None

    @property
    def target_point(self) -> tuple[float, float]:
        return (
            RETURN_DEPTH_X[self.depth]
            if self.target_x is None
            else self.target_x,
            RETURN_DIRECTION_Y[self.direction]
            if self.target_y is None
            else self.target_y,
        )


@dataclass(frozen=True)
class ServiceTargets:
    depth: Literal["short", "two_bounce", "long"] = "short"
    direction: Literal["forehand", "elbow", "backhand"] = "elbow"
    spin_rps: tuple[float, float, float] = (0.0, -45.0, 35.0)
    bounce_tolerance_mm: float = 75.0
    spin_tolerance_rps: float = 10.0
    min_net_clearance_mm: float = 5.0
    max_height_above_net_mm: float | None = None
    max_rebound_height_above_net_mm: float | None = None
    target_x: float | None = None
    target_y: float | None = None

    @property
    def target_point(self) -> tuple[float, float]:
        return (
            SERVICE_DEPTH_X[self.depth]
            if self.target_x is None
            else self.target_x,
            SERVICE_DIRECTION_Y[self.direction]
            if self.target_y is None
            else self.target_y,
        )


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    violations: tuple[str, ...]
    target_point: tuple[float, float]
    first_bounce: tuple[float, float] | None
    bounce_error_mm: float
    spin_error_rps: float
    net_clearance_mm: float
    bounces_on_target_side: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ExchangeResult:
    service_params: RacketImpactParameters
    service_result: SimulationResult
    contact: ContactSelection
    contact_index: int
    return_params: RacketImpactParameters
    return_result: SimulationResult
    service_validation: ValidationReport
    return_validation: ValidationReport

    @property
    def passed(self) -> bool:
        return self.service_validation.passed and self.return_validation.passed


def contact_index(
    service_result: SimulationResult,
    selection: ContactSelection,
) -> int:
    """Resolve a phase selection to one trajectory sample."""

    moments = identify_trajectory_moments(service_result)
    if selection.moment not in moments:
        raise ValueError(
            f"Moment {selection.moment} is not available for this service."
        )
    moment = moments[selection.moment]
    if selection.moment == 3 or moment.interval is None:
        return moment.index
    fraction = float(np.clip(selection.fraction, 0.0, 1.0))
    target_time = moment.interval[0] + fraction * (
        moment.interval[1] - moment.interval[0]
    )
    if service_result.t is None:
        return moment.index
    return int(np.argmin(np.abs(service_result.t - target_time)))


def contact_state(
    service_result: SimulationResult,
    selection: ContactSelection,
) -> tuple[int, InitialConditions]:
    """Return the incoming state at the selected contact."""

    index = contact_index(service_result, selection)
    return index, InitialConditions(
        pos=tuple(float(value) for value in service_result.x[:, index]),
        vel=tuple(float(value) for value in service_result.v[:, index]),
        omega=tuple(float(value) for value in service_result.omega[:, index]),
    )


def simulate_exchange(
    service_params: RacketImpactParameters,
    return_params: RacketImpactParameters,
    contact: ContactSelection,
    service_targets: ServiceTargets,
    return_targets: StrokeTargets,
    dt: float = 0.005,
    t_max: float = 3.0,
) -> ExchangeResult:
    """Simulate and validate both strokes in an exchange."""

    from .validation import validate_return, validate_service

    service_result = simulate_racket_impact(service_params, dt=dt, t_max=t_max)
    index = contact_index(service_result, contact)
    expected_position = service_result.x[:, index]
    if not np.allclose(
        expected_position,
        return_params.ball_position,
        atol=1e-6,
    ):
        raise ValueError(
            "Return parameters do not start at the selected service contact."
        )
    return_result = simulate_racket_impact(return_params, dt=dt, t_max=t_max)
    return ExchangeResult(
        service_params=service_params,
        service_result=service_result,
        contact=contact,
        contact_index=index,
        return_params=return_params,
        return_result=return_result,
        service_validation=validate_service(
            service_params,
            service_result,
            service_targets,
        ),
        return_validation=validate_return(
            return_params,
            return_result,
            return_targets,
        ),
    )
