"""Shared data models for simulation, impact, and trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .constants import BALL_RADIUS, NET_HEIGHT, TABLE_HEIGHT, TABLE_WIDTH


@dataclass
class InitialConditions:
    pos: Tuple[float, float, float] = (
        0.0,
        TABLE_WIDTH * 4 / 8,
        TABLE_HEIGHT + 2 * NET_HEIGHT,
    )
    vel: Tuple[float, float, float] = (7000.0, -3000.0, -3000.0)
    omega: Tuple[float, float, float] = (0.0, 0.0, 75.0 * 2 * np.pi)


@dataclass
class SimulationResult:
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    t: Optional[np.ndarray] = None
    events: Tuple["SimulationEvent", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationEvent:
    """Discrete event emitted by the numerical simulation."""

    kind: str
    index: int
    time: float
    point: Tuple[float, float, float]
    side: Optional[str] = None


@dataclass
class RacketImpactParameters:
    """Ball and racket state at impact.

    Units are millimetres, seconds, radians, and degrees for racket angles.
    """

    ball_velocity: Tuple[float, float, float] = (-2000.0, 0.0, -1000.0)
    ball_omega: Tuple[float, float, float] = (0.0, 0.0, 75.0 * 2 * np.pi)
    rubber_friction: float = 0.6
    rubber_restitution: float = 0.85
    racket_angle: Tuple[float, float, float] = (0.0, -30.0, 0.0)
    racket_velocity: Tuple[float, float, float] = (2000.0, 0.0, 1000.0)
    ball_position: Tuple[float, float, float] = (
        200.0,
        TABLE_WIDTH / 2,
        TABLE_HEIGHT + 240.0,
    )
    contact_model: str = "coulomb"


@dataclass
class TrajectoryMoment:
    name: str
    index: int
    time: float
    point: Tuple[float, float, float]
    interval: Optional[Tuple[float, float]] = None
    midpoint: Optional[Tuple[float, float, float]] = None


TABLE_LEVEL = TABLE_HEIGHT + BALL_RADIUS
