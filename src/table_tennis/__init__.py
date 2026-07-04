"""Table-tennis physics, impact, event, and exchange simulation."""

from .events import identify_trajectory_moments
from .models import (
    InitialConditions,
    RacketImpactParameters,
    SimulationEvent,
    SimulationResult,
    TrajectoryMoment,
)
from .physics import simulate, simulate_racket_impact
from .rally import (
    ExerciseDefinition,
    ExerciseResult,
    ExerciseStroke,
    RallySegment,
    simulate_exercise,
)

__version__ = "0.1.0"

__all__ = [
    "InitialConditions",
    "RacketImpactParameters",
    "SimulationEvent",
    "SimulationResult",
    "TrajectoryMoment",
    "identify_trajectory_moments",
    "simulate",
    "simulate_racket_impact",
    "ExerciseDefinition",
    "ExerciseResult",
    "ExerciseStroke",
    "RallySegment",
    "simulate_exercise",
]
