"""Stable access to calibrated direct and racket service presets.

The benchmark modules own execution and reporting; consumers should use this
module to obtain preset cases and selector metadata.
"""

from __future__ import annotations

from ..benchmarks.direct import (
    DEPTHS as DIRECT_DEPTHS,
    LANES as DIRECT_LANES,
    SERVICE_TYPES as DIRECT_SERVICE_TYPES,
    VELOCITY_OVERRIDES,
    BenchmarkCase as DirectServiceCase,
    build_cases as build_direct_cases,
)
from ..benchmarks.racket import (
    DEPTHS as RACKET_DEPTHS,
    LANES as RACKET_LANES,
    SERVICE_TYPES as RACKET_SERVICE_TYPES,
    BenchmarkCase as RacketServiceCase,
    build_cases as build_racket_cases,
)

__all__ = [
    "DIRECT_DEPTHS",
    "DIRECT_LANES",
    "DIRECT_SERVICE_TYPES",
    "RACKET_DEPTHS",
    "RACKET_LANES",
    "RACKET_SERVICE_TYPES",
    "VELOCITY_OVERRIDES",
    "DirectServiceCase",
    "RacketServiceCase",
    "build_direct_cases",
    "build_racket_cases",
]
