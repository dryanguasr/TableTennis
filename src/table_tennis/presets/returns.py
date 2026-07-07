"""Declarative pilot service and return presets."""

from __future__ import annotations

from ..constants import TABLE_HEIGHT, TABLE_WIDTH
from ..models import RacketImpactParameters


PILOT_SERVICE_PARAMS = RacketImpactParameters(
    ball_velocity=(0.0, 0.0, -2500.0),
    ball_omega=(0.0, 0.0, 0.0),
    rubber_friction=1.2,
    rubber_restitution=0.8,
    racket_angle=(80.0, -26.012939763783727, 157.69325761986929),
    racket_velocity=(
        8220.448364203,
        5443.895728504967,
        4499.294508077814,
    ),
    ball_position=(-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
)


# fraction, racket Y/Z angles, and racket X/Y/Z velocity.
RETURN_PRESET_VECTORS = {
    "cut_short": (
        0.6058127526,
        -71.4352306922,
        154.8932070839,
        -7196.3299367566,
        -3134.9159620040,
        -1040.5907323439,
    ),
    "cut_two_bounce": (
        0.5745186922,
        -69.6102375679,
        156.7030217977,
        -8132.2199894941,
        -3296.1144429588,
        -1542.9377203099,
    ),
    "cut_long": (
        0.6255100331,
        -49.0935933681,
        176.2071834963,
        -7242.1824750261,
        -1102.4321345841,
        -3792.4900882322,
    ),
    "top_two_bounce": (
        0.1174897666,
        13.4307678979,
        167.7057370057,
        -2191.6341778466,
        374.1802467606,
        8099.0045864804,
    ),
    "top_long": (
        0.2009658354,
        7.5494181416,
        174.7355679470,
        -5350.8094000692,
        -1567.5107481067,
        7307.6792295766,
    ),
}


PROFILE_TARGETS = {
    "cut_short": ("short", (-20.0, 35.0, 15.0)),
    "cut_two_bounce": ("two_bounce", (-20.0, 35.0, 15.0)),
    "cut_long": ("long", (-20.0, 35.0, 15.0)),
    "top_two_bounce": ("two_bounce", (0.0, -45.0, 0.0)),
    "top_long": ("long", (0.0, -45.0, 0.0)),
}

# Topspin retains more forward speed after the ACE table transition than a
# chopped ball. Its reproducible two-bounce preset therefore targets a point
# closer to the net while keeping the shared UI default adjustable.
PROFILE_TARGET_X = {
    "top_two_bounce": 1200.0,
}
