"""Declarative pilot service and return presets."""

from __future__ import annotations

from ..constants import TABLE_HEIGHT, TABLE_WIDTH
from ..models import RacketImpactParameters


PILOT_SERVICE_PARAMS = RacketImpactParameters(
    ball_velocity=(0.0, 0.0, -2500.0),
    ball_omega=(0.0, 0.0, 0.0),
    rubber_friction=1.2,
    rubber_restitution=0.8,
    racket_angle=(80.0, -39.917547723834424, 43.99328907077693),
    racket_velocity=(
        11889.392438867555,
        -3254.6407255923978,
        -7032.871005925697,
    ),
    ball_position=(-300.0, TABLE_WIDTH * 0.16, TABLE_HEIGHT + 260.0),
)


# fraction, racket Y/Z angles, and racket X/Y/Z velocity.
RETURN_PRESET_VECTORS = {
    "cut_short": (
        0.50,
        -78.14191918,
        153.86744395,
        -8364.4985,
        -3688.3990,
        -563.4203,
    ),
    "cut_two_bounce": (
        0.52,
        -73.68370336,
        169.25697650,
        -7472.91053468,
        -3236.43705113,
        -1223.39926964,
    ),
    "cut_long": (
        0.52,
        -60.99848906,
        174.49909112,
        -7186.06164702,
        -1724.03438608,
        -2415.93112993,
    ),
    "top_two_bounce": (
        0.50,
        60.33717674,
        80.28041765,
        -8691.60743428,
        230.93454907,
        102.63689315,
    ),
    "top_long": (
        0.50,
        3.10621828,
        179.51627300,
        -3109.3123,
        -766.793262,
        8048.41311,
    ),
}


PROFILE_TARGETS = {
    "cut_short": ("short", (-15.0, 35.0, 10.0)),
    "cut_two_bounce": ("two_bounce", (-15.0, 35.0, 10.0)),
    "cut_long": ("long", (-15.0, 35.0, 10.0)),
    "top_two_bounce": ("two_bounce", (0.0, -45.0, 0.0)),
    "top_long": ("long", (0.0, -45.0, 0.0)),
}
