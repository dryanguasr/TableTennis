"""Physical, geometric, and numerical constants used by the simulator."""

from __future__ import annotations


# Units are millimetres, grams, seconds, and g·mm/s².
BALL_MASS = 2.7
BALL_RADIUS = 20.0
BALL_ROT_INERTIA = 2 / 3 * BALL_MASS * BALL_RADIUS**2

NET_RESTITUTION = 0.5
TABLE_FRICTION = 0.25

# ACE baseline from Dürr et al., Nature 652, 886–891 (2026).
# Air density is converted from 1.204 kg/m³ to g/mm³.
AIR_DENSITY = 1.204e-6
DRAG_COEFFICIENT = 0.55
MAGNUS_SCALE = 0.1
MAGNUS_OFFSET = 0.001
G = 9810.0

TABLE_LENGTH = 2740.0
TABLE_WIDTH = 1525.0
TABLE_HEIGHT = 760.0
NET_HEIGHT = 152.5
NET_EXTRA = 180.0

DT = 0.005
T_MAX = 1.5

# Default camera and animation settings.
PLOT_PERIOD = 5
YAW = -45.0
PITCH = 23.5
