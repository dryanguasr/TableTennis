"""Physical, geometric, and numerical constants used by the simulator."""

from __future__ import annotations


# Units follow the original MATLAB model: millimetres, grams, seconds, and mN.
BALL_MASS = 2.7
BALL_RADIUS = 20.25
BALL_ROT_INERTIA = 2 / 3 * BALL_MASS * BALL_RADIUS**2

TABLE_RESTITUTION = 0.89
NET_RESTITUTION = 0.5
TABLE_FRICTION = 0.25

DRAG = 2.7
ROT_DRAG = 350.0
MAGNUS = 0.01
G = 9800.0

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
