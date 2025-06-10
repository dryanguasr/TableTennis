"""3D table tennis ball simulation.

The module contains a small physics engine that approximates the motion of a
table tennis ball.  It can be executed as a script to visualise the motion or it
can be imported by other tools (e.g. notebooks) to reuse the ``simulate``
function.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants in the same units as the MATLAB code (mm, g, mN)
BALL_MASS = 2.7  # g
BALL_RADIUS = 20.25  # mm
BALL_ROT_INERTIA = 2 / 3 * BALL_MASS * BALL_RADIUS**2
TABLE_RESTITUTION = 0.77
NET_RESTITUTION = 0.5
DRAG = 2.7  # mN/(mm/s)
ROT_DRAG = 350.0  # mN mm/[rad/s]
MAGNUS = 0.01  # mN/(mm/s^2)
TABLE_FRICTION = 0.25
TABLE_LENGTH = 2740  # mm
TABLE_WIDTH = 1525  # mm
TABLE_HEIGHT = 760  # mm
NET_HEIGHT = 152.5  # mm
NET_EXTRA = 180  # mm
G = 9800  # mm/s^2

# Simulation parameters
ANIMATE = True
PLOT_PERIOD = 5
YAW = -45
PITCH = 23.5
DT = 0.005
T_MAX = 1.5


@dataclass
class InitialConditions:
    pos: Tuple[float, float, float] = (0.0, TABLE_WIDTH * 4 / 8, TABLE_HEIGHT + 2 * NET_HEIGHT)
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


def simulate(ic: InitialConditions, dt: float = DT, t_max: float = T_MAX) -> SimulationResult:
    """Simulate the ball trajectory.

    Parameters
    ----------
    ic:
        Initial conditions.
    dt:
        Time step in seconds.
    t_max:
        Total simulation time in seconds.
    """

    t = np.arange(0, t_max + dt, dt)
    n = len(t)

    x = np.zeros((3, n))
    v = np.zeros((3, n))
    a = np.zeros((3, n))
    theta = np.zeros((3, n))
    omega = np.zeros((3, n))
    alpha = np.zeros((3, n))

    x[:, 0] = ic.pos
    v[:, 0] = ic.vel
    omega[:, 0] = ic.omega

    for k in range(1, n):
        F = (
            G * BALL_MASS * np.array([0.0, 0.0, -1.0])
            - DRAG * v[:, k - 1]
            + MAGNUS * np.cross(omega[:, k - 1], v[:, k - 1])
        )
        a[:, k] = F / BALL_MASS
        v[:, k] = v[:, k - 1] + a[:, k] * dt
        x[:, k] = x[:, k - 1] + v[:, k] * dt

        tau = -ROT_DRAG * omega[:, k - 1]
        alpha[:, k] = tau / BALL_ROT_INERTIA
        omega[:, k] = omega[:, k - 1] + alpha[:, k] * dt
        theta[:, k] = theta[:, k - 1] + omega[:, k] * dt

        if (0 < x[0, k] < TABLE_LENGTH and 0 < x[1, k] < TABLE_WIDTH and x[2, k] < TABLE_HEIGHT + BALL_RADIUS):
            x[2, k] = TABLE_HEIGHT + BALL_RADIUS
            delta_lin_rot = np.cross(omega[:, k], np.array([0.0, 0.0, BALL_RADIUS])) - np.array(
                [v[0, k], v[1, k], 0.0]
            )
            v[:, k] += TABLE_FRICTION * delta_lin_rot
            omega[:, k] += TABLE_FRICTION * np.cross(delta_lin_rot, np.array([0.0, 0.0, 1.0])) / BALL_RADIUS
            v[2, k] = -TABLE_RESTITUTION * v[2, k]

        if (
            TABLE_LENGTH / 2 - BALL_RADIUS <= x[0, k] <= TABLE_LENGTH / 2 + BALL_RADIUS
            and -NET_EXTRA <= x[1, k] <= TABLE_WIDTH + NET_EXTRA
            and TABLE_HEIGHT + BALL_RADIUS < x[2, k] < TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
        ):
            omega[:, k] = NET_RESTITUTION * omega[:, k]
            v[0, k] = -NET_RESTITUTION * v[0, k]

    return SimulationResult(x, v, a, theta, omega, alpha)


def plot_table(ax: plt.Axes, pos, vel, acc, orient, ang_vel, ang_acc, yaw, pitch) -> None:
    """Draw the table, ball and vectors for a single frame."""

    ax.clear()

    X, Y = np.meshgrid([0, TABLE_LENGTH], [0, TABLE_WIDTH])
    Z = np.ones_like(X) * TABLE_HEIGHT
    ax.plot_surface(X, Y, Z, color="b", alpha=0.5)

    for xs in ([0, 20], [TABLE_LENGTH - 20, TABLE_LENGTH]):
        X, Y = np.meshgrid(xs, [0, TABLE_WIDTH])
        Z = np.ones_like(X) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(X, Y, Z, color="w", edgecolor="none")
    for ys in ([0, 20], [TABLE_WIDTH - 20, TABLE_WIDTH], [TABLE_WIDTH / 2 - 5, TABLE_WIDTH / 2 + 5]):
        X, Y = np.meshgrid([0, TABLE_LENGTH], ys)
        Z = np.ones_like(X) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(X, Y, Z, color="w", edgecolor="none")

    Y, Z = np.meshgrid([-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30], [TABLE_HEIGHT + 130, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="w")
    Y, Z = np.meshgrid([-NET_EXTRA, -NET_EXTRA + 30], [TABLE_HEIGHT, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="k")
    Y, Z = np.meshgrid([TABLE_WIDTH + NET_EXTRA - 30, TABLE_WIDTH + NET_EXTRA], [TABLE_HEIGHT, TABLE_HEIGHT + NET_HEIGHT])
    X = np.ones_like(Z) * TABLE_LENGTH / 2
    ax.plot_surface(X, Y, Z, color="k")

    net_square_side = 20
    net_lines_v = np.arange(-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30 + net_square_side, net_square_side)
    net_lines_h = np.arange(TABLE_HEIGHT, TABLE_HEIGHT + 130 + net_square_side, net_square_side)
    for z in net_lines_h:
        ax.plot([TABLE_LENGTH / 2, TABLE_LENGTH / 2], [-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30], [z, z], "k", linewidth=0.5)
    for y in net_lines_v:
        ax.plot([TABLE_LENGTH / 2, TABLE_LENGTH / 2], [y, y], [TABLE_HEIGHT, TABLE_HEIGHT + 130], "k", linewidth=0.5)

    u, v_ang = np.mgrid[0 : 2 * np.pi : 25j, 0 : np.pi : 13j]
    ball_x = BALL_RADIUS * np.cos(u) * np.sin(v_ang) + pos[0]
    ball_y = BALL_RADIUS * np.sin(u) * np.sin(v_ang) + pos[1]
    ball_z = BALL_RADIUS * np.cos(v_ang) + pos[2]
    ax.plot_surface(ball_x, ball_y, ball_z, color="w", edgecolor="none")

    ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], length=10 / 98, color="g")
    ax.quiver(pos[0], pos[1], pos[2], acc[0], acc[1], acc[2], length=1 / 49, color="c")
    ax.quiver(pos[0], pos[1], pos[2], ang_vel[0], ang_vel[1], ang_vel[2], length=1, color="r")

    ax.view_init(pitch, yaw)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(-500, TABLE_LENGTH + 500)
    ax.set_ylim(-500, TABLE_WIDTH + 500)
    ax.set_zlim(0, 1500)
    ax.grid(True)


def animate_simulation(result: SimulationResult, save: Optional[str] = None) -> None:
    """Animate the trajectory using ``matplotlib.animation``."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        plot_table(
            ax,
            result.x[:, frame],
            result.v[:, frame],
            result.a[:, frame],
            result.theta[:, frame],
            result.omega[:, frame],
            result.alpha[:, frame],
            YAW,
            PITCH,
        )

    frames = range(0, result.x.shape[1], PLOT_PERIOD)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=DT * 1000 * PLOT_PERIOD)

    if save:
        writer = animation.FFMpegWriter(fps=int(1 / (DT * PLOT_PERIOD)))
        ani.save(save, writer=writer)
    else:
        plt.show()


def plot_results(result: SimulationResult, dt: float = DT) -> None:
    """Plot position, velocity and rotation graphs after the simulation."""

    t = np.arange(result.x.shape[1]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, result.x.T / 1000)
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("position [m]")
    axes[0, 0].set_ylim(0, 3)

    axes[1, 0].plot(t, result.v.T / 1000)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("velocity [m/s]")

    axes[0, 1].plot(t, result.theta.T / (2 * np.pi))
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("rotation [rev]")

    axes[1, 1].plot(t, result.omega.T / (2 * np.pi))
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("angular velocity [rev/s]")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="3D table tennis simulation")
    parser.add_argument("--save", metavar="FILE", help="Save the animation to an mp4 file")
    args = parser.parse_args()

    result = simulate(InitialConditions())
    animate_simulation(result, save=args.save)


if __name__ == "__main__":
    main()
