"""Matplotlib drawing helpers for tables, balls, rackets, and result charts."""

from __future__ import annotations

import numpy as np

from ..constants import (
    BALL_RADIUS,
    NET_EXTRA,
    NET_HEIGHT,
    TABLE_HEIGHT,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from ..models import SimulationResult
from ..physics import rotation_matrix_xyz


def _elliptic_cylinder(
    center,
    radius_y,
    radius_z,
    thickness,
    rotation,
    resolution=32,
):
    u = np.linspace(0, 2 * np.pi, resolution)
    xs = np.array([-thickness / 2, thickness / 2])
    x_grid, u_grid = np.meshgrid(xs, u)
    local = np.stack(
        [
            x_grid,
            radius_y * np.cos(u_grid),
            radius_z * np.sin(u_grid),
        ],
        axis=0,
    ).reshape(3, -1)
    world = (rotation @ local).reshape(3, *x_grid.shape) + np.array(center).reshape(
        3, 1, 1
    )
    return world[0], world[1], world[2]


def draw_racket(ax, center=(0, 0, 0), angle=(0, 0, 0)) -> None:
    """Draw a standard racket as five blade layers plus a handle."""

    rotation = rotation_matrix_xyz(angle)
    layers = [
        ("black", 2.0),
        ("#f0c060", 2.2),
        ("#deb887", 6.0),
        ("#f0c060", 2.2),
        ("red", 2.0),
    ]
    offset = -sum(width for _, width in layers) / 2
    for color, width in layers:
        layer_center = np.array(center) + rotation @ np.array(
            [offset + width / 2, 0.0, 0.0]
        )
        x_grid, y_grid, z_grid = _elliptic_cylinder(
            layer_center, 75.0, 80.0, width, rotation
        )
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color=color,
            alpha=0.9,
            edgecolor="none",
        )
        offset += width

    handle_center = np.array(center) + rotation @ np.array([0.0, -120.0, 0.0])
    x_grid, y_grid, z_grid = _elliptic_cylinder(
        handle_center,
        25.0,
        18.0,
        95.0,
        rotation @ rotation_matrix_xyz((0.0, 0.0, 90.0)),
    )
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="#8b5a2b",
        alpha=0.95,
        edgecolor="none",
    )


def draw_ball(ax, center, color="white") -> None:
    """Draw a table-tennis ball centered at ``center``."""

    u, v_angle = np.mgrid[0 : 2 * np.pi : 25j, 0 : np.pi : 13j]
    ball_x = BALL_RADIUS * np.cos(u) * np.sin(v_angle) + center[0]
    ball_y = BALL_RADIUS * np.sin(u) * np.sin(v_angle) + center[1]
    ball_z = BALL_RADIUS * np.cos(v_angle) + center[2]
    ax.plot_surface(ball_x, ball_y, ball_z, color=color, edgecolor="none")


def draw_table(ax) -> None:
    """Draw the table, markings, and net without clearing the axes."""

    x_grid, y_grid = np.meshgrid([0, TABLE_LENGTH], [0, TABLE_WIDTH])
    z_grid = np.ones_like(x_grid) * TABLE_HEIGHT
    ax.plot_surface(x_grid, y_grid, z_grid, color="b", alpha=0.5)

    for x_values in ([0, 20], [TABLE_LENGTH - 20, TABLE_LENGTH]):
        x_grid, y_grid = np.meshgrid(x_values, [0, TABLE_WIDTH])
        z_grid = np.ones_like(x_grid) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(x_grid, y_grid, z_grid, color="w", edgecolor="none")
    for y_values in (
        [0, 20],
        [TABLE_WIDTH - 20, TABLE_WIDTH],
        [TABLE_WIDTH / 2 - 5, TABLE_WIDTH / 2 + 5],
    ):
        x_grid, y_grid = np.meshgrid([0, TABLE_LENGTH], y_values)
        z_grid = np.ones_like(x_grid) * (TABLE_HEIGHT + 0.1)
        ax.plot_surface(x_grid, y_grid, z_grid, color="w", edgecolor="none")

    y_grid, z_grid = np.meshgrid(
        [-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30],
        [TABLE_HEIGHT + 130, TABLE_HEIGHT + NET_HEIGHT],
    )
    x_grid = np.ones_like(z_grid) * TABLE_LENGTH / 2
    ax.plot_surface(x_grid, y_grid, z_grid, color="w")

    for y_values in (
        [-NET_EXTRA, -NET_EXTRA + 30],
        [TABLE_WIDTH + NET_EXTRA - 30, TABLE_WIDTH + NET_EXTRA],
    ):
        y_grid, z_grid = np.meshgrid(
            y_values, [TABLE_HEIGHT, TABLE_HEIGHT + NET_HEIGHT]
        )
        x_grid = np.ones_like(z_grid) * TABLE_LENGTH / 2
        ax.plot_surface(x_grid, y_grid, z_grid, color="k")

    net_square_side = 20
    net_lines_v = np.arange(
        -NET_EXTRA + 30,
        TABLE_WIDTH + NET_EXTRA - 30 + net_square_side,
        net_square_side,
    )
    net_lines_h = np.arange(
        TABLE_HEIGHT,
        TABLE_HEIGHT + 130 + net_square_side,
        net_square_side,
    )
    for z_value in net_lines_h:
        ax.plot(
            [TABLE_LENGTH / 2, TABLE_LENGTH / 2],
            [-NET_EXTRA + 30, TABLE_WIDTH + NET_EXTRA - 30],
            [z_value, z_value],
            "k",
            linewidth=0.5,
        )
    for y_value in net_lines_v:
        ax.plot(
            [TABLE_LENGTH / 2, TABLE_LENGTH / 2],
            [y_value, y_value],
            [TABLE_HEIGHT, TABLE_HEIGHT + 130],
            "k",
            linewidth=0.5,
        )


def set_scene_axes(ax, yaw: float, pitch: float) -> None:
    ax.view_init(pitch, yaw)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(-500, TABLE_LENGTH + 500)
    ax.set_ylim(-500, TABLE_WIDTH + 500)
    ax.set_zlim(0, 1500)
    ax.grid(True)


def plot_results(result: SimulationResult) -> None:
    """Plot position, velocity, rotation, and angular velocity."""

    import matplotlib.pyplot as plt

    if result.t is None:
        raise ValueError("SimulationResult.t is required to plot results")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(result.t, result.x.T / 1000)
    axes[0, 0].set(xlabel="time [s]", ylabel="position [m]", ylim=(0, 3))
    axes[1, 0].plot(result.t, result.v.T / 1000)
    axes[1, 0].set(xlabel="time [s]", ylabel="velocity [m/s]")
    axes[0, 1].plot(result.t, result.theta.T / (2 * np.pi))
    axes[0, 1].set(xlabel="time [s]", ylabel="rotation [rev]")
    axes[1, 1].plot(result.t, result.omega.T / (2 * np.pi))
    axes[1, 1].set(xlabel="time [s]", ylabel="angular velocity [rev/s]")
    fig.tight_layout()
    plt.show()
