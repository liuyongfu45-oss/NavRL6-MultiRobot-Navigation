"""Visualisation: arena rendering, trajectory plots, metric charts and video.

Rendering style matches the three reference videos: coloured robot disks,
matching-colour goal stars, grey static obstacles, orange dynamic obstacles,
and drawn trajectory trails.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, scenarios
from .environment import MultiRobotNavEnv

_ROBOT_LABELS = [f"Robot {i + 1}" for i in range(config.NUM_ROBOTS)]


def render_frame(env: MultiRobotNavEnv, ax: plt.Axes, title: str | None = None) -> None:
    """Draw the current arena state on ``ax``."""
    ax.clear()
    limit = config.WORLD_SIZE + 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(True, alpha=0.3)

    # Trails
    for i, trail in enumerate(env.trajectories):
        if len(trail) > 1:
            pts = np.asarray(trail)
            ax.plot(pts[:, 0], pts[:, 1], color=config.ROBOT_COLORS[i],
                    linewidth=1.4, alpha=0.85)

    # Static obstacles (grey)
    for ob in env.static_obstacles():
        ax.add_patch(plt.Circle(ob.position, ob.radius, color=config.STATIC_COLOR, alpha=0.8))
    # Dynamic obstacles (orange)
    for ob in env.dynamic_obstacles():
        ax.add_patch(plt.Circle(ob.position, ob.radius, color=config.DYNAMIC_COLOR, alpha=0.85))

    # Goal stars
    for i in range(config.NUM_ROBOTS):
        ax.scatter(*env.goals[i], marker="*", s=180,
                   color=config.ROBOT_COLORS[i], edgecolor="white", zorder=4)

    # Robots
    for i in range(config.NUM_ROBOTS):
        ax.scatter(*env.robots[i], s=160, color=config.ROBOT_COLORS[i],
                   edgecolor="white", zorder=5)

    if title is None:
        title = (f"{config.SCENARIO_TITLES[env.scenario]} | step {env.step_count} | "
                 f"Time {env.step_count * config.DT:.2f}s | "
                 f"reached {env.reached.sum()}/{config.NUM_ROBOTS}")
    ax.set_title(title, fontsize=11)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=config.ROBOT_COLORS[i],
                   markersize=11, label=_ROBOT_LABELS[i])
        for i in range(config.NUM_ROBOTS)
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=config.DYNAMIC_COLOR, markersize=11,
                                     label="Dynamic obstacle"))
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=config.STATIC_COLOR, markersize=11,
                                     label="Static obstacle"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)


def plot_trajectories(env: MultiRobotNavEnv, scenario: str, out_path: str | Path) -> None:
    """Save the final trajectory plot of a finished episode."""
    fig, ax = plt.subplots(figsize=(7, 7))
    render_frame(env, ax, title=config.SCENARIO_TITLES[scenario])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scenario_comparison(stats: list, out_path: str | Path) -> None:
    """Grouped bar chart comparing the three scenarios."""
    metrics = ["success_rate", "collision_rate", "spl", "avg_min_clearance"]
    labels = ["Success rate", "Collision rate", "SPL", "Avg min clearance (m)"]
    scenarios_names = [s.scenario.capitalize() for s in stats]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    x = np.arange(len(scenarios_names))
    for ax, metric, label in zip(axes, metrics, labels):
        values = [getattr(s, metric) for s in stats]
        bars = ax.bar(x, values, color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios_names)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(history: dict, out_path: str | Path) -> None:
    """Plot mean episode reward and reached-count during training."""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    episodes = np.arange(1, len(history["mean_reward"]) + 1)
    ax1.plot(episodes, history["mean_reward"], color="#4c72b0", label="Mean reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean episode reward", color="#4c72b0")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(episodes, history["reached"], color="#dd8452", label="Reached robots")
    ax2.set_ylabel("Reached / 6", color="#dd8452")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def record_video(
    env: MultiRobotNavEnv,
    policy,
    out_path: str | Path,
    fps: int = 30,
) -> None:
    """Run an episode and save an MP4 video of the arena."""
    from matplotlib.animation import FuncAnimation

    obs = env.reset()
    frames: list[MultiRobotNavEnv] = []

    while True:
        frames.append(_snapshot(env))
        actions = policy.act(obs, env)
        obs, _, _, info = env.step(actions)
        if info["episode_done"]:
            break
    frames.append(_snapshot(env))

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    def _draw(frame_idx: int) -> None:
        _restore(env, frames[frame_idx])
        render_frame(env, ax)

    anim = FuncAnimation(fig, _draw, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, fps=fps, writer="ffmpeg")
    plt.close(fig)


def _snapshot(env: MultiRobotNavEnv) -> dict:
    return {
        "robots": env.robots.copy(),
        "velocities": env.velocities.copy(),
        "reached": env.reached.copy(),
        "collided": env.collided.copy(),
        "step_count": env.step_count,
        "trajectories": [t.copy() for t in env.trajectories],
        "obstacles": [(o.position.copy(), o.radius,
                        None if o.velocity is None else o.velocity.copy(), o.kind)
                       for o in env.obstacles],
    }


def _restore(env: MultiRobotNavEnv, snap: dict) -> None:
    env.robots = snap["robots"]
    env.velocities = snap["velocities"]
    env.reached = snap["reached"]
    env.collided = snap["collided"]
    env.step_count = snap["step_count"]
    env.trajectories = snap["trajectories"]
    for ob, data in zip(env.obstacles, snap["obstacles"]):
        ob.position, ob.radius, ob.velocity, ob.kind = data
