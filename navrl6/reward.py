"""Reward function for the six-robot navigation task.

The total per-robot reward is a weighted sum of interpretable terms.  Every
term is documented in the reward table of ``README.md`` (Section 2): terms that
guide the robot toward the goal are positive, terms that punish unsafe or
wasteful behaviour are negative.

Reward components
-----------------
``progress``
    Shaping reward proportional to the reduction of the goal distance:
    ``w * (d_prev - d)``.  Positive when the robot approaches the goal,
    negative when it moves away.
``goal_bonus``
    One-off large positive bonus when the goal is reached.
``collision``
    Large negative penalty for colliding with an obstacle, another robot or a
    wall.
``clearance``
    Linear negative penalty inside the safety margin around obstacles.
``separation``
    Negative penalty when another robot enters the separation band.
``time``
    Small constant negative per-step cost, encouraging shortest solutions.
``control``
    Quadratic negative action-effort penalty, encouraging smooth motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class RewardResult:
    total: float
    breakdown: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        out = {"total": self.total}
        out.update(self.breakdown)
        return out


def compute_reward(
    prev_dist: float,
    dist: float,
    reached: bool,
    collided: bool,
    clearance: float,
    separation: float,
    action: np.ndarray,
    cfg: config.RewardConfig | None = None,
) -> RewardResult:
    """Compute the reward for one robot transition.

    Parameters
    ----------
    prev_dist, dist : float
        Goal distance before / after the transition (m).
    reached : bool
        Goal reached in this transition.
    collided : bool
        Collision detected in this transition.
    clearance : float
        Free space to the nearest obstacle (m), i.e. distance minus obstacle
        radius.  Negative on penetration.
    separation : float
        Free space to the nearest other robot (m).
    action : array-like, shape (2,)
        Executed velocity command ``(vx, vy)``.
    cfg : RewardConfig, optional
        Reward weights.  Defaults to :class:`config.RewardConfig`.
    """
    cfg = cfg or config.RewardConfig()
    action = np.asarray(action, dtype=float)

    # 1. Shaping: progress toward the goal (signed).
    progress = cfg.progress_weight * (prev_dist - dist)

    # 2. Goal-reaching bonus (positive).
    goal_bonus = cfg.goal_bonus if reached else 0.0

    # 3. Collision penalty (negative).
    collision = cfg.collision_penalty if collided else 0.0

    # 4. Clearance penalty inside the safety margin (negative).
    if clearance < config.SAFETY_MARGIN:
        clearance_penalty = cfg.clearance_penalty * (
            (config.SAFETY_MARGIN - clearance) / config.SAFETY_MARGIN
        )
    else:
        clearance_penalty = 0.0

    # 5. Robot-robot separation penalty (negative).
    separation_band = 2.0 * config.ROBOT_RADIUS + 0.6
    if separation < separation_band:
        separation_penalty = cfg.separation_penalty * (
            (separation_band - separation) / separation_band
        )
    else:
        separation_penalty = 0.0

    # 6. Per-step time cost (negative).
    time_penalty = cfg.time_penalty

    # 7. Control effort (negative, quadratic).
    control_penalty = cfg.control_penalty * float(np.dot(action, action))

    breakdown = {
        "progress": float(progress),
        "goal_bonus": float(goal_bonus),
        "collision": float(collision),
        "clearance": float(clearance_penalty),
        "separation": float(separation_penalty),
        "time": float(time_penalty),
        "control": float(control_penalty),
    }
    total = float(sum(breakdown.values()))
    return RewardResult(total=total, breakdown=breakdown)
