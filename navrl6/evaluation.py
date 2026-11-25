"""Rollout runner and a potential-field baseline policy.

The runner evaluates any policy that exposes ``act(obs, env) -> actions``.
Two policies are provided / supported:

* :class:`ReactivePolicy` -- a hand-tuned artificial potential-field baseline
  used to sanity-check the environment and to produce reference trajectories;
* :class:`agent.PPOAgent` -- the trained reinforcement-learning policy.
"""

from __future__ import annotations

import numpy as np

from . import config, scenarios
from .environment import MultiRobotNavEnv
from .metrics import RobotEpisode, build_record


class ReactivePolicy:
    """Artificial potential-field baseline (goal attraction + obstacle avoidance).

    The controller steers toward the goal, adds repulsive vectors from nearby
    obstacles / robots / walls, and a tangential (wall-following) component so
    that it can route around obstacles instead of stopping in a local minimum.
    It does **not** learn; it serves as a non-RL reference for comparison.
    """

    def __init__(
        self,
        speed: float = 1.7,
        influence: float = 3.2,
        repulsion: float = 3.0,
        tangent: float = 0.9,
    ) -> None:
        self.speed = speed
        self.influence = influence
        self.repulsion = repulsion
        self.tangent = tangent

    def act(self, obs: np.ndarray, env: MultiRobotNavEnv) -> np.ndarray:
        actions = np.zeros((config.NUM_ROBOTS, 2))
        for i in range(config.NUM_ROBOTS):
            goal_vec = env.goals[i] - env.robots[i]
            goal_dist = np.linalg.norm(goal_vec)
            if goal_dist < 1e-6:
                actions[i] = 0.0
                continue
            attractive = goal_vec / goal_dist

            repulsive = np.zeros(2)
            tangential = np.zeros(2)
            for ob in env.obstacles:
                vec = env.robots[i] - ob.position
                dist = np.linalg.norm(vec)
                free = dist - ob.radius
                if 0 < free < self.influence:
                    unit = vec / (dist + 1e-8)
                    weight = 1.0 - free / self.influence
                    repulsive += unit * self.repulsion * weight
                    # Perpendicular sidestep, chosen on the goal-facing side.
                    perp = np.array([-unit[1], unit[0]])
                    if np.dot(perp, attractive) < 0:
                        perp = -perp
                    tangential += perp * self.tangent * weight

            for j in range(config.NUM_ROBOTS):
                if j == i:
                    continue
                vec = env.robots[i] - env.robots[j]
                dist = np.linalg.norm(vec)
                if 0 < dist < 2.0:
                    repulsive += (vec / (dist + 1e-8)) * 1.2 * (1.0 - dist / 2.0)

            # Wall repulsion near the boundary.
            for axis in (0, 1):
                margin = config.WORLD_SIZE - abs(env.robots[i, axis])
                if margin < 2.0:
                    repulsive[axis] += -np.sign(env.robots[i, axis]) * \
                        2.0 * (1.0 - margin / 2.0)

            force = attractive + repulsive + tangential
            norm = np.linalg.norm(force)
            if norm > 1e-8:
                force = force / norm * self.speed
            actions[i] = force
        return actions


def rollout(
    env: MultiRobotNavEnv,
    policy,
    deterministic: bool = True,
) -> tuple[list[RobotEpisode], dict]:
    """Run one episode and return per-robot records plus run history."""
    obs = env.reset()
    episode_reward = np.zeros(config.NUM_ROBOTS)
    min_clearance = np.full(config.NUM_ROBOTS, np.inf)
    history = {"reached": [], "mean_reward": []}

    while True:
        actions = policy.act(obs, env)
        obs, rewards, dones, info = env.step(actions)
        episode_reward += rewards

        for i, robot_info in enumerate(info["robots"]):
            if "clearance" in robot_info:
                min_clearance[i] = min(min_clearance[i], robot_info["clearance"])

        history["reached"].append(info["reached_count"])
        history["mean_reward"].append(float(np.mean(episode_reward)))

        if info["episode_done"]:
            break

    records: list[RobotEpisode] = []
    for i in range(config.NUM_ROBOTS):
        records.append(
            build_record(
                trajectory=env.trajectories[i],
                reached=env.reached[i],
                collided=env.collided[i],
                steps=env.step_count,
                min_clearance=min_clearance[i],
                total_reward=episode_reward[i],
                init_position=scenarios.ROBOT_STARTS[i],
                goal=scenarios.ROBOT_GOALS[i],
            )
        )
    return records, history


def evaluate_scenario(
    scenario: str,
    policy,
    episodes: int = 10,
    seed: int = 0,
) -> list[RobotEpisode]:
    """Run ``episodes`` rollouts on one scenario and collect all robot records."""
    all_records: list[RobotEpisode] = []
    for ep in range(episodes):
        env = MultiRobotNavEnv(scenario=scenario, seed=seed + ep)
        records, _ = rollout(env, policy)
        all_records.extend(records)
    return all_records
