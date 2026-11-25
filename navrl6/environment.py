"""Multi-robot navigation environment (NavRL6).

A Gym-style 2D continuous-control environment with six holonomic robots
navigating to their own goal stars while avoiding 15 obstacles.  Three
scenarios are supported:

* ``static``   -- 15 fixed circular obstacles;
* ``dynamic``  -- 15 moving obstacles that bounce off the walls;
* ``mixed``    -- 8 static + 7 dynamic obstacles.

Observation (per robot, 12 dimensions)
--------------------------------------
====  ===============================================
dim   content
====  ===============================================
0-1   goal relative vector ``(goal - robot)`` (m)
2     Euclidean goal distance (m)
3-4   robot velocity ``(vx, vy)`` (m/s)
5-6   nearest obstacle relative vector (m)
7     nearest obstacle free-space distance (m)
8-9   nearest other robot relative vector (m)
10    nearest other robot free-space distance (m)
11    fraction of episode steps remaining
====  ===============================================

Action (per robot, 2 dimensions)
-------------------------------
Velocity command ``(vx, vy)`` in m/s, clipped to
``[-ROBOT_MAX_SPEED, ROBOT_MAX_SPEED]``.
"""

from __future__ import annotations

import numpy as np

from . import config, scenarios
from .config import RewardConfig
from .obstacles import Obstacle, build_obstacles
from .reward import compute_reward

OBS_DIM = 12
ACT_DIM = 2


class MultiRobotNavEnv:
    """Six-robot navigation arena."""

    def __init__(
        self,
        scenario: str = "mixed",
        seed: int = 0,
        reward_cfg: RewardConfig | None = None,
    ) -> None:
        if scenario not in scenarios.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario!r}")
        self.scenario = scenario
        self.seed = seed
        self.reward_cfg = reward_cfg or RewardConfig()

        self.rng = np.random.default_rng(seed)
        self.robots = scenarios.ROBOT_STARTS.copy()
        self.goals = scenarios.ROBOT_GOALS.copy()
        self.velocities = np.zeros((config.NUM_ROBOTS, ACT_DIM))
        self.obstacles: list[Obstacle] = build_obstacles(scenario, seed)

        self.reached = np.zeros(config.NUM_ROBOTS, dtype=bool)
        self.collided = np.zeros(config.NUM_ROBOTS, dtype=bool)
        self.step_count = 0
        self._prev_dist = self.goal_distances().copy()
        self.trajectories: list[list[np.ndarray]] = [[] for _ in range(config.NUM_ROBOTS)]
        self.last_info: dict = {}

    # ------------------------------------------------------------------ #
    # State queries
    # ------------------------------------------------------------------ #
    def goal_distances(self) -> np.ndarray:
        return np.linalg.norm(self.goals - self.robots, axis=1)

    def _nearest_obstacle(self, i: int) -> tuple[np.ndarray, float]:
        best_vec = np.zeros(2)
        best_free = np.inf
        for ob in self.obstacles:
            vec = ob.position - self.robots[i]
            free = np.linalg.norm(vec) - ob.radius
            if free < best_free:
                best_free, best_vec = free, vec
        return best_vec, float(best_free)

    def _nearest_robot(self, i: int) -> tuple[np.ndarray, float]:
        best_vec = np.zeros(2)
        best_free = np.inf
        for j in range(config.NUM_ROBOTS):
            if j == i:
                continue
            vec = self.robots[j] - self.robots[i]
            free = np.linalg.norm(vec) - 2.0 * config.ROBOT_RADIUS
            if free < best_free:
                best_free, best_vec = free, vec
        return best_vec, float(best_free)

    def _wall_collision(self, i: int) -> bool:
        p = self.robots[i]
        return bool(
            np.any(p > config.WORLD_SIZE - config.ROBOT_RADIUS)
            or np.any(p < -config.WORLD_SIZE + config.ROBOT_RADIUS)
        )

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.robots = scenarios.ROBOT_STARTS.copy()
        self.velocities = np.zeros((config.NUM_ROBOTS, ACT_DIM))
        self.obstacles = build_obstacles(self.scenario, self.seed)
        self.reached = np.zeros(config.NUM_ROBOTS, dtype=bool)
        self.collided = np.zeros(config.NUM_ROBOTS, dtype=bool)
        self.step_count = 0
        self._prev_dist = self.goal_distances().copy()
        self.trajectories = [
            [self.robots[i].copy()] for i in range(config.NUM_ROBOTS)
        ]
        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=float).reshape(
            config.NUM_ROBOTS, ACT_DIM
        )
        # Clip command to the maximum speed norm.
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        scale = np.minimum(1.0, config.ROBOT_MAX_SPEED / np.maximum(norms, 1e-8))
        actions = actions * scale

        collision_flag = np.zeros(config.NUM_ROBOTS, dtype=bool)

        for i in range(config.NUM_ROBOTS):
            if self.reached[i] or self.collided[i]:
                self.velocities[i] = 0.0
                continue

            self.velocities[i] = actions[i]
            self.robots[i] = self.robots[i] + self.velocities[i] * config.DT
            self.trajectories[i].append(self.robots[i].copy())

            # Wall collision.
            if self._wall_collision(i):
                # Push the robot back inside the arena.
                for axis in (0, 1):
                    if self.robots[i, axis] > config.WORLD_SIZE - config.ROBOT_RADIUS:
                        self.robots[i, axis] = config.WORLD_SIZE - config.ROBOT_RADIUS
                    elif self.robots[i, axis] < -config.WORLD_SIZE + config.ROBOT_RADIUS:
                        self.robots[i, axis] = -config.WORLD_SIZE + config.ROBOT_RADIUS
                collision_flag[i] = True

            # Obstacle collision.
            for ob in self.obstacles:
                if ob.collision_with(self.robots[i], config.ROBOT_RADIUS):
                    collision_flag[i] = True
                    break

        # Robot-robot collision (mutual).
        for i in range(config.NUM_ROBOTS):
            if self.reached[i] or self.collided[i]:
                continue
            for j in range(i + 1, config.NUM_ROBOTS):
                if self.reached[j] or self.collided[j]:
                    continue
                if np.linalg.norm(self.robots[i] - self.robots[j]) < 2.0 * config.ROBOT_RADIUS:
                    collision_flag[i] = True
                    collision_flag[j] = True

        # Advance dynamic obstacles after robot movement.
        for ob in self.obstacles:
            ob.update(config.DT, config.WORLD_SIZE)

        # Goal / collision bookkeeping and reward.
        dist = self.goal_distances()
        rewards = np.zeros(config.NUM_ROBOTS)
        infos: list[dict] = []

        for i in range(config.NUM_ROBOTS):
            if self.reached[i] or self.collided[i]:
                infos.append({"reached": self.reached[i], "collided": self.collided[i]})
                continue

            reached = bool(dist[i] <= config.GOAL_RADIUS)
            collided = bool(collision_flag[i])
            _, clearance = self._nearest_obstacle(i)
            _, separation = self._nearest_robot(i)

            result = compute_reward(
                prev_dist=float(self._prev_dist[i]),
                dist=float(dist[i]),
                reached=reached,
                collided=collided,
                clearance=clearance,
                separation=separation,
                action=actions[i],
                cfg=self.reward_cfg,
            )
            rewards[i] = result.total

            if reached:
                self.reached[i] = True
            if collided:
                self.collided[i] = True
                self.velocities[i] = 0.0

            infos.append(
                {
                    "reached": self.reached[i],
                    "collided": self.collided[i],
                    "goal_dist": float(dist[i]),
                    "clearance": clearance,
                    "separation": separation,
                    "reward_breakdown": result.breakdown,
                }
            )

        self._prev_dist = dist.copy()
        self.step_count += 1
        self.last_info = {"robots": infos}

        dones = self.reached | self.collided
        episode_done = bool(np.all(dones) or self.step_count >= config.MAX_STEPS)
        info = {
            "robots": infos,
            "episode_done": episode_done,
            "step": self.step_count,
            "time": self.step_count * config.DT,
            "reached_count": int(self.reached.sum()),
        }
        return self._get_obs(), rewards, dones, info

    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((config.NUM_ROBOTS, OBS_DIM))
        time_left = 1.0 - self.step_count / config.MAX_STEPS
        for i in range(config.NUM_ROBOTS):
            goal_vec = self.goals[i] - self.robots[i]
            obs[i, 0:2] = goal_vec
            obs[i, 2] = np.linalg.norm(goal_vec)
            obs[i, 3:5] = self.velocities[i]
            obs_vec, obs_free = self._nearest_obstacle(i)
            obs[i, 5:7] = obs_vec
            obs[i, 7] = obs_free
            rob_vec, rob_free = self._nearest_robot(i)
            obs[i, 8:10] = rob_vec
            obs[i, 10] = rob_free
            obs[i, 11] = time_left
        return obs

    # ------------------------------------------------------------------ #
    # Helpers used by the visualiser / evaluator
    # ------------------------------------------------------------------ #
    def static_obstacles(self) -> list[Obstacle]:
        return [o for o in self.obstacles if not o.is_dynamic]

    def dynamic_obstacles(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.is_dynamic]
