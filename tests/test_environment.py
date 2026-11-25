"""Unit tests for the multi-robot navigation environment."""

import numpy as np
import pytest

from navrl6 import config
from navrl6.environment import ACT_DIM, OBS_DIM, MultiRobotNavEnv


@pytest.mark.parametrize("scenario", ["static", "dynamic", "mixed"])
def test_reset_shape(scenario):
    env = MultiRobotNavEnv(scenario=scenario, seed=1)
    obs = env.reset()
    assert obs.shape == (config.NUM_ROBOTS, OBS_DIM)
    assert np.isfinite(obs).all()


def test_step_outputs():
    env = MultiRobotNavEnv(scenario="static", seed=2)
    env.reset()
    actions = np.zeros((config.NUM_ROBOTS, ACT_DIM))
    obs, rewards, dones, info = env.step(actions)
    assert obs.shape == (config.NUM_ROBOTS, OBS_DIM)
    assert rewards.shape == (config.NUM_ROBOTS,)
    assert dones.shape == (config.NUM_ROBOTS,)
    assert "episode_done" in info
    assert len(info["robots"]) == config.NUM_ROBOTS


def test_obstacle_counts():
    static_env = MultiRobotNavEnv(scenario="static", seed=3)
    assert len(static_env.static_obstacles()) == 15
    assert len(static_env.dynamic_obstacles()) == 0

    dynamic_env = MultiRobotNavEnv(scenario="dynamic", seed=3)
    assert len(dynamic_env.static_obstacles()) == 0
    assert len(dynamic_env.dynamic_obstacles()) == 15

    mixed_env = MultiRobotNavEnv(scenario="mixed", seed=3)
    assert len(mixed_env.static_obstacles()) == config.N_STATIC_MIXED
    assert len(mixed_env.dynamic_obstacles()) == config.N_DYNAMIC_MIXED
    assert len(mixed_env.obstacles) == 15


def test_deterministic_layout_with_seed():
    a = MultiRobotNavEnv(scenario="mixed", seed=42)
    b = MultiRobotNavEnv(scenario="mixed", seed=42)
    for oa, ob in zip(a.obstacles, b.obstacles):
        np.testing.assert_allclose(oa.position, ob.position)


def test_goal_reached():
    env = MultiRobotNavEnv(scenario="static", seed=5)
    env.reset()
    # Place robot 0 directly on its goal.
    env.robots[0] = env.goals[0].copy()
    actions = np.zeros((config.NUM_ROBOTS, ACT_DIM))
    _, _, dones, info = env.step(actions)
    assert env.reached[0]
    assert info["robots"][0]["reached"]


def test_obstacle_collision():
    env = MultiRobotNavEnv(scenario="static", seed=6)
    env.reset()
    ob = env.static_obstacles()[0]
    # Place robot 0 inside the obstacle disk.
    env.robots[0] = ob.position.copy()
    actions = np.zeros((config.NUM_ROBOTS, ACT_DIM))
    env.step(actions)
    assert env.collided[0]


def test_wall_collision():
    env = MultiRobotNavEnv(scenario="static", seed=7)
    env.reset()
    env.robots[0] = np.array([config.WORLD_SIZE, 0.0])
    actions = np.zeros((config.NUM_ROBOTS, ACT_DIM))
    env.step(actions)
    assert env.collided[0]


def test_action_speed_clipped():
    env = MultiRobotNavEnv(scenario="static", seed=8)
    env.reset()
    actions = np.array([[10.0, 0.0]] * config.NUM_ROBOTS)
    env.step(actions)
    speed = np.linalg.norm(env.velocities[0])
    assert speed <= config.ROBOT_MAX_SPEED + 1e-6


def test_reward_breakdown_present():
    env = MultiRobotNavEnv(scenario="dynamic", seed=9)
    env.reset()
    env.step(np.zeros((config.NUM_ROBOTS, ACT_DIM)))
    for robot_info in env.last_info["robots"]:
        if "reward_breakdown" in robot_info:
            assert set(robot_info["reward_breakdown"]) == {
                "progress", "goal_bonus", "collision", "clearance",
                "separation", "time", "control",
            }


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        MultiRobotNavEnv(scenario="rainy", seed=0)
