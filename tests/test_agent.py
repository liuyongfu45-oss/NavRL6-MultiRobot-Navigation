"""Unit tests for the PPO agent."""

import numpy as np
import torch

from navrl6 import config
from navrl6.agent import PPOAgent
from navrl6.environment import ACT_DIM, OBS_DIM


def test_act_shape_and_speed():
    agent = PPOAgent()
    obs = np.zeros(OBS_DIM)
    action = agent.act(obs)
    assert action.shape == (ACT_DIM,)
    assert np.linalg.norm(action) <= config.ROBOT_MAX_SPEED + 1e-6


def test_update_runs():
    agent = PPOAgent()
    for _ in range(5):
        agent.store(
            obs=np.random.randn(OBS_DIM),
            action=np.random.randn(ACT_DIM),
            log_prob=0.0,
            value=0.0,
            reward=1.0,
            done=False,
        )
    losses = agent.update()
    assert {"policy_loss", "value_loss", "entropy"} <= set(losses)


def test_save_load_round_trip(tmp_path):
    agent = PPOAgent()
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    restored = PPOAgent()
    restored.load(str(path))
    obs = np.zeros(OBS_DIM)
    a1 = agent.act(obs, deterministic=True)
    a2 = restored.act(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)
