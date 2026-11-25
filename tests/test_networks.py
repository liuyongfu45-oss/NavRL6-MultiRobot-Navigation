"""Unit tests for the actor-critic MLP."""

import numpy as np
import torch

from navrl6 import config
from navrl6.environment import ACT_DIM, OBS_DIM
from navrl6.networks import ActorCritic, build_model


def test_forward_shapes():
    model = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    obs = torch.randn(5, OBS_DIM)
    dist, value = model(obs)
    assert dist.mean.shape == (5, ACT_DIM)
    assert value.shape == (5,)


def test_evaluate_shapes():
    model = ActorCritic()
    obs = torch.randn(4, OBS_DIM)
    actions = torch.randn(4, ACT_DIM)
    log_prob, entropy, value = model.evaluate(obs, actions)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)


def test_backward_pass():
    model = ActorCritic()
    obs = torch.randn(3, OBS_DIM)
    dist, value = model(obs)
    actions = dist.sample()
    log_prob, _, value = model.evaluate(obs, actions)
    loss = -log_prob.mean() + value.pow(2).mean()
    loss.backward()
    # At least one parameter receives a gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads


def test_tanh_mean_bounded():
    model = ActorCritic()
    obs = torch.randn(100, OBS_DIM)
    mean, _ = model(obs)
    assert torch.all(mean.mean <= 1.0) and torch.all(mean.mean >= -1.0)


def test_build_model_from_config():
    ppo_cfg = config.PPOConfig()
    model = build_model(ppo_cfg)
    assert isinstance(model, ActorCritic)
    assert len(ppo_cfg.hidden_sizes) == 2
