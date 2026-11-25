"""Neural network architecture: shared actor-critic MLP (PPO).

**Network family.** A feed-forward fully-connected multi-layer perceptron
(MLP) with two separate heads:

* **Actor (policy) head** -- maps the 12-d robot observation to a diagonal
  Gaussian distribution over the 2-d velocity command.
* **Critic (value) head** -- maps the same observation to a scalar state
  value estimate.

Both heads share the same input representation but have their own hidden
layers.  The policy is homogeneous: the *same* network is applied to all six
robots.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from . import config
from .environment import ACT_DIM, OBS_DIM


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name!r}")


class ActorCritic(nn.Module):
    """Gaussian actor-critic MLP.

    Parameters
    ----------
    obs_dim, act_dim : int
        Observation / action dimension.
    hidden_sizes : tuple[int, ...]
        Width of each hidden layer in actor and critic.
    activation : str
        ``relu``, ``tanh`` or ``elu``.
    init_log_std : float
        Initial value of the (learnable) log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()

        # Actor layers
        actor_layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            actor_layers += [nn.Linear(last, h), _activation(activation)]
            last = h
        actor_layers += [nn.Linear(last, act_dim), nn.Tanh()]
        self.actor = nn.Sequential(*actor_layers)

        # Critic layers
        critic_layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            critic_layers += [nn.Linear(last, h), _activation(activation)]
            last = h
        critic_layers += [nn.Linear(last, 1)]
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    # ------------------------------------------------------------------ #
    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


def build_model(ppo_cfg: config.PPOConfig, obs_dim: int = OBS_DIM,
                act_dim: int = ACT_DIM) -> ActorCritic:
    """Instantiate the actor-critic from a :class:`config.PPOConfig`."""
    return ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=tuple(ppo_cfg.hidden_sizes),
        activation=ppo_cfg.activation,
        init_log_std=ppo_cfg.init_log_std,
    )
