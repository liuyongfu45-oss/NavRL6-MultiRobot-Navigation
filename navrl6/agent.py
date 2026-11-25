"""Proximal Policy Optimization (PPO) agent for the shared actor-critic MLP.

The agent pools transitions collected from all six robots (homogeneous policy)
and optimises the clipped surrogate objective with a separate critic head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import config
from .environment import ACT_DIM, OBS_DIM
from .networks import build_model


class PPOAgent:
    """PPO wrapper around :class:`networks.ActorCritic`."""

    def __init__(
        self,
        ppo_cfg: config.PPOConfig | None = None,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        device: str = "cpu",
    ) -> None:
        self.cfg = ppo_cfg or config.PPOConfig()
        self.device = torch.device(device)
        self.model = build_model(self.cfg, obs_dim, act_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.cfg.lr_actor
        )
        self.critic_opt = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.cfg.lr_critic
        )

        # Rollout buffer
        self.clear_buffer()

    # ------------------------------------------------------------------ #
    # Buffer
    # ------------------------------------------------------------------ #
    def clear_buffer(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: np.ndarray, env=None, deterministic: bool = False) -> np.ndarray:
        # ``env`` is accepted for a uniform policy interface with the reactive
        # baseline; the learned policy only depends on the observation.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action, _, _ = self.model.act(obs_t, deterministic=deterministic)
        action = action.cpu().numpy()
        return action * config.ROBOT_MAX_SPEED

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    def _compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=bool)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else values[t + 1] if t + 1 < len(rewards) else 0.0
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return (
            torch.as_tensor(advantages, dtype=torch.float32),
            torch.as_tensor(returns, dtype=torch.float32),
        )

    def update(self) -> dict[str, float]:
        if not self.obs:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        obs_t = torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        advantages, returns = self._compute_advantages()
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.obs)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                log_prob, entropy, value = self.model.evaluate(obs_t[idx], act_t[idx])

                ratio = torch.exp(log_prob - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio,
                                     1 + self.cfg.clip_ratio) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, returns[idx])
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy.mean()
                )

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        self.clear_buffer()
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
