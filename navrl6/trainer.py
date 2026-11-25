"""PPO training loop for the six-robot navigation environment.

One shared policy is trained for all robots; transitions from every robot are
pooled into the same PPO buffer.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config
from .agent import PPOAgent
from .environment import MultiRobotNavEnv


def train(
    scenario: str = "mixed",
    episodes: int = 200,
    ppo_cfg: config.PPOConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[PPOAgent, dict]:
    """Train a PPO agent on one scenario.

    Returns
    -------
    agent : PPOAgent
        Trained agent.
    history : dict
        ``mean_reward`` and ``reached`` per episode, for the training curve.
    """
    ppo_cfg = ppo_cfg or config.PPOConfig(seed=seed)
    agent = PPOAgent(ppo_cfg)
    history: dict[str, list[float]] = {"mean_reward": [], "reached": []}

    for ep in range(episodes):
        env = MultiRobotNavEnv(scenario=scenario, seed=seed + ep)
        obs = env.reset()
        ep_reward = np.zeros(config.NUM_ROBOTS)

        while True:
            actions = np.zeros((config.NUM_ROBOTS, 2))
            raw_actions = []
            log_probs = []
            values = []
            for i in range(config.NUM_ROBOTS):
                obs_t = torch.as_tensor(obs[i], dtype=torch.float32)
                dist, value = agent.model(obs_t)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action).sum(-1)
                actions[i] = raw_action.detach().numpy() * config.ROBOT_MAX_SPEED
                raw_actions.append(raw_action.detach().numpy())
                log_probs.append(log_prob.detach().item())
                values.append(value.detach().item())

            next_obs, rewards, dones, info = env.step(actions)

            for i in range(config.NUM_ROBOTS):
                agent.store(
                    obs=obs[i],
                    action=np.asarray(raw_actions[i]),
                    log_prob=log_probs[i],
                    value=values[i],
                    reward=float(rewards[i]),
                    done=bool(dones[i]),
                )

            ep_reward += rewards
            obs = next_obs
            if info["episode_done"]:
                break

        agent.update()
        history["mean_reward"].append(float(ep_reward.mean()))
        history["reached"].append(int(env.reached.sum()))

        if verbose and (ep % max(1, episodes // 10) == 0 or ep == episodes - 1):
            print(f"[{scenario}] episode {ep + 1}/{episodes} | "
                  f"mean reward {ep_reward.mean():.2f} | reached {env.reached.sum()}/6")

    return agent, history
