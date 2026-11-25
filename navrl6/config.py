"""Global configuration for the NavRL6 multi-robot navigation task.

All physical constants match the three reference simulation videos
(``NavRL6_S1_Static.mp4``, ``NavRL6_S2_Dynamic.mp4``, ``NavRL6_S3_Mixed.mp4``):

* 6 holonomic robots, 15 obstacles per scenario
* 2D square world of ``[-18, 18]`` metres
* time step ``dt = 0.10 s`` (title bar: ``Time 0.10s`` per step)
* three scenarios: ``static`` (15 static), ``dynamic`` (15 moving),
  ``mixed`` (8 static + 7 dynamic)
"""

from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
WORLD_SIZE: float = 18.0          # half-length of the square arena (m)
DT: float = 0.10                  # simulation step (s)
MAX_STEPS: int = 160              # episode horizon (steps)

# --------------------------------------------------------------------------- #
# Robots
# --------------------------------------------------------------------------- #
NUM_ROBOTS: int = 6
ROBOT_RADIUS: float = 0.35        # m
ROBOT_MAX_SPEED: float = 2.0      # m/s (norm of the velocity command)
GOAL_RADIUS: float = 0.55         # m, distance inside which goal is reached
SAFETY_MARGIN: float = 1.20       # m, clearance band around obstacles

# --------------------------------------------------------------------------- #
# Obstacles
# --------------------------------------------------------------------------- #
NUM_OBSTACLES: int = 15
N_STATIC_MIXED: int = 8           # static obstacles in the mixed scenario
N_DYNAMIC_MIXED: int = 7          # dynamic obstacles in the mixed scenario
STATIC_RADIUS: Tuple[float, float] = (0.70, 1.20)
DYNAMIC_RADIUS: Tuple[float, float] = (0.55, 0.95)
DYNAMIC_SPEED: float = 0.90       # m/s

# Reference colours taken from the video legend
ROBOT_COLORS: list = [
    "#1f77b4",  # Robot 1 - blue
    "#d62728",  # Robot 2 - red
    "#2ca02c",  # Robot 3 - green
    "#9467bd",  # Robot 4 - purple
    "#17becf",  # Robot 5 - cyan
    "#e377c2",  # Robot 6 - pink
]
STATIC_COLOR: str = "#9e9e9e"
DYNAMIC_COLOR: str = "#ff7f0e"

SCENARIO_TITLES: dict = {
    "static": "NavRL6 | Multi-Robot 15 Static Obstacles",
    "dynamic": "NavRL6 | Multi-Robot 15 Dynamic Obstacles",
    "mixed": "NavRL6 | Multi-Robot Mixed Dynamic & Static Obstacles",
}


# --------------------------------------------------------------------------- #
# Reward shaping
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    """Weights of every term in the multi-robot reward function.

    Signs are fixed by design: goal-progress / reach terms are positive,
    collision / intrusion / control terms are negative. See the reward table
    in ``README.md`` (Section 2).
    """

    progress_weight: float = 1.50      # goal-distance reduction  (signed)
    goal_bonus: float = 100.0         # one-off bonus on reaching the goal
    collision_penalty: float = -10.0  # obstacle / robot / wall collision
    clearance_penalty: float = -0.80   # linear penalty inside safety margin
    separation_penalty: float = -0.50  # penalty when another robot is close
    time_penalty: float = -0.01        # per-step cost (efficiency)
    control_penalty: float = -0.05     # quadratic action effort


# --------------------------------------------------------------------------- #
# Neural network / PPO hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class PPOConfig:
    """Hyper-parameters of the shared actor-critic MLP trained with PPO.

    The policy is homogeneous: one shared network controls all six robots.
    """

    # Architecture
    hidden_sizes: Tuple[int, ...] = (128, 128)
    activation: str = "relu"           # ReLU in both actor and critic
    init_log_std: float = -0.5         # initial log-std of the Gaussian policy

    # Optimisation
    lr_actor: float = 3.0e-4
    lr_critic: float = 1.0e-3
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE lambda
    clip_ratio: float = 0.20          # PPO clipped objective epsilon
    ppo_epochs: int = 10               # optimisation epochs per rollout
    batch_size: int = 64
    rollout_steps: int = 2048         # transitions collected per update
    entropy_coef: float = 0.01         # exploration entropy bonus
    value_coef: float = 0.5            # critic loss weight
    max_grad_norm: float = 0.5         # gradient clipping

    # Episode / environment
    max_steps: int = MAX_STEPS
    seed: int = 0
