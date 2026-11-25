"""NavRL6: six-robot reinforcement-learning navigation benchmark.

Package contents
----------------
* :mod:`navrl6.config` -- world, reward and PPO hyper-parameters;
* :mod:`navrl6.environment` -- multi-robot navigation environment;
* :mod:`navrl6.obstacles` -- static / dynamic circular obstacles;
* :mod:`navrl6.reward` -- shaped reward function;
* :mod:`navrl6.networks` -- actor-critic MLP;
* :mod:`navrl6.agent` -- PPO agent;
* :mod:`navrl6.trainer` -- training loop;
* :mod:`navrl6.evaluation` -- rollout runner and potential-field baseline;
* :mod:`navrl6.metrics` -- performance metrics and table export;
* :mod:`navrl6.visualization` -- plots, trajectories and video recording.
"""

from . import config, scenarios
from .agent import PPOAgent
from .environment import ACT_DIM, OBS_DIM, MultiRobotNavEnv
from .evaluation import ReactivePolicy, evaluate_scenario, rollout
from .metrics import RobotEpisode, ScenarioStats, summarize

__version__ = "1.0.0"

__all__ = [
    "config",
    "scenarios",
    "PPOAgent",
    "MultiRobotNavEnv",
    "OBS_DIM",
    "ACT_DIM",
    "ReactivePolicy",
    "evaluate_scenario",
    "rollout",
    "RobotEpisode",
    "ScenarioStats",
    "summarize",
    "__version__",
]
