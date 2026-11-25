"""Command-line entry point.

Examples
--------
Train a PPO agent on the mixed scenario::

    python -m navrl6.run train --scenario mixed --episodes 200

Evaluate the baseline policy on all scenarios and write metric tables/figures::

    python -m navrl6.run evaluate --episodes 20 --results results

Record a reference video for one scenario::

    python -m navrl6.run record --scenario static --out videos/NavRL6_S1_Static.mp4

Run the full evaluation pipeline (metrics + figures)::

    python -m navrl6.run all --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .agent import PPOAgent
from .environment import MultiRobotNavEnv
from .evaluation import ReactivePolicy, evaluate_scenario, rollout
from .metrics import stats_to_csv, stats_to_markdown, summarize
from .scenarios import SCENARIOS
from .trainer import train
from .visualization import (
    plot_scenario_comparison,
    plot_trajectories,
    plot_training_curve,
    record_video,
)


def _load_policy(agent_path: str | None):
    if agent_path:
        agent = PPOAgent()
        agent.load(agent_path)
        return agent
    return ReactivePolicy()


def cmd_train(args: argparse.Namespace) -> None:
    agent, history = train(
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out))
    print(f"Checkpoint saved to {out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    results_dir = Path(args.results)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    policy = _load_policy(args.agent)
    all_stats = []
    for scenario in SCENARIOS:
        records = evaluate_scenario(
            scenario, policy, episodes=args.episodes, seed=args.seed
        )
        stats = summarize(scenario, records)
        all_stats.append(stats)
        print(f"[{scenario}] success={stats.success_rate:.2f} "
              f"collision={stats.collision_rate:.2f} SPL={stats.spl:.2f} "
              f"clearance={stats.avg_min_clearance:.2f}")

        # Trajectory snapshot from one representative episode.
        env = MultiRobotNavEnv(scenario=scenario, seed=args.seed)
        rollout(env, policy)
        plot_trajectories(env, scenario, figures_dir / f"trajectory_{scenario}.png")

    stats_to_csv(all_stats, results_dir / "metrics.csv")
    (results_dir / "metrics.md").write_text(
        stats_to_markdown(all_stats), encoding="utf-8"
    )
    plot_scenario_comparison(all_stats, figures_dir / "scenario_comparison.png")
    print(f"Metrics written to {results_dir}")


def cmd_record(args: argparse.Namespace) -> None:
    env = MultiRobotNavEnv(scenario=args.scenario, seed=args.seed)
    policy = _load_policy(args.agent)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    record_video(env, policy, args.out)
    print(f"Video saved to {args.out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_evaluate(args)
    # Short smoke training to produce a learning curve (CPU friendly).
    _, history = train(scenario="mixed", episodes=args.smoke_episodes, verbose=False)
    plot_training_curve(history, Path(args.results) / "figures" / "training_curve.png")
    print("Full pipeline finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NavRL6 multi-robot navigation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a PPO agent")
    p_train.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    p_train.add_argument("--episodes", type=int, default=200)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--out", default="results/ppo_navrl6.pt")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a policy on all scenarios")
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--results", default="results")
    p_eval.add_argument("--agent", default=None, help="optional PPO checkpoint")
    p_eval.set_defaults(func=cmd_evaluate)

    p_rec = sub.add_parser("record", help="record a navigation video")
    p_rec.add_argument("--scenario", choices=SCENARIOS, default="static")
    p_rec.add_argument("--seed", type=int, default=0)
    p_rec.add_argument("--agent", default=None)
    p_rec.add_argument("--out", default="videos/NavRL6.mp4")
    p_rec.set_defaults(func=cmd_record)

    p_all = sub.add_parser("all", help="evaluate + smoke training + figures")
    p_all.add_argument("--episodes", type=int, default=20)
    p_all.add_argument("--smoke-episodes", type=int, default=30)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--results", default="results")
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
