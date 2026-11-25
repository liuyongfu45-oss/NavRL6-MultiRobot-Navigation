"""Performance metrics for multi-robot navigation.

Per-robot episode record
------------------------
* ``reached`` -- goal reached before the episode ended;
* ``collided`` -- collision with obstacle / robot / wall;
* ``steps`` -- time-to-goal (decision steps);
* ``path_length`` -- travelled arc length (m);
* ``min_clearance`` -- smallest free space to any obstacle during the run (m);
* ``total_reward`` -- accumulated episode reward;
* ``spl`` -- Success weighted by Path Length.

Aggregate scenario metrics
--------------------------
* **Success rate** -- fraction of robots that reached their goal;
* **Collision rate** -- fraction of robots that collided;
* **Average steps / path length / clearance / reward**;
* **SPL** -- :math:`\\frac{1}{N}\\sum_i S_i \\frac{D_i}{\\max(P_i, D_i)}`,
  where :math:`D_i` is the straight-line start-to-goal distance and
  :math:`P_i` the travelled path length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, scenarios


@dataclass
class RobotEpisode:
    reached: bool
    collided: bool
    steps: int
    path_length: float
    min_clearance: float
    total_reward: float
    init_goal_distance: float


@dataclass
class ScenarioStats:
    scenario: str
    episodes: int
    robots: int
    success_rate: float
    collision_rate: float
    avg_steps: float
    avg_path_length: float
    avg_min_clearance: float
    avg_reward: float
    spl: float
    records: list[RobotEpisode] = field(default_factory=list)


def path_length(trajectory: list[np.ndarray]) -> float:
    """Arc length of a recorded trajectory."""
    if len(trajectory) < 2:
        return 0.0
    pts = np.asarray(trajectory)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def build_record(
    trajectory: list[np.ndarray],
    reached: bool,
    collided: bool,
    steps: int,
    min_clearance: float,
    total_reward: float,
    init_position: np.ndarray,
    goal: np.ndarray,
) -> RobotEpisode:
    init_dist = float(np.linalg.norm(goal - init_position))
    return RobotEpisode(
        reached=bool(reached),
        collided=bool(collided),
        steps=int(steps),
        path_length=path_length(trajectory),
        min_clearance=float(min_clearance),
        total_reward=float(total_reward),
        init_goal_distance=init_dist,
    )


def spl_value(records: list[RobotEpisode]) -> float:
    """Success weighted by Path Length (SPL)."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        if r.reached:
            total += r.init_goal_distance / max(r.path_length, r.init_goal_distance, 1e-6)
    return float(total / len(records))


def summarize(scenario: str, records: list[RobotEpisode]) -> ScenarioStats:
    n = len(records)
    if n == 0:
        raise ValueError("No records to summarise.")
    return ScenarioStats(
        scenario=scenario,
        episodes=max(1, n // config.NUM_ROBOTS),
        robots=n,
        success_rate=float(np.mean([r.reached for r in records])),
        collision_rate=float(np.mean([r.collided for r in records])),
        avg_steps=float(np.mean([r.steps for r in records])),
        avg_path_length=float(np.mean([r.path_length for r in records])),
        avg_min_clearance=float(np.mean([r.min_clearance for r in records])),
        avg_reward=float(np.mean([r.total_reward for r in records])),
        spl=spl_value(records),
        records=records,
    )


# --------------------------------------------------------------------------- #
# Table / CSV export
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "success_rate": "Success rate ↑",
    "collision_rate": "Collision rate ↓",
    "avg_steps": "Avg. steps ↓",
    "avg_path_length": "Avg. path length (m) ↓",
    "avg_min_clearance": "Avg. min. clearance (m) ↑",
    "avg_reward": "Avg. episode reward ↑",
    "spl": "SPL ↑",
}


def stats_to_markdown(stats: list[ScenarioStats]) -> str:
    """Render a comparison table for the given scenario summaries."""
    headers = ["Scenario"] + [METRIC_LABELS[k] for k in METRIC_LABELS]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for s in stats:
        row = [s.scenario.capitalize()]
        for k in METRIC_LABELS:
            row.append(f"{getattr(s, k):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def stats_to_csv(stats: list[ScenarioStats], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario"] + list(METRIC_LABELS.keys()))
        for s in stats:
            writer.writerow([s.scenario] + [f"{getattr(s, k):.4f}" for k in METRIC_LABELS])
