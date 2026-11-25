"""Unit tests for navigation metrics."""

import numpy as np

from navrl6.metrics import (
    RobotEpisode,
    path_length,
    spl_value,
    stats_to_csv,
    stats_to_markdown,
    summarize,
)


def _record(reached=True, collided=False, path_length=10.0, steps=50,
             clearance=1.0, reward=5.0, init_dist=10.0) -> RobotEpisode:
    return RobotEpisode(
        reached=reached, collided=collided, steps=steps,
        path_length=path_length, min_clearance=clearance,
        total_reward=reward, init_goal_distance=init_dist,
    )


def test_path_length_straight_line():
    pts = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
    assert abs(path_length(pts) - 5.0) < 1e-8


def test_path_length_empty():
    assert path_length([np.array([0.0, 0.0])]) == 0.0


def test_spl_perfect_path():
    records = [_record(path_length=10.0, init_dist=10.0)]
    assert abs(spl_value(records) - 1.0) < 1e-8


def test_spl_longer_path_below_one():
    records = [_record(path_length=20.0, init_dist=10.0)]
    assert 0 < spl_value(records) < 1


def test_spl_failure_contributes_zero():
    records = [_record(reached=False, path_length=20.0)]
    assert spl_value(records) == 0.0


def test_summarize_aggregates():
    records = [
        _record(reached=True, reward=8.0),
        _record(reached=False, collided=True, reward=-5.0),
    ]
    stats = summarize("static", records)
    assert stats.robots == 2
    assert abs(stats.success_rate - 0.5) < 1e-8
    assert abs(stats.collision_rate - 0.5) < 1e-8
    assert stats.avg_reward == 1.5


def test_markdown_table_contains_scenario():
    stats = summarize("static", [_record()])
    table = stats_to_markdown([stats])
    assert "Static" in table
    assert "Success rate" in table


def test_csv_export(tmp_path):
    records = [_record()]
    stats = summarize("dynamic", records)
    out = tmp_path / "metrics.csv"
    stats_to_csv([stats], out)
    text = out.read_text(encoding="utf-8")
    assert "dynamic" in text
    assert "success_rate" in text
