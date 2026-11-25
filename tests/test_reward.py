"""Unit tests for the reward function."""

import numpy as np
import pytest

from navrl6.reward import compute_reward


def test_progress_positive_when_approaching():
    result = compute_reward(
        prev_dist=10.0, dist=9.0, reached=False, collided=False,
        clearance=2.0, separation=2.0, action=np.array([0.5, 0.0]),
    )
    assert result.breakdown["progress"] > 0


def test_progress_negative_when_receding():
    result = compute_reward(
        prev_dist=9.0, dist=10.0, reached=False, collided=False,
        clearance=2.0, separation=2.0, action=np.array([0.5, 0.0]),
    )
    assert result.breakdown["progress"] < 0


def test_goal_bonus_positive():
    result = compute_reward(
        prev_dist=0.8, dist=0.3, reached=True, collided=False,
        clearance=2.0, separation=2.0, action=np.zeros(2),
    )
    assert result.breakdown["goal_bonus"] > 0
    assert result.total > 0


def test_collision_penalty_negative():
    result = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=True,
        clearance=-0.2, separation=2.0, action=np.zeros(2),
    )
    assert result.breakdown["collision"] < 0
    assert result.total < 0


def test_clearance_penalty_inside_margin():
    outside = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=2.0, action=np.zeros(2),
    )
    inside = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=0.3, separation=2.0, action=np.zeros(2),
    )
    assert outside.breakdown["clearance"] == 0.0
    assert inside.breakdown["clearance"] < 0


def test_separation_penalty_for_close_robot():
    far = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=3.0, action=np.zeros(2),
    )
    close = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=0.2, action=np.zeros(2),
    )
    assert far.breakdown["separation"] == 0.0
    assert close.breakdown["separation"] < 0


def test_time_penalty_is_always_small_negative():
    result = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=3.0, action=np.zeros(2),
    )
    assert result.breakdown["time"] < 0


def test_control_penalty_grows_with_action():
    small = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=3.0, action=np.array([0.1, 0.0]),
    )
    large = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=3.0, action=np.array([1.5, 0.0]),
    )
    assert large.breakdown["control"] < small.breakdown["control"]


def test_breakdown_keys():
    result = compute_reward(
        prev_dist=5.0, dist=5.0, reached=False, collided=False,
        clearance=3.0, separation=3.0, action=np.zeros(2),
    )
    assert set(result.breakdown) == {
        "progress", "goal_bonus", "collision", "clearance",
        "separation", "time", "control",
    }
    assert "total" in result.as_dict()
